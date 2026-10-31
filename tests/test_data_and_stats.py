import sys

import numpy as np
import pytest

from src import fingerprint as fpr
from src.build_sft import MODES, build
from src.common import is_usable, read_jsonl, write_json, write_jsonl
from src.eval_forgetting import parse_final_number
from src.stats import accuracy_ci, bootstrap_ci, contrast, paired_test, seed_average
from src.teacher_stats import compute
from src.train import is_trained, parse_log


# ---------------------------------------------------------------- 2x2 dataset construction
def _build_all(tmp_path, pool, traces, n_train=-1):
    return {m: build(traces, pool, tmp_path / f"{m}_{n_train}", m, n_train=n_train, valid_size=4) for m in MODES}


def test_modes_share_questions_and_validation(tmp_path, pool_and_traces):
    pool, traces, _ = pool_and_traces
    metas = _build_all(tmp_path, pool, traces)
    assert len({m["train_ids_sha"] for m in metas.values()}) == 1
    assert len({m["valid_ids_sha"] for m in metas.values()}) == 1
    assert metas["reasoning"]["funnel"] == {"questions_with_traces": 30, "teacher_ever_correct": 20,
                                            "usable_verified": 20, "valid": 4, "train": 16}


def test_verified_vs_unverified_targets(tmp_path, pool_and_traces):
    pool, traces, _ = pool_and_traces
    metas = _build_all(tmp_path, pool, traces)
    assert metas["reasoning"]["frac_targets_correct"] == 1.0
    assert metas["sql_only"]["frac_targets_correct"] == 1.0
    u = metas["reasoning_unverified"]  # first attempt: wrong exactly for questions solved at attempt 1
    assert 0 < u["frac_targets_correct"] < 1
    assert abs(u["frac_targets_correct"] + u["frac_targets_differ_from_verified"] - 1) < 1e-9


def test_sql_only_has_empty_think_and_same_sql(tmp_path, pool_and_traces):
    pool, traces, _ = pool_and_traces
    _build_all(tmp_path, pool, traces)
    r = read_jsonl(tmp_path / "reasoning_-1" / "train.jsonl")
    s = read_jsonl(tmp_path / "sql_only_-1" / "train.jsonl")
    for a, b in zip(r, s):
        ra, sb = a["messages"][-1]["content"], b["messages"][-1]["content"]
        assert sb.startswith("<think>\n\n</think>")
        assert ra.split("```sql")[1] == sb.split("```sql")[1]
        assert a["messages"][:-1] == b["messages"][:-1]
    vs = read_jsonl(tmp_path / "sql_only_-1" / "valid.jsonl")
    assert all(v["messages"][-1]["content"].startswith("<think>\n\n</think>") for v in vs)


def test_subsets_are_nested(tmp_path, pool_and_traces):
    pool, traces, _ = pool_and_traces
    build(traces, pool, tmp_path / "small", "reasoning", n_train=5, valid_size=4)
    build(traces, pool, tmp_path / "big", "reasoning", n_train=10, valid_size=4)
    small = read_jsonl(tmp_path / "small" / "train.jsonl")
    big = read_jsonl(tmp_path / "big" / "train.jsonl")
    assert small == big[:5]
    assert read_jsonl(tmp_path / "small" / "valid.jsonl") == read_jsonl(tmp_path / "big" / "valid.jsonl")


def test_length_and_truncation_filters_use_tokens(tmp_path, pool_and_traces):
    pool, traces_path, traces = pool_and_traces
    traces = [dict(t) for t in traces]
    for t in traces:
        if t["id"] == "train_00":
            t["example_tokens"] = 9999       # correct but would be truncated by mlx_lm
        if t["id"] == "train_03":
            t["truncated"] = True
    write_jsonl(traces_path, traces)
    m = build(traces_path, pool, tmp_path / "f", "reasoning", valid_size=0, max_example_tokens=4096)
    assert m["funnel"]["teacher_ever_correct"] == 20 and m["funnel"]["usable_verified"] == 18
    assert m["example_tokens"]["max"] <= 4096


def test_correct_but_too_long_falls_through_to_later_attempt(tmp_path, pool_and_traces):
    """Teacher keeps sampling after a correct-but-unusable attempt; build_sft must pick the usable one."""
    pool, traces_path, traces = pool_and_traces
    traces = [dict(t) for t in traces if t["id"] != "train_00"]
    traces += [{**t, "id": "train_00", "sample_idx": k, "pred_sql": "SELECT name FROM item WHERE price > 0 "
                "ORDER BY name", "correct": True, "category": "correct", "example_tokens": n, "reasoning": f"r{k}"}
               for k, n in ((0, 9999), (1, 800)) for t in traces[:1]]
    write_jsonl(traces_path, traces)
    for mode in ("reasoning", "reasoning_unverified"):
        build(traces_path, pool, tmp_path / mode, mode, valid_size=0)
        ex = [e for e in read_jsonl(tmp_path / mode / "train.jsonl") if e["messages"][1]["content"].endswith("q0")]
        assert len(ex) == 1 and "<think>\nr1\n</think>" in ex[0]["messages"][-1]["content"]


def test_dataset_fingerprint_refuses_stale_rebuild(tmp_path, pool_and_traces):
    pool, traces_path, traces = pool_and_traces
    out = tmp_path / "d"
    m1 = build(traces_path, pool, out, "reasoning", valid_size=4)
    assert build(traces_path, pool, out, "reasoning", valid_size=4) == m1  # identical spec -> reused
    write_jsonl(traces_path, traces + [{**traces[0], "sample_idx": 9}])  # teacher generated more
    with pytest.raises(fpr.IncompatibleRun, match="traces_sha"):
        build(traces_path, pool, out, "reasoning", valid_size=4)
    with pytest.raises(fpr.IncompatibleRun, match="valid_size"):
        build(traces_path, pool, tmp_path / "d2", "reasoning", valid_size=4) and \
            build(traces_path, pool, tmp_path / "d2", "reasoning", valid_size=5)


def test_missing_example_tokens_is_an_error():
    with pytest.raises(KeyError, match="example_tokens"):
        is_usable({"id": "x", "sample_idx": 0, "correct": True, "pred_sql": "SELECT 1", "truncated": False,
                   "reasoning": ""})


# ---------------------------------------------------------------- teacher stats
def test_teacher_pass_at_k(pool_and_traces):
    _, _, traces = pool_and_traces
    s = compute(traces, k=3)
    assert s["funnel"]["questions_finished"] == 30 and s["funnel"]["usable_verified"] == 20
    assert abs(s["pass_at"]["pass@1"] - 1 / 3) < 1e-9
    assert abs(s["pass_at"]["pass@2"] - 2 / 3) < 1e-9
    assert abs(s["pass_at"]["pass@3"] - 2 / 3) < 1e-9
    assert abs(s["never_usable"] - 1 / 3) < 1e-9


def test_teacher_stats_counts_correct_but_unusable(pool_and_traces):
    _, _, traces = pool_and_traces
    traces = [dict(t) for t in traces]
    traces[0]["example_tokens"] = 9999  # train_00 solved at attempt 0, but unusable -> not finished
    s = compute(traces, k=3)
    assert s["correct_but_unusable_attempts"] == {"n": 1, "truncated": 0, "too_long": 1}
    assert s["funnel"]["questions_finished"] == 29


def test_teacher_stats_ignores_unfinished(pool_and_traces):
    _, _, traces = pool_and_traces
    partial = [t for t in traces if not (t["id"] == "train_02" and t["sample_idx"] > 0)]
    assert compute(partial, k=3)["funnel"]["questions_finished"] == 29


# ---------------------------------------------------------------- statistics
def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).random(500) < 0.6
    m, lo, hi = bootstrap_ci(v)
    assert lo < m < hi and hi - lo < 0.12


def test_paired_test_detects_real_difference():
    rng = np.random.default_rng(1)
    base = {f"q{i}": float(rng.random() < 0.5) for i in range(800)}
    better = {k: 1.0 if (v == 1.0 or rng.random() < 0.3) else 0.0 for k, v in base.items()}
    c = paired_test(better, base)
    assert c["diff"] > 0 and c["ci"][0] > 0 and c["p_value"] < 0.01 and c["b_better"] == 0
    assert paired_test(base, base)["p_value"] == 1.0


def test_permutation_p_is_calibrated_under_null():
    """Two systems with identical accuracy: p-values should be roughly uniform (few false positives)."""
    rng = np.random.default_rng(2)
    ps = []
    for _ in range(60):
        a = {f"q{i}": float(rng.random() < .6) for i in range(300)}
        b = {f"q{i}": float(rng.random() < .6) for i in range(300)}
        ps.append(paired_test(a, b, n_boot=2000)["p_value"])
    assert 0.02 <= np.mean(np.array(ps) < 0.05) <= 0.15
    assert 0.35 < np.mean(ps) < 0.65


def test_clustered_ci_wider_with_cluster_effects():
    rng = np.random.default_rng(3)
    db_acc = {f"db{j}": rng.uniform(.1, .9) for j in range(20)}  # strong per-schema difficulty
    clusters = {f"q{i}": f"db{i % 20}" for i in range(1000)}
    per_q = {q: float(rng.random() < db_acc[c]) for q, c in clusters.items()}
    ci = accuracy_ci(per_q, clusters)
    assert (ci["ci_clustered"][1] - ci["ci_clustered"][0]) > 1.5 * (ci["ci"][1] - ci["ci"][0])
    assert ci["n_clusters"] == 20


def test_interaction_contrast():
    ids = [f"q{i}" for i in range(400)]
    one, zero = {i: 1.0 for i in ids}, {i: 0.0 for i in ids}
    half = {i: float(n % 2) for n, i in enumerate(ids)}
    # reasoning helps by 1.0 when verified, by 0.5 when unverified -> interaction +0.5
    c = contrast([one, zero, half, zero], [1, -1, -1, 1])
    assert abs(c["diff"] - 0.5) < 1e-9 and c["p_value"] < 0.01


def test_seed_average_uses_common_questions():
    assert seed_average([{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0, "c": 1.0}]) == {"a": 0.5, "b": 0.0}


# ---------------------------------------------------------------- fingerprints & training guard
def test_fingerprint_check(tmp_path):
    p = tmp_path / "spec.json"
    assert fpr.check(p, {"lr": 1e-5}, "x") == (fpr.fingerprint({"lr": 1e-5}), False)
    fpr.save(p, {"lr": 1e-5})
    assert fpr.check(p, {"lr": 1e-5}, "x")[1] is True
    with pytest.raises(fpr.IncompatibleRun, match=r"lr: 1e-05 -> 5e-05"):
        fpr.check(p, {"lr": 5e-5}, "x")


def _train_argv(data, adapter, lr):
    return ["train", "--data_dir", str(data), "--adapter_path", str(adapter), "--lr", str(lr)]


def test_train_refuses_finished_adapter_with_other_hyperparameters(tmp_path, pool_and_traces, monkeypatch):
    from src import train

    pool, traces, _ = pool_and_traces
    data, adapter = tmp_path / "data", tmp_path / "adapter"
    build(traces, pool, data, "reasoning", valid_size=4)
    # simulate a finished run trained with lr=2e-5
    monkeypatch.setattr(sys, "argv", _train_argv(data, adapter, 2e-5))
    args = train.argparse.ArgumentParser()
    adapter.mkdir()
    import types

    ns = types.SimpleNamespace(data_dir=str(data), adapter_path=str(adapter), model="Qwen/Qwen3-1.7B", rank=16,
                               scale=20.0, dropout=0.05, num_layers=-1, epochs=2.0, iters=None, batch_size=2, lr=2e-5,
                               warmup_frac=0.05, max_seq_length=4096, val_batches=25, grad_checkpoint=True, seed=0)
    cfg = train.make_config(ns)
    spec = train.training_spec(cfg, ns.model, data, fpr.read_json(data / "spec.json"))
    fpr.save(adapter / "spec.json", spec)
    (adapter / "DONE").write_text("ok")
    del args

    train.main()  # same hyperparameters -> skip quietly (no subprocess launched)
    monkeypatch.setattr(sys, "argv", _train_argv(data, adapter, 5e-5))
    with pytest.raises(fpr.IncompatibleRun, match="learning_rate"):
        train.main()


def test_train_refuses_sequence_length_below_dataset_limit(tmp_path, pool_and_traces, monkeypatch):
    from src import train

    pool, traces, _ = pool_and_traces
    build(traces, pool, tmp_path / "data", "reasoning", valid_size=4, max_example_tokens=4096)
    monkeypatch.setattr(sys, "argv", _train_argv(tmp_path / "data", tmp_path / "a", 2e-5) + ["--max_seq_length", "2048"])
    with pytest.raises(SystemExit, match="truncate"):
        train.main()


def test_done_marker(tmp_path):
    a = tmp_path / "adapter"
    a.mkdir()
    (a / "adapters.safetensors").write_text("partial")
    assert not is_trained(a)
    (a / "DONE").write_text("ok")
    assert is_trained(a)


def test_parse_train_log():
    log = ("Iter 1: Val loss 2.500, Val took 3.1s\n"
           "Iter 10: Train loss 2.100, Learning Rate 1.000e-05, It/sec 0.5, Tokens/sec 400.0, "
           "Trained Tokens 8000, Peak mem 6.200 GB\n"
           "Iter 20: Train loss 1.500, Learning Rate 1.000e-05, It/sec 0.5, Tokens/sec 420.0, "
           "Trained Tokens 16000, Peak mem 6.400 GB\n"
           "Iter 20: Val loss 1.700, Val took 3.0s\n")
    s = parse_log(log)
    assert s["final_train_loss"] == 1.5 and s["best_val_iter"] == 20 and s["peak_mem_gb"] == 6.4


# ---------------------------------------------------------------- orchestration & misc
def test_ablation_plan_separates_training_and_eval_seeds():
    from src.run_ablations import PROFILES, lora_specs

    cfg = {**PROFILES["full"], "rank_n": -1}
    specs = lora_specs(cfg)
    assert len(specs) == 20
    headline = [s for s in specs if s["group"] == "traces"]
    assert {s["seed"] for s in headline} == {0, 1, 2} and all(s["eval_limit"] == -1 for s in headline)
    assert all(s["eval_limit"] == 300 for s in specs if s["key"] not in {h["key"] for h in headline})
    assert cfg["eval_seeds"] == [0]  # every model shares the same inference seed


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    from src import run_ablations

    monkeypatch.setattr(sys, "argv", ["x", "--profile", "pilot", "--dry_run", "--runs_dir", str(tmp_path / "r")])
    run_ablations.main()
    out = capsys.readouterr().out
    assert "--eval_seed 0" in out and "base_student_s" not in out


def test_full_plan_teacher_gets_one_thinking_eval(tmp_path, monkeypatch, capsys):
    from src import run_ablations

    monkeypatch.setattr(sys, "argv", ["x", "--profile", "full", "--dry_run", "--runs_dir", str(tmp_path / "r")])
    run_ablations.main()
    lines = [l for l in capsys.readouterr().out.splitlines() if "src.evaluate_sql" in l]
    teacher_think = [l for l in lines if "teacher/eval_think" in l]
    base_think = [l for l in lines if "base_student/eval_think" in l]
    assert len(teacher_think) == 1 and len(base_think) == 3 and len(lines) == 46
    assert not (tmp_path / "r").exists()


def test_parse_final_number():
    assert parse_final_number("so 3 + 4 = 7\nAnswer: 1,234") == (1234.0, True)
    assert parse_final_number("Answer: $18.50") == (18.5, True)
    assert parse_final_number("the total is 42.") == (42.0, False)
    assert parse_final_number("no numbers") == (None, False)


def test_write_json_atomic(tmp_path):
    write_json(tmp_path / "a.json", {"x": 1})
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------- v3.1: identities & artifacts
def test_model_identity_pins_revision_and_hashes_local_weights(tmp_path):
    assert fpr.model_identity("Qwen/Qwen3-1.7B") == {"repo": "Qwen/Qwen3-1.7B", "requested_revision": None,
                                                      "revision": "test-revision"}
    assert fpr.model_identity("Qwen/Qwen3-1.7B@abc123")["revision"] == "abc123"
    m = tmp_path / "fused"
    m.mkdir()
    (m / "model.safetensors").write_bytes(b"weights-v1")
    a = fpr.model_identity(str(m))
    (m / "model.safetensors").write_bytes(b"weights-v2")
    assert fpr.model_identity(str(m))["weights_sha"] != a["weights_sha"]


def test_test_suite_identity_tracks_db_contents(tmp_path):
    (tmp_path / "shop").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "shop" / "shop.sqlite").write_bytes(b"v1")
    (tmp_path / "other" / "o.sqlite").write_bytes(b"x")
    before = fpr.sqlite_dir_identity(tmp_path, {"shop"})
    (tmp_path / "other" / "o.sqlite").write_bytes(b"y")       # unrelated DB: no effect
    assert fpr.sqlite_dir_identity(tmp_path, {"shop"}) == before
    (tmp_path / "shop" / "shop.sqlite").write_bytes(b"v2")    # same path, new contents
    assert fpr.sqlite_dir_identity(tmp_path, {"shop"}) != before
    assert fpr.sqlite_dir_identity(None) is None


def test_benchmark_artifacts_rebuilt_when_adapter_changes(tmp_path):
    from src.benchmark import artifact_is_current, ensure_artifact

    built = []

    def fake_build(cmd):  # stands in for mlx_lm.fuse / convert
        out = tmp_path / "fused"
        out.mkdir()
        (out / "config.json").write_text("{}")
        built.append(cmd[-1])

    path = tmp_path / "fused"
    spec_a = {"adapter": "aaa", "adapter_weights_sha": "1"}
    spec_b = {"adapter": "bbb", "adapter_weights_sha": "2"}
    ensure_artifact(path, spec_a, ["fuse", "A"], run=fake_build)
    ensure_artifact(path, spec_a, ["fuse", "A"], run=fake_build)   # identical inputs -> reused
    assert built == ["A"] and artifact_is_current(path, spec_a)
    ensure_artifact(path, spec_b, ["fuse", "B"], run=fake_build)   # new adapter -> rebuilt, never reused
    assert built == ["A", "B"] and artifact_is_current(path, spec_b) and not artifact_is_current(path, spec_a)
    (path / "artifact_spec.json").unlink()                          # v3-era artifact without a spec
    ensure_artifact(path, spec_b, ["fuse", "B2"], run=fake_build)
    assert built[-1] == "B2"


# ---------------------------------------------------------------- v3.2: pinning is enforced, adapter weights hashed
def test_loader_receives_the_pinned_snapshot(monkeypatch, tmp_path):
    """The path handed to mlx_lm.load is the snapshot whose commit model_identity records."""
    import types

    from src.common import load_model

    calls = []
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(
        load=lambda path, adapter_path=None: calls.append((path, adapter_path)) or ("model", "tok")))
    load_model("Qwen/Qwen3-1.7B", adapter_path="a")
    load_model("Qwen/Qwen3-8B@deadbeef")
    local = tmp_path / "fused"
    local.mkdir()
    load_model(str(local))
    assert calls[0] == ("/fake/hf/Qwen/Qwen3-1.7B/snapshots/test-revision", "a")
    assert calls[1][0].endswith("/snapshots/deadbeef")
    assert calls[2][0] == str(local)
    for model, (path, _) in zip(["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B@deadbeef"], calls):
        assert fpr.model_identity(model)["revision"] == path.rsplit("/", 1)[-1]


def test_unresolvable_model_fails_loudly(monkeypatch):
    monkeypatch.setattr(fpr, "resolve_hf_snapshot", lambda m: (None, None))
    with pytest.raises(RuntimeError, match="could not resolve"):
        fpr.resolve_model_path("org/missing-model")


def test_adapter_identity_changes_with_weights_even_if_spec_unchanged(tmp_path):
    a = tmp_path / "adapter"
    a.mkdir()
    fpr.save(a / "spec.json", {"rank": 16})
    (a / "adapter_config.json").write_text("{}")
    (a / "adapters.safetensors").write_bytes(b"w1")
    before = fpr.adapter_identity(str(a))
    (a / "adapters.safetensors").write_bytes(b"w2")  # e.g. overwritten by another training run
    after = fpr.adapter_identity(str(a))
    assert before["spec"] == after["spec"] and before["weights_sha"] != after["weights_sha"]
    assert fpr.fingerprint(before) != fpr.fingerprint(after)
    assert fpr.adapter_identity(None) is None


def test_sql_eval_refuses_when_adapter_weights_change(tmp_path, monkeypatch):
    from src import evaluate_sql

    a = tmp_path / "adapter"
    a.mkdir()
    fpr.save(a / "spec.json", {"rank": 16})
    (a / "adapters.safetensors").write_bytes(b"w1")
    data = tmp_path / "dev.jsonl"
    write_jsonl(data, [{"id": "dev_0", "db_id": "shop"}])
    out = tmp_path / "eval"
    argv = ["x", "--adapter_path", str(a), "--data", str(data), "--out_dir", str(out), "--no-thinking"]
    monkeypatch.setattr(evaluate_sql, "load_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop")))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="stop"):  # writes spec.json, then would load the model
        evaluate_sql.main()
    (a / "adapters.safetensors").write_bytes(b"w2")
    with pytest.raises(fpr.IncompatibleRun, match="weights_sha"):
        evaluate_sql.main()
