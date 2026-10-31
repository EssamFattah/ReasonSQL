"""End-to-end check of the reporting layer on synthetic runs in the v3 layout (no MLX needed)."""
import sys

import numpy as np

from src import fingerprint as fpr
from src.common import write_json, write_jsonl
from src.evaluate_sql import summarize_preds

N = 200


def fake_eval(d, acc, seed, with_ts=True):
    rng = np.random.default_rng(seed)
    preds = []
    for i in range(N):
        ok = bool(rng.random() < acc)
        preds.append({"id": f"dev_{i}", "db_id": f"db{i % 10}", "difficulty": ["easy", "medium", "hard"][i % 3],
                      "correct": ok, "ts_correct": ok and rng.random() < .95 if with_ts else None,
                      "valid": ok or i % 2 == 0, "truncated": False, "pred_sql": "SELECT 1",
                      "category": "correct" if ok else "wrong_values", "struct_diff": [] if ok else ["joins"],
                      "gen_tokens": 300, "ttft_s": .2, "total_s": 5.0, "gen_tps": 60.0})
    write_jsonl(d / "sql_preds.jsonl", preds)
    write_json(d / "sql_eval.json", summarize_preds(preds))


def fake_forgetting(d, acc, seed):
    rng = np.random.default_rng(100 + seed)
    rows = [{"id": f"mmlu_{i}", "correct": bool(rng.random() < acc)} for i in range(N)]
    rows += [{"id": f"gsm8k_{i}", "correct": bool(rng.random() < acc)} for i in range(N // 2)]
    write_jsonl(d / "forgetting_preds.jsonl", rows)
    write_json(d / "forgetting.json", {"mmlu_acc": acc, "gsm8k_acc": acc})


def make_runs(runs):
    fpr.save(runs / "base_student" / "config.json", {"kind": "baseline", "key": "base_student"})
    for e in range(3):  # inference-only replicates
        fake_eval(runs / "base_student" / f"eval_think_e{e}", .45, e)
    fake_eval(runs / "base_student" / "eval_nothink", .35, 9)
    fake_forgetting(runs / "base_student", .55, 0)
    fpr.save(runs / "teacher" / "config.json", {"kind": "baseline", "key": "teacher"})
    fake_eval(runs / "teacher" / "eval_think_e0", .75, 2)

    specs = [("reasoning", "all", 16, .62), ("sql_only", "all", 16, .52), ("reasoning_unverified", "all", 16, .58),
             ("sql_unverified", "all", 16, .5), ("reasoning", 250, 16, .5), ("reasoning", 1000, 16, .58),
             ("reasoning", "all", 4, .57), ("reasoning", "all", 64, .61)]
    for mode, n, r, acc in specs:
        for s in ([0, 1, 2] if n == "all" and r == 16 else [0]):
            key = f"{mode}_n{n}_r{r}"
            d = runs / f"{key}_s{s}"
            fpr.save(d / "config.json", {"kind": "lora", "key": key, "mode": mode, "rank": r, "seed": s,
                                         "data_meta": {"n_train": 2000 if n == "all" else n,
                                                       "funnel": {"usable_verified": 2100, "train": 2000}}})
            fake_eval(d / "eval_think_e0", acc, 10 * s + r)
            fake_eval(d / "eval_nothink", acc - .05, 10 * s + r + 1)
            fake_forgetting(d, .53, s)
            write_json(d / "adapter" / "train_summary.json",
                       {"iters": 100, "wall_time_s": 3600, "peak_mem_gb": 7.0, "final_val_loss": .5,
                        "best_val_loss": .48})
    write_json(runs / "reasoning_nall_r16_s0" / "checkpoint_curve.json",
               {"points": [{"iter": 0, "ex": .35, "mmlu": .55}, {"iter": 50, "ex": .5, "mmlu": .54}]})


def test_summarize_end_to_end(tmp_path, monkeypatch):
    runs, out = tmp_path / "runs", tmp_path / "results"
    make_runs(runs)
    write_json(out / "benchmark.json", {"results": [
        {"name": "teacher", "n_prompts": 100, "thinking": True, "disk_gb": 4.6, "load_s": 3.0, "peak_mem_gb": 6.1,
         "median_ttft_s": .9, "mean_gen_tps": 25, "median_latency_s": 30, "ex_on_bench_prompts": .7,
         "latency_by_output_tokens": {"<256 tok": {"n": 10, "median_latency_s": 4.0}}}]})

    from src import summarize
    monkeypatch.setattr(sys, "argv", ["summarize", "--runs_dir", str(runs), "--out_dir", str(out)])
    summarize.main()
    report = (out / "report.md").read_text()
    for section in ("Execution accuracy", "Key comparisons", "Interaction (nothink)", "Gap closed",
                    "Accuracy by query difficulty", "Catastrophic forgetting", "Training cost", "Deployment benchmark",
                    "db:[", "p_db=", "ⁱ", "Dataset funnel"):
        assert section in report, section
    for fig in ("ex_vs_data_size.png", "ex_vs_rank.png", "trace_ablation.png", "forgetting.png",
                "curve_reasoning_nall_r16.png"):
        assert (out / fig).exists(), fig
    assert (out / "ex_by_db.csv").exists() and (out / "summary.json").exists()

    monkeypatch.setattr(sys, "argv", ["summarize", "--runs_dir", str(runs), "--out_dir", str(out / "ts"),
                                      "--metric", "ts"])
    summarize.main()
    assert "test-suite accuracy" in (out / "ts" / "report.md").read_text()
