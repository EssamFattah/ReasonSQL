"""Execution accuracy (EX) on Spider dev, with failure categories, difficulty/database breakdowns,
latency/throughput and peak memory. Works for the base student, a LoRA adapter, the teacher, or a
fused/quantised model directory.

Seeds: --eval_seed controls inference sampling only, independently of how a model was trained.
run_ablations uses the SAME eval seeds for every model, so training seeds vary training only.

Optional --test_suite_dir: also score against the Spider test-suite database variants
(<dir>/<db_id>/*.sqlite). A prediction is test-suite-correct only if it matches gold on the
original DB and on every variant, which removes most single-instance false positives. The
official predictions/gold files are also exported for the reference test-suite-sql-eval script.

Resumable and fingerprinted (out_dir/spec.json): changing the model, adapter, protocol, eval seed,
limit or data refuses to reuse existing predictions.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import fingerprint as fpr
from src.common import (SYSTEM_PROMPT, append_jsonl, build_messages, chat_prompt, check_sql, difficulty,
                        generate_text, item_seed, load_model, peak_memory_gb, read_json, read_jsonl,
                        reset_peak_memory, sampling_for, score_generation, set_seed, sql_features, sql_one_line, tally,
                        write_json, write_jsonl)
from src.env_info import env_info


model_identity = fpr.model_identity  # re-exported (used by eval_forgetting)
adapter_identity = fpr.adapter_identity


def test_suite_correct(ts_dir: Path, row: dict, pred_sql: str, base_correct: bool):
    variants = sorted((ts_dir / row["db_id"]).glob("*.sqlite"))
    if not variants:
        return None
    return base_correct and all(check_sql(str(v), row["gold_sql"], pred_sql)["correct"] for v in variants)


def summarize_preds(preds: list[dict]) -> dict:
    def ex_by(key):
        groups = defaultdict(list)
        for p in preds:
            groups[p[key]].append(p["correct"])
        return {k: {"n": len(v), "ex": float(np.mean(v))} for k, v in sorted(groups.items())}

    ts = [p["ts_correct"] for p in preds if p.get("ts_correct") is not None]
    return {
        "n": len(preds),
        "execution_accuracy": float(np.mean([p["correct"] for p in preds])),
        "test_suite_accuracy": float(np.mean(ts)) if ts else None,
        "test_suite_coverage": len(ts) / len(preds) if preds else 0,
        "valid_sql_rate": float(np.mean([p["valid"] for p in preds])),
        "truncation_rate": float(np.mean([p["truncated"] for p in preds])),
        "avg_gen_tokens": float(np.mean([p["gen_tokens"] or 0 for p in preds])),
        "median_latency_s": float(np.median([p["total_s"] for p in preds])),
        "median_ttft_s": float(np.median([p["ttft_s"] or 0 for p in preds])),
        "avg_gen_tps": float(np.mean([p["gen_tps"] or 0 for p in preds])),
        "failure_categories": tally(p["category"] for p in preds),
        "wrong_answer_structural_diffs": tally(s for p in preds if not p["correct"] for s in p.get("struct_diff", [])),
        "ex_by_difficulty": ex_by("difficulty"),
        "ex_by_db": ex_by("db_id"),
    }


def export_official(out: Path, preds: list[dict]) -> None:
    """Files for github.com/taoyds/test-suite-sql-eval (one query per line, dev order)."""
    one_line = lambda s: sql_one_line(s) or "SELECT 1"  # noqa: E731  (literals preserved; empty -> wrong query)
    d = out / "official"
    d.mkdir(parents=True, exist_ok=True)
    (d / "pred.sql").write_text("\n".join(one_line(p["pred_sql"]) for p in preds) + "\n")
    (d / "gold.sql").write_text("\n".join(f"{one_line(p['gold_sql'])}\t{p['db_id']}" for p in preds) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--data", default="data/processed/dev.jsonl")
    ap.add_argument("--limit", type=int, default=-1, help="dev is pre-shuffled; -1 = full 1034")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=None, help="override the protocol's temperature")
    ap.add_argument("--eval_seed", type=int, default=0, help="inference sampling seed (not the training seed)")
    ap.add_argument("--test_suite_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    sampling = sampling_for(args.thinking)
    if args.temp is not None:
        sampling["temp"] = args.temp
    rows = read_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    spec = {"model": model_identity(args.model), "adapter": adapter_identity(args.adapter_path),
            "thinking": args.thinking, "sampling": sampling, "max_tokens": args.max_tokens,
            "eval_seed": args.eval_seed if sampling["temp"] > 0 else None,  # greedy ignores the seed
            "limit": args.limit, "data_sha": fpr.file_sha(args.data), "system_prompt": SYSTEM_PROMPT,
            # content hash of the test-suite DBs this eval touches, not just the directory name
            "test_suite": fpr.sqlite_dir_identity(args.test_suite_dir, {r["db_id"] for r in rows})}
    _, existed = fpr.check(out / "spec.json", spec, "sql eval")
    if existed and (out / "sql_eval.json").exists():
        print(f"{out} complete with identical spec; skipping")
        return
    fpr.save(out / "spec.json", spec)

    partial = out / "sql_preds.partial.jsonl"
    done = {p["id"] for p in read_jsonl(partial)}
    ts_dir = Path(args.test_suite_dir) if args.test_suite_dir else None

    model, tok = load_model(args.model, args.adapter_path)
    reset_peak_memory()
    for row in tqdm(rows, desc=f"eval {'think' if args.thinking else 'nothink'}", initial=len(done)):
        if row["id"] in done:
            continue
        set_seed(item_seed(args.eval_seed, row["id"]))
        prompt = chat_prompt(tok, build_messages(row["schema"], row["question"]), args.thinking)
        raw, st = generate_text(model, tok, prompt, args.max_tokens, **sampling)
        scored = score_generation(row, raw, st["finish_reason"])
        feats = sql_features(row["gold_sql"])
        rec = {"id": row["id"], "db_id": row["db_id"], "question": row["question"], "gold_sql": row["gold_sql"],
               "raw": raw, "difficulty": difficulty(feats), "gold_features": feats,
               **{k: v for k, v in scored.items() if k not in ("reasoning", "answer")},
               **{k: st[k] for k in ("gen_tokens", "ttft_s", "total_s", "gen_tps", "finish_reason")}}
        if ts_dir:
            rec["ts_correct"] = test_suite_correct(ts_dir, row, scored["pred_sql"], scored["correct"])
        append_jsonl(partial, rec)

    order = {r["id"]: i for i, r in enumerate(rows)}
    preds = sorted((p for p in read_jsonl(partial) if p["id"] in order), key=lambda p: order[p["id"]])
    summary = {"model": args.model, "adapter_path": args.adapter_path, "thinking": args.thinking,
               "spec_fingerprint": read_json(out / "spec.json")["fingerprint"], "protocol": spec,
               "peak_mem_gb": peak_memory_gb(), "env": env_info(), **summarize_preds(preds)}
    write_jsonl(out / "sql_preds.jsonl", preds)
    export_official(out, preds)
    write_json(out / "sql_eval.json", summary)
    partial.unlink(missing_ok=True)
    print({k: summary[k] for k in ("n", "execution_accuracy", "test_suite_accuracy", "valid_sql_rate",
                                   "truncation_rate", "avg_gen_tokens")})


if __name__ == "__main__":
    main()
