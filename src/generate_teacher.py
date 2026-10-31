"""Generate reasoning traces with the teacher (Qwen3-8B, thinking mode) and verify each by execution.

Rejection sampling: up to --num_samples attempts per question, stopping at the first *usable*
trace (common.is_usable): correct by execution AND not truncated AND the full training example
fits the student's context. A correct-but-unusable attempt (e.g. too long) doesn't stop sampling,
so generation, statistics and dataset construction all agree on what a usable trace is.

Every attempt is logged (correct or not): the unverified ablations and pass@k analysis use them.
Resumable, with a spec fingerprint: changing the model, sampling, max_tokens, seed, input data or
the length limit refuses to append to an existing traces file. --limit and --num_samples can be
raised to extend a run.
"""
import argparse
from collections import defaultdict

from tqdm import tqdm

from src import fingerprint as fpr
from src.common import (DEFAULT_MAX_EXAMPLE_TOKENS, SYSTEM_PROMPT, THINK_SAMPLING, append_jsonl, build_messages,
                        chat_prompt, example_token_count, generate_text, is_usable, item_seed, load_model,
                        read_jsonl, score_generation, set_seed, write_json)
from src.env_info import env_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--input", default="data/processed/train_pool.jsonl")
    ap.add_argument("--out", default="data/teacher/traces.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="number of questions to process")
    ap.add_argument("--num_samples", type=int, default=3, help="max attempts per question")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_example_tokens", type=int, default=DEFAULT_MAX_EXAMPLE_TOKENS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = {"model": fpr.model_identity(args.model), "max_tokens": args.max_tokens, "seed": args.seed, "sampling": THINK_SAMPLING,
            "max_example_tokens": args.max_example_tokens, "input_sha": fpr.file_sha(args.input),
            "system_prompt": SYSTEM_PROMPT}
    spec_path = args.out + ".spec.json"
    fpr.check(spec_path, spec, "teacher traces")
    fpr.save(spec_path, spec)
    write_json(args.out + ".env.json", {"args": vars(args), "env": env_info()})

    rows = read_jsonl(args.input)[: args.limit]
    prev = defaultdict(list)
    for r in read_jsonl(args.out):
        prev[r["id"]].append(r)

    model, tok = load_model(args.model)
    usable = lambda t: is_usable(t, args.max_example_tokens)  # noqa: E731
    stats = defaultdict(int)
    for row in tqdm(rows, desc="teacher"):
        attempts = prev[row["id"]]
        if any(usable(a) for a in attempts) or len(attempts) >= args.num_samples:
            continue
        messages = build_messages(row["schema"], row["question"])
        prompt = chat_prompt(tok, messages, enable_thinking=True)
        for k in range(len(attempts), args.num_samples):
            set_seed(item_seed(args.seed, row["id"], k))
            raw, st = generate_text(model, tok, prompt, args.max_tokens, **THINK_SAMPLING)
            scored = score_generation(row, raw, st["finish_reason"])
            rec = {"id": row["id"], "sample_idx": k, "db_id": row["db_id"], "question": row["question"],
                   "gold_sql": row["gold_sql"], "n_tokens": st["gen_tokens"], "gen_time_s": round(st["total_s"], 2),
                   "gen_tps": st["gen_tps"], **scored,
                   "example_tokens": example_token_count(tok, messages, scored["reasoning"], scored["pred_sql"])}
            append_jsonl(args.out, rec)
            stats["attempts"] += 1
            stats["correct_attempts"] += rec["correct"]
            if usable(rec):
                stats["usable"] += 1
                break
        stats["questions"] += 1

    print(dict(stats))
    print("Run `python -m src.teacher_stats` for pass@k, the data funnel and failure categories.")


if __name__ == "__main__":
    main()
