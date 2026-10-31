"""Teacher yield analysis and the data funnel.

  questions started -> finished -> teacher ever correct -> usable verified trace -> train/valid

pass@k here = share of finished questions with a usable trace within the first k sequential
attempts (sampling stops at the first usable one, so this is the rejection-sampling yield).
"Correct but unusable" counts attempts dropped by the truncation/length filters.
Only finished questions count (usable trace found, or all K attempts used).
"""
import argparse

import numpy as np

from src.common import DEFAULT_MAX_EXAMPLE_TOKENS, group_by, is_usable, read_jsonl, tally, write_json


def compute(traces: list[dict], k: int, max_example_tokens: int = DEFAULT_MAX_EXAMPLE_TOKENS) -> dict:
    usable = lambda t: is_usable(t, max_example_tokens)  # noqa: E731
    by_q = {q: sorted(a, key=lambda t: t["sample_idx"]) for q, a in group_by(traces, "id").items()}
    finished = {q: a for q, a in by_q.items() if any(usable(t) for t in a) or len(a) >= k}
    n = len(finished)
    usable_at = [next((t["sample_idx"] for t in a if usable(t)), None) for a in finished.values()]
    all_t = [t for a in finished.values() for t in a]
    failed = [t for t in all_t if not t["correct"]]
    correct_unusable = [t for t in traces if t["correct"] and not usable(t)]  # all attempts, finished or not
    lens = [t["example_tokens"] for t in all_t if usable(t)]

    def frac(xs):
        return float(np.mean(xs)) if xs else None

    return {
        "funnel": {
            "questions_started": len(by_q),
            "questions_finished": n,
            "teacher_ever_correct": sum(any(t["correct"] for t in a) for a in finished.values()),
            "usable_verified": sum(u is not None for u in usable_at),
        },
        "max_attempts": k, "max_example_tokens": max_example_tokens, "attempts": len(all_t),
        "gold_query_errors": sum(any(t["category"] == "gold_error" for t in a) for a in finished.values()),
        "pass_at": {f"pass@{i}": frac([u is not None and u < i for u in usable_at]) for i in range(1, k + 1)},
        "never_usable": frac([u is None for u in usable_at]),
        "gain_from_resampling": frac([u is not None and u > 0 for u in usable_at]),
        "correct_but_unusable_attempts": {"n": len(correct_unusable),
                                          "truncated": sum(t["truncated"] for t in correct_unusable),
                                          "too_long": sum((t["example_tokens"] or 0) > max_example_tokens
                                                          for t in correct_unusable)},
        "failed_attempt_categories": tally(t["category"] for t in failed),
        "failed_attempt_struct_diffs": tally(s for t in failed for s in t.get("struct_diff", [])),
        "truncation_rate": frac([t["truncated"] for t in all_t]),
        "avg_tokens_per_attempt": frac([t["n_tokens"] or 0 for t in all_t]),
        "avg_seconds_per_attempt": frac([t["gen_time_s"] for t in all_t]),
        "total_generation_hours": float(sum(t["gen_time_s"] for t in all_t) / 3600),
        "usable_example_tokens": ({"p10": float(np.percentile(lens, 10)), "median": float(np.median(lens)),
                                   "p90": float(np.percentile(lens, 90)), "max": int(max(lens))} if lens else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/teacher/traces.jsonl")
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--max_example_tokens", type=int, default=DEFAULT_MAX_EXAMPLE_TOKENS)
    ap.add_argument("--out", default="results/teacher_stats.json")
    args = ap.parse_args()
    s = compute(read_jsonl(args.traces), args.num_samples, args.max_example_tokens)
    write_json(args.out, s)
    print("funnel:", s["funnel"])
    print("  ".join(f"{k}={v:.1%}" for k, v in s["pass_at"].items() if v is not None))
    print("correct but unusable:", s["correct_but_unusable_attempts"])
    print("failed attempts:", s["failed_attempt_categories"])


if __name__ == "__main__":
    main()
