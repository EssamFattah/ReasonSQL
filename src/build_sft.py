"""Build mlx_lm LoRA datasets ({train,valid}.jsonl, chat format) from teacher traces.

The trace ablation is a 2x2 design on the SAME questions:

                         execution-verified target        first well-formed attempt (unchecked)
  reasoning + SQL        reasoning                        reasoning_unverified
  SQL only               sql_only                         sql_unverified

* Rows differ only in whether the reasoning is included (same attempt, same SQL, same questions).
* Columns differ only in whether the target was checked by execution. Unverified targets are the
  teacher's first attempt passing the cheap format filters (non-empty SQL, not truncated, fits the
  context window): what you'd get without an execution check. For questions solved on the first
  usable try both columns are identical, so the comparison isolates the wrong attempts.

Which questions are eligible is decided by common.is_usable (the same rule the teacher used to
stop sampling). Length is checked on the full tokenised example, so mlx_lm never truncates away
the SQL. The question order and validation set are fixed by --split_seed and shared by every
mode; training subsets are nested. The output dir is fingerprinted against its inputs: if the
traces file changes (e.g. you generated more), rebuilding into the same dir is refused.
"""
import argparse
import random
from pathlib import Path

import numpy as np

from src import fingerprint as fpr
from src.common import (DEFAULT_MAX_EXAMPLE_TOKENS, build_messages, format_target, group_by, is_usable,
                        is_well_formed, read_json, read_jsonl, write_json, write_jsonl)

MODES = {  # mode: (use verified target, include reasoning)
    "reasoning": (True, True),
    "sql_only": (True, False),
    "reasoning_unverified": (False, True),
    "sql_unverified": (False, False),
}


def hash_ids(ids) -> str:
    import hashlib

    return hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]


def fill_example_tokens(traces, pool, tokenizer_name="Qwen/Qwen3-1.7B"):
    """For traces generated before example_tokens was recorded."""
    from transformers import AutoTokenizer

    from src.common import example_token_count

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    for t in traces:
        if t.get("example_tokens") is None:
            src = pool[t["id"]]
            t["example_tokens"] = example_token_count(tok, build_messages(src["schema"], src["question"]),
                                                      t["reasoning"], t["pred_sql"])
    return traces


def build(traces_path, pool_path, out_dir, mode, n_train=-1, valid_size=100,
          max_example_tokens=DEFAULT_MAX_EXAMPLE_TOKENS, max_chars=None, split_seed=0, fill_tokens=False) -> dict:
    out = Path(out_dir)
    spec = {"mode": mode, "n_train": n_train, "valid_size": valid_size, "max_example_tokens": max_example_tokens,
            "max_chars": max_chars, "split_seed": split_seed,
            "traces_sha": fpr.file_sha(traces_path), "pool_sha": fpr.file_sha(pool_path)}
    _, existed = fpr.check(out / "spec.json", spec, f"dataset {out.name}")
    if existed and (out / "meta.json").exists():
        return read_json(out / "meta.json")

    use_verified, use_reasoning = MODES[mode]
    traces = read_jsonl(traces_path)
    pool = {r["id"]: r for r in read_jsonl(pool_path)}
    if fill_tokens:
        traces = fill_example_tokens(traces, pool)
    by_q = {q: sorted(a, key=lambda t: t["sample_idx"]) for q, a in group_by(traces, "id").items()}

    def first(attempts, pred):
        return next((t for t in attempts if pred(t)), None)

    verified = {q: t for q, a in by_q.items()
                if (t := first(a, lambda x: is_usable(x, max_example_tokens, max_chars)))}
    ids = sorted(verified)
    random.Random(split_seed).shuffle(ids)
    valid_ids, train_ids = ids[:valid_size], ids[valid_size:]
    if n_train > 0:
        train_ids = train_ids[:n_train]

    def target(qid):
        if use_verified:
            return verified[qid]
        return first(by_q[qid], lambda x: is_well_formed(x, max_example_tokens, max_chars))

    def example(t):
        src = pool[t["id"]]
        msgs = build_messages(src["schema"], src["question"])
        msgs.append({"role": "assistant",
                     "content": format_target(t["reasoning"] if use_reasoning else "", t["pred_sql"])})
        return {"messages": msgs}

    train_targets = [target(q) for q in train_ids]
    write_jsonl(out / "train.jsonl", [example(t) for t in train_targets])
    # Validation targets are always verified, in this mode's format, so val loss is comparable.
    write_jsonl(out / "valid.jsonl", [example(verified[q]) for q in valid_ids])

    ever_correct = sum(any(t["correct"] for t in a) for a in by_q.values())
    meta = {
        "mode": mode, "n_train_requested": n_train, "n_train": len(train_ids), "n_valid": len(valid_ids),
        "funnel": {"questions_with_traces": len(by_q), "teacher_ever_correct": ever_correct,
                   "usable_verified": len(verified), "valid": len(valid_ids), "train": len(train_ids)},
        "frac_targets_correct": float(np.mean([t["correct"] for t in train_targets])) if train_targets else None,
        "frac_targets_differ_from_verified": float(np.mean(
            [t["sample_idx"] != verified[t["id"]]["sample_idx"] for t in train_targets])) if train_targets else None,
        "example_tokens": ({"median": float(np.median([t["example_tokens"] for t in train_targets])),
                            "max": int(max(t["example_tokens"] for t in train_targets))} if train_targets else None),
        "train_ids_sha": hash_ids(train_ids), "valid_ids_sha": hash_ids(valid_ids),
    }
    write_json(out / "meta.json", meta)
    fpr.save(out / "spec.json", spec)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="data/teacher/traces.jsonl")
    ap.add_argument("--pool", default="data/processed/train_pool.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=list(MODES), default="reasoning")
    ap.add_argument("--n_train", type=int, default=-1, help="-1 = all available")
    ap.add_argument("--valid_size", type=int, default=100)
    ap.add_argument("--max_example_tokens", type=int, default=DEFAULT_MAX_EXAMPLE_TOKENS)
    ap.add_argument("--max_chars", type=int, default=None, help="optional extra cap on reasoning length")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--fill_tokens", action="store_true", help="compute example_tokens for old traces")
    ap.add_argument("--show", action="store_true", help="print one rendered training example")
    args = ap.parse_args()

    meta = build(args.traces, args.pool, args.out_dir, args.mode, args.n_train, args.valid_size,
                 args.max_example_tokens, args.max_chars, args.split_seed, args.fill_tokens)
    print(meta)
    if args.show:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        ex = read_jsonl(Path(args.out_dir) / "train.jsonl")[0]
        print(tok.apply_chat_template(ex["messages"], tokenize=False))


if __name__ == "__main__":
    main()
