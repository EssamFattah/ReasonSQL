"""Catastrophic-forgetting check: general ability before vs after LoRA fine-tuning.

  MMLU  (knowledge) - 0-shot, scored by next-token logits over A/B/C/D (fast, deterministic)
  GSM8K (reasoning) - generated solution, greedy, final-number exact match

Every model sees the same questions (fixed shuffle seed) with the same scoring, and per-question
results are saved so summarize.py can compute paired confidence intervals vs the base model.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.common import (chat_prompt, generate_text, item_seed, load_model, sampling_for, set_seed,
                        split_reasoning, write_json, write_jsonl)
from src import fingerprint as fpr
from src.env_info import env_info
from src.evaluate_sql import adapter_identity, model_identity


def mmlu_prompt(ex) -> str:
    return (ex["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", ex["choices"]))
            + "\nAnswer with the letter of the correct option only.")


def letter_token_ids(tok, prompt: str) -> list[int]:
    """Token ids of A-D *as they appear right after the prompt* (guards against tokenisation artefacts)."""
    base = tok.encode(prompt, add_special_tokens=False)
    ids = []
    for letter in "ABCD":
        full = tok.encode(prompt + letter, add_special_tokens=False)
        if full[: len(base)] != base or len(full) != len(base) + 1:
            raise RuntimeError(f"Letter {letter!r} doesn't tokenise as a single token after the prompt; "
                               "MMLU logit scoring would be unreliable for this tokenizer.")
        ids.append(full[-1])
    return ids


def eval_mmlu(model, tok, n: int, seed: int):
    import mlx.core as mx
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(n))
    letter_ids = None
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="mmlu")):
        prompt = chat_prompt(tok, [{"role": "user", "content": mmlu_prompt(ex)}], enable_thinking=False)
        if letter_ids is None:
            letter_ids = mx.array(letter_token_ids(tok, prompt))
        ids = tok.encode(prompt, add_special_tokens=False)
        logits = model(mx.array(ids)[None])[0, -1]
        pred = int(mx.argmax(logits[letter_ids]).item())
        rows.append({"id": f"mmlu_{i}", "subject": ex["subject"], "pred": pred, "gold": ex["answer"],
                     "correct": pred == ex["answer"]})
    return rows


_ANSWER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")


def parse_final_number(text: str):
    """Prefer the requested 'Answer: N' line; fall back to the last number (and flag it)."""
    m = _ANSWER_RE.findall(text)
    if m:
        s, formatted = m[-1], True
    else:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if not nums:
            return None, False
        s, formatted = nums[-1], False
    try:
        return float(s.replace(",", "").rstrip(".")), formatted
    except ValueError:
        return None, formatted


def eval_gsm8k(model, tok, n: int, seed: int, thinking: bool, max_tokens: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed).select(range(n))
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="gsm8k")):
        set_seed(item_seed(seed, "gsm8k", i))
        q = ex["question"] + "\nSolve step by step. Put the final answer on the last line as 'Answer: <number>'."
        raw, st = generate_text(model, tok, chat_prompt(tok, [{"role": "user", "content": q}], thinking),
                                max_tokens, **sampling_for(thinking))
        _, answer, _ = split_reasoning(raw)
        gold = float(ex["answer"].split("####")[-1].strip().replace(",", ""))
        pred, formatted = parse_final_number(answer)
        rows.append({"id": f"gsm8k_{i}", "gold": gold, "pred": pred, "answer_format_ok": formatted,
                     "truncated": st["finish_reason"] == "length",
                     "correct": pred is not None and abs(pred - gold) < 1e-6})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--mmlu_n", type=int, default=1000)
    ap.add_argument("--gsm8k_n", type=int, default=300)
    ap.add_argument("--gsm8k_thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--gsm8k_max_tokens", type=int, default=512)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0, help="question-sampling seed: keep identical across models")
    args = ap.parse_args()

    out = Path(args.out_dir)
    spec = {"model": model_identity(args.model), "adapter": adapter_identity(args.adapter_path),
            **{k: v for k, v in vars(args).items() if k not in ("model", "adapter_path", "out_dir")}}
    _, existed = fpr.check(out / "forgetting.spec.json", spec, "forgetting eval")
    if existed and (out / "forgetting.json").exists():
        print(f"{out}/forgetting.json complete with identical spec; skipping")
        return
    fpr.save(out / "forgetting.spec.json", spec)
    model, tok = load_model(args.model, args.adapter_path)
    res = {"model": args.model, "adapter_path": args.adapter_path, "protocol": vars(args), "env": env_info()}
    rows = []
    if args.mmlu_n > 0:
        m = eval_mmlu(model, tok, args.mmlu_n, args.seed)
        rows += m
        res["mmlu_acc"] = float(np.mean([r["correct"] for r in m]))
    if args.gsm8k_n > 0:
        g = eval_gsm8k(model, tok, args.gsm8k_n, args.seed, args.gsm8k_thinking, args.gsm8k_max_tokens)
        rows += g
        res["gsm8k_acc"] = float(np.mean([r["correct"] for r in g]))
        res["gsm8k_answer_format_rate"] = float(np.mean([r["answer_format_ok"] for r in g]))
        res["gsm8k_truncation_rate"] = float(np.mean([r["truncated"] for r in g]))
    write_jsonl(out / "forgetting_preds.jsonl", rows)
    write_json(out / "forgetting.json", res)
    print({k: v for k, v in res.items() if k.endswith(("acc", "rate"))})


if __name__ == "__main__":
    main()
