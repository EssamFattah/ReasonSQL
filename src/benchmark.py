"""Deployment benchmark: teacher vs student, before and after fusing + quantising the adapter.

  python -m src.benchmark --adapter_path runs/reasoning_nall_r16_s0/adapter

Steps: fuse the LoRA adapter into the student (mlx_lm.fuse), quantise the fused model to 8 and
4 bits (mlx_lm.convert), then run each model on the same dev prompts in a fresh subprocess
(so peak memory isn't polluted by the previous model) and record load time, disk size,
time-to-first-token, generation throughput, end-to-end latency, peak memory and EX on those
prompts, plus latency bucketed by output length (a short query and a long reasoning trace behave
differently). Disk size counts the whole model directory (weights, tokenizer, configs).
For a quality claim, run evaluate_sql on the quantised model over the full dev set.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src import fingerprint as fpr
from src.common import (build_messages, chat_prompt, generate_text, item_seed, load_model, peak_memory_gb,
                        read_json, read_jsonl, reset_peak_memory, sampling_for, score_generation, set_seed, write_json)
from src.env_info import env_info


def dir_size_gb(path: Path) -> float:
    """Everything needed to load the model: weights, tokenizer, configs (symlinks into the HF cache followed)."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


LENGTH_BUCKETS = [(0, 256, "<256 tok"), (256, 1024, "256-1024 tok"), (1024, 10**9, ">1024 tok")]


def latency_by_length(recs) -> dict:
    out = {}
    for lo, hi, name in LENGTH_BUCKETS:
        sel = [r for r in recs if lo <= (r["gen_tokens"] or 0) < hi]
        if sel:
            out[name] = {"n": len(sel), "median_latency_s": float(np.median([r["total_s"] for r in sel])),
                         "median_ms_per_token": float(np.median([1000 * r["total_s"] / max(1, r["gen_tokens"])
                                                                 for r in sel]))}
    return out


def model_size_gb(model: str, adapter: str | None) -> float | None:
    p = Path(model)
    if not p.exists():
        try:
            from huggingface_hub import snapshot_download

            p = Path(snapshot_download(model, local_files_only=True, allow_patterns=fpr._HF_PATTERNS))
        except Exception:  # noqa: BLE001
            return None
    size = dir_size_gb(p)
    return size + (dir_size_gb(Path(adapter)) if adapter else 0.0)


def bench_single(args):
    """Runs inside its own process."""
    rows = read_jsonl(args.data)[: args.n_prompts]
    t0 = time.time()
    model, tok = load_model(args.model, args.adapter_path)
    load_s = time.time() - t0
    # warm-up so the first measured prompt doesn't pay compilation costs
    generate_text(model, tok, chat_prompt(tok, [{"role": "user", "content": "Hi"}], False), 8)
    reset_peak_memory()
    recs = []
    for row in rows:
        set_seed(item_seed(0, row["id"]))
        prompt = chat_prompt(tok, build_messages(row["schema"], row["question"]), args.thinking)
        raw, st = generate_text(model, tok, prompt, args.max_tokens, **sampling_for(args.thinking))
        recs.append({**st, "correct": score_generation(row, raw, st["finish_reason"])["correct"]})
    res = {
        "name": args.name, "model": args.model, "adapter_path": args.adapter_path, "thinking": args.thinking,
        "n_prompts": len(recs), "load_s": round(load_s, 2), "disk_gb": model_size_gb(args.model, args.adapter_path),
        "peak_mem_gb": peak_memory_gb(),
        "median_ttft_s": float(np.median([r["ttft_s"] for r in recs])),
        "mean_gen_tps": float(np.mean([r["gen_tps"] for r in recs])),
        "mean_prompt_tps": float(np.mean([r["prompt_tps"] for r in recs])),
        "median_latency_s": float(np.median([r["total_s"] for r in recs])),
        "mean_gen_tokens": float(np.mean([r["gen_tokens"] for r in recs])),
        "latency_by_output_tokens": latency_by_length(recs),
        "ex_on_bench_prompts": float(np.mean([r["correct"] for r in recs])),
    }
    Path(args.single_out).write_text(json.dumps(res, indent=2))


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


ARTIFACT_SPEC = "artifact_spec.json"


def artifact_is_current(path: Path, spec: dict) -> bool:
    old = read_json(path / ARTIFACT_SPEC)
    return (path / "config.json").exists() and old is not None and old["fingerprint"] == fpr.fingerprint(spec)


def ensure_artifact(path: Path, spec: dict, build_cmd: list[str], run=sh) -> str:
    """Reuse a fused/quantised model only if it was built from exactly these inputs; otherwise rebuild.

    Derived artifacts are rebuilt rather than refused: they're cheap, and a stale one would
    silently benchmark the wrong adapter.
    """
    if artifact_is_current(path, spec):
        print(f"{path}: up to date ({fpr.fingerprint(spec)})")
    else:
        if path.exists():
            print(f"{path}: built from different inputs (or unverifiable) -> rebuilding")
            shutil.rmtree(path)  # mlx_lm.convert refuses to write into an existing directory
        run(build_cmd)
        fpr.save(path / ARTIFACT_SPEC, spec)
    return fpr.fingerprint(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_path", help="LoRA adapter of the model to deploy")
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--artifacts_dir", default="artifacts")
    ap.add_argument("--q_bits", nargs="*", type=int, default=[8, 4])
    ap.add_argument("--q_group_size", type=int, default=64)
    ap.add_argument("--data", default="data/processed/dev.jsonl")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--out", default="results/benchmark.json")
    # internal (single-model mode)
    ap.add_argument("--single_out")
    ap.add_argument("--model")
    ap.add_argument("--name")
    args = ap.parse_args()

    if args.single_out:
        return bench_single(args)

    art = Path(args.artifacts_dir)
    fused = art / "student_fused"
    fused_spec = {"base": fpr.model_identity(args.student), "adapter": fpr.adapter_identity(args.adapter_path)}
    fused_fp = ensure_artifact(fused, fused_spec, [
        sys.executable, "-m", "mlx_lm.fuse", "--model", fpr.resolve_model_path(args.student), "--adapter-path", args.adapter_path,
        "--save-path", str(fused)])
    models = [("teacher-8B-4bit", args.teacher, None),
              ("student-1.7B-base-bf16", args.student, None),
              ("student-1.7B-bf16+LoRA", args.student, args.adapter_path),
              ("student-1.7B-fused-bf16", str(fused), None)]
    for b in args.q_bits:
        q = art / f"student_fused_q{b}"
        ensure_artifact(q, {"fused": fused_fp, "q_bits": b, "q_group_size": args.q_group_size}, [
            sys.executable, "-m", "mlx_lm.convert", "--hf-path", str(fused), "--mlx-path", str(q),
            "-q", "--q-bits", str(b), "--q-group-size", str(args.q_group_size)])
        models.append((f"student-1.7B-fused-q{b}", str(q), None))

    results = []
    for name, model, adapter in models:
        tmp = art / f"_bench_{name}.json"
        cmd = [sys.executable, "-m", "src.benchmark", "--single_out", str(tmp), "--name", name, "--model", model,
               "--data", args.data, "--n_prompts", str(args.n_prompts), "--max_tokens", str(args.max_tokens),
               "--thinking" if args.thinking else "--no-thinking"]
        sh(cmd + (["--adapter_path", adapter] if adapter else []))
        results.append(json.loads(tmp.read_text()))
        print(results[-1])
    write_json(args.out, {"env": env_info(), "results": results})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
