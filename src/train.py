"""LoRA fine-tune the student with mlx_lm.

* Writes the full YAML config next to the adapter, plus env info.
* Parses the training log into train_summary.json (loss curves, peak memory, wall time).
* Writes a DONE marker only when training exits cleanly. mlx_lm overwrites adapters.safetensors
  at every checkpoint, so without the marker an interrupted run would look finished.
  A run without DONE is wiped and retrained from scratch.
* adapter/spec.json fingerprints every hyperparameter plus the dataset's fingerprint and file
  hashes. A finished adapter with a different spec is never silently reused: training refuses.
* Refuses to train if the dataset allows examples longer than --max_seq_length (mlx_lm would
  truncate them and cut off the SQL target).

Note on LoRA scale: mlx_lm computes `scale * (x @ A @ B)` with no 1/rank factor, so keeping
`scale` fixed across the rank ablation is equivalent to keeping alpha/r constant.
"""
import argparse
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src import fingerprint as fpr
from src.common import DEFAULT_MAX_EXAMPLE_TOKENS, read_json, write_json
from src.env_info import env_info


def is_trained(adapter_path) -> bool:
    return (Path(adapter_path) / "DONE").exists()


def count_lines(p: Path) -> int:
    return sum(1 for line in p.open() if line.strip())


def make_config(args) -> dict:
    data_dir = Path(args.data_dir)
    n_train = count_lines(data_dir / "train.jsonl")
    iters = args.iters or max(1, math.ceil(args.epochs * n_train / args.batch_size))
    warmup = max(1, int(args.warmup_frac * iters))
    every = max(25, iters // 10)  # eval and checkpoint on the same steps -> every val point has weights
    return {
        "model": args.model,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "iters": iters,
        "learning_rate": args.lr,
        "lr_schedule": {"name": "cosine_decay", "warmup": warmup, "warmup_init": 0.0,
                        "arguments": [args.lr, max(1, iters - warmup), args.lr * 0.1]},
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": args.dropout},
        "adapter_path": args.adapter_path,
        "max_seq_length": args.max_seq_length,
        "mask_prompt": True,  # loss only on the assistant turn (reasoning + SQL)
        "grad_checkpoint": args.grad_checkpoint,
        "steps_per_report": 10,
        "steps_per_eval": every,
        "save_every": every,
        "val_batches": args.val_batches,
        "seed": args.seed,
    }


def training_spec(cfg: dict, model: str, data_dir: Path, data_spec: dict) -> dict:
    """Everything that determines the trained weights (output paths excluded)."""
    # "model" is a machine-specific path; model_identity (repo + commit / weights hash) stands in for it.
    return {**{k: v for k, v in cfg.items() if k not in ("adapter_path", "data", "model")},
            "model_identity": fpr.model_identity(model),  # pins the HF revision of the base weights
            "data_fingerprint": data_spec["fingerprint"],
            "train_sha": fpr.file_sha(data_dir / "train.jsonl"), "valid_sha": fpr.file_sha(data_dir / "valid.jsonl")}


def parse_log(text: str) -> dict:
    train = [(int(i), float(l)) for i, l in re.findall(r"Iter (\d+): Train loss ([\d.]+)", text)]
    val = [(int(i), float(l)) for i, l in re.findall(r"Iter (\d+): Val loss ([\d.]+)", text)]
    mem = [float(m) for m in re.findall(r"Peak mem ([\d.]+) GB", text)]
    tps = [float(t) for t in re.findall(r"Tokens/sec ([\d.]+)", text)]
    best = min(val, key=lambda x: x[1]) if val else None
    return {
        "train_curve": train, "val_curve": val,
        "final_train_loss": train[-1][1] if train else None,
        "final_val_loss": val[-1][1] if val else None,
        "best_val_loss": best[1] if best else None, "best_val_iter": best[0] if best else None,
        "peak_mem_gb": max(mem) if mem else None,
        "train_tokens_per_s": sum(tps) / len(tps) if tps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--num_layers", type=int, default=-1, help="-1 = all transformer layers")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=None, help="overrides --epochs")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_EXAMPLE_TOKENS)
    ap.add_argument("--val_batches", type=int, default=25)
    ap.add_argument("--grad_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    adapter, data_dir = Path(args.adapter_path), Path(args.data_dir)
    data_spec = read_json(data_dir / "spec.json")
    if data_spec is None:
        raise SystemExit(f"{data_dir}/spec.json missing: build datasets with src.build_sft")
    if data_spec["spec"]["max_example_tokens"] > args.max_seq_length:
        raise SystemExit(f"dataset allows {data_spec['spec']['max_example_tokens']}-token examples but "
                         f"--max_seq_length={args.max_seq_length}: mlx_lm would truncate SQL targets")

    cfg = make_config(args)
    cfg["model"] = fpr.resolve_model_path(args.model)  # mlx_lm.lora loads the pinned snapshot
    spec = training_spec(cfg, args.model, data_dir, data_spec)
    spec_path = adapter / "spec.json"
    if is_trained(adapter):
        if not spec_path.exists():
            raise fpr.IncompatibleRun(f"{adapter} was trained by an older version (no spec.json); "
                                      "use a new --runs_dir")
        fpr.check(spec_path, spec, f"adapter {adapter}")  # raises on mismatch
        print(f"{adapter} already trained with identical spec; skipping")
        return
    if adapter.exists():
        print(f"{adapter} has no DONE marker (interrupted run) -> wiping and retraining")
        shutil.rmtree(adapter)
    adapter.mkdir(parents=True)
    fpr.save(spec_path, spec)

    cfg_path = adapter / "train_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"iters={cfg['iters']} rank={args.rank} seed={args.seed} -> {cfg_path}")

    t0 = time.time()
    log_lines = []
    with (adapter / "train.log").open("w") as log:
        proc = subprocess.Popen([sys.executable, "-m", "mlx_lm.lora", "--config", str(cfg_path)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
            log_lines.append(line)
        code = proc.wait()

    summary = {"exit_code": code, "wall_time_s": round(time.time() - t0, 1), "iters": cfg["iters"],
               "config": cfg, "env": env_info(), **parse_log("".join(log_lines))}
    write_json(adapter / "train_summary.json", summary)
    if code != 0:
        sys.exit(code)
    (adapter / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
