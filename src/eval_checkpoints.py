"""SQL accuracy and general ability (MMLU) as fine-tuning progresses, from saved checkpoints.

  python -m src.eval_checkpoints --run_dir runs/reasoning_nall_r16_s0

Shows the specialisation-vs-forgetting trade-off over training steps. Defaults to greedy
no-thinking SQL evaluation on a dev subset (cheap and deterministic); iteration 0 is the base model.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.common import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--mmlu_n", type=int, default=300)
    args = ap.parse_args()

    run = Path(args.run_dir)
    adapter = run / "adapter"
    ckpts = sorted((int(p.name.split("_")[0]), p) for p in adapter.glob("*_adapters.safetensors"))
    final_iter = (read_json(adapter / "train_summary.json") or {}).get("iters")
    if final_iter and final_iter not in {i for i, _ in ckpts} and (adapter / "adapters.safetensors").exists():
        ckpts.append((final_iter, adapter / "adapters.safetensors"))
    points = [(0, None)] + ckpts
    curve = []
    for it, weights in points:
        out = run / "checkpoints" / f"iter_{it:07d}"
        adapter_dir = None
        if weights is not None:
            adapter_dir = out / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(adapter / "adapter_config.json", adapter_dir / "adapter_config.json")
            link = adapter_dir / "adapters.safetensors"
            if not link.exists():
                link.symlink_to(weights.resolve())
        extra = ["--adapter_path", str(adapter_dir)] if adapter_dir else []
        mode = "eval_think" if args.thinking else "eval_nothink"
        subprocess.run([sys.executable, "-m", "src.evaluate_sql", "--model", args.model, "--limit", str(args.limit),
                        "--out_dir", str(out / mode), "--thinking" if args.thinking else "--no-thinking"] + extra,
                       check=True)
        subprocess.run([sys.executable, "-m", "src.eval_forgetting", "--model", args.model, "--out_dir", str(out),
                        "--mmlu_n", str(args.mmlu_n), "--gsm8k_n", "0"] + extra, check=True)
        curve.append({"iter": it,
                      "ex": read_json(out / mode / "sql_eval.json")["execution_accuracy"],
                      "mmlu": read_json(out / "forgetting.json")["mmlu_acc"]})
        print(json.dumps(curve[-1]))
    write_json(run / "checkpoint_curve.json", {"thinking": args.thinking, "limit": args.limit,
                                               "mmlu_n": args.mmlu_n, "points": curve})


if __name__ == "__main__":
    main()
