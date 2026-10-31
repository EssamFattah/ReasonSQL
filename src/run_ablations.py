"""Run baselines + ablations end to end, resumably and safely.

  python -m src.run_ablations --profile pilot           # Phase 1: one seed, 300 dev questions
  python -m src.run_ablations --profile full            # Phase 2: 3 training seeds on key runs, full dev
  python -m src.run_ablations --profile full --dry_run  # print the plan only

Every stage is always invoked and decides for itself whether to skip, by comparing its spec
fingerprint with the one stored next to its outputs. Matching -> skip/resume; different ->
the run stops with a list of the settings that changed. Changing a hyperparameter therefore
can never be silently "resumed" into old results.

Two kinds of randomness are kept separate:
  * training seed (--seeds): LoRA init + data order. Varies across training replicates.
  * eval seed (--eval_seeds): inference sampling in thinking mode. The SAME eval seeds are used
    for every model, so differences between training seeds aren't confounded with sampling luck.
    Baselines can get extra eval seeds (--baseline_eval_seeds) to measure inference-only variance.
  Direct (no-think) mode is greedy, so it is evaluated once.

Layout:
  runs/_data/<mode>_n<N>/                         dataset (spec.json, meta.json, train/valid.jsonl)
  runs/<mode>_n<N>_r<R>_s<train_seed>/
      adapter/                                    weights, spec.json, train_summary.json, DONE
      eval_think_e<eval_seed>/  eval_nothink/     sql_eval.json, sql_preds.jsonl, spec.json, official/
      forgetting.json  forgetting_preds.jsonl  forgetting.spec.json
      config.json (fingerprinted run identity)  env.json
  runs/base_student/  runs/teacher/               baselines
"""
import argparse
import subprocess
import sys
from pathlib import Path

from src import fingerprint as fpr
from src.build_sft import MODES, build
from src.common import DEFAULT_MAX_EXAMPLE_TOKENS, read_json, write_json
from src.env_info import env_info

STUDENT = "Qwen/Qwen3-1.7B"
TEACHER = "mlx-community/Qwen3-8B-4bit"
DEFAULT_RANK = 16

PROFILES = {
    "pilot": dict(groups=["baselines", "traces"], seeds=[0], multi_seed_groups=[], eval_seeds=[0],
                  baseline_eval_seeds=[0], eval_limit=300, sweep_eval_limit=300,
                  sizes=[250, 500, -1], ranks=[8, 16, 32], mmlu_n=300, gsm8k_n=100),
    # Headline runs (baselines + the 2x2) use the full dev set; size/rank sweeps use the first 300
    # questions of the same pre-shuffled dev set (a strict subset, so comparisons stay paired).
    "full": dict(groups=["baselines", "size", "rank", "traces"], seeds=[0, 1, 2], multi_seed_groups=["traces"],
                 eval_seeds=[0], baseline_eval_seeds=[0, 1, 2], eval_limit=-1, sweep_eval_limit=300,
                 sizes=[250, 500, 1000, 2000, -1], ranks=[4, 8, 16, 32, 64], mmlu_n=1000, gsm8k_n=300),
}


def tag(n: int) -> str:
    return "all" if n < 0 else str(n)


def lora_specs(cfg) -> list[dict]:
    groups = {
        "size": [("reasoning", n, DEFAULT_RANK) for n in cfg["sizes"]],
        "rank": [("reasoning", cfg["rank_n"], r) for r in cfg["ranks"]],
        "traces": [(m, cfg["rank_n"], DEFAULT_RANK) for m in MODES],
    }
    headline = set(groups["traces"]) if "traces" in cfg["groups"] else set()
    specs, seen = [], set()
    for g in cfg["groups"]:
        seeds = cfg["seeds"] if g in cfg["multi_seed_groups"] else cfg["seeds"][:1]
        for mode, n, rank in groups.get(g, []):
            for s in seeds:
                if (mode, n, rank, s) not in seen:  # e.g. size=all/r16 == rank16 == traces/reasoning
                    seen.add((mode, n, rank, s))
                    specs.append({"mode": mode, "n_train": n, "rank": rank, "seed": s,
                                  "key": f"{mode}_n{tag(n)}_r{rank}", "group": g,
                                  "eval_limit": cfg["eval_limit"] if (mode, n, rank) in headline
                                  else cfg["sweep_eval_limit"]})
    return specs


class Runner:
    def __init__(self, cfg, dry):
        self.cfg, self.dry = cfg, dry

    def sh(self, cmd):
        print("$", " ".join(cmd), flush=True)
        if not self.dry:
            subprocess.run(cmd, check=True)

    def evals(self, run_dir: Path, model, adapter, eval_seeds, limit):
        base = [sys.executable, "-m", "src.evaluate_sql", "--model", model, "--limit", str(limit)]
        if adapter:
            base += ["--adapter_path", adapter]
        if self.cfg.get("test_suite_dir"):
            base += ["--test_suite_dir", self.cfg["test_suite_dir"]]
        for e in eval_seeds:
            self.sh(base + ["--thinking", "--eval_seed", str(e), "--out_dir", str(run_dir / f"eval_think_e{e}")])
        self.sh(base + ["--no-thinking", "--out_dir", str(run_dir / "eval_nothink")])

    def forgetting(self, run_dir: Path, adapter):
        if not self.cfg["forgetting"]:
            return
        cmd = [sys.executable, "-m", "src.eval_forgetting", "--model", STUDENT, "--out_dir", str(run_dir),
               "--mmlu_n", str(self.cfg["mmlu_n"]), "--gsm8k_n", str(self.cfg["gsm8k_n"])]
        self.sh(cmd + (["--adapter_path", adapter] if adapter else []))

    def identity(self, run_dir: Path, ident: dict):
        """config.json = the run's fingerprinted identity; refuse if it already names something else."""
        if self.dry:
            return
        fpr.check(run_dir / "config.json", ident, f"run {run_dir.name}")
        fpr.save(run_dir / "config.json", ident)
        write_json(run_dir / "env.json", {"env": env_info(), "run_config": self.cfg})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), default="pilot")
    ap.add_argument("--groups", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int, help="training seeds")
    ap.add_argument("--multi_seed_groups", nargs="*")
    ap.add_argument("--eval_seeds", nargs="+", type=int, help="thinking-mode inference seeds, shared by all models")
    ap.add_argument("--baseline_eval_seeds", nargs="+", type=int)
    ap.add_argument("--sizes", nargs="+", type=int)
    ap.add_argument("--ranks", nargs="+", type=int)
    ap.add_argument("--rank_n", type=int, default=-1, help="train size for rank/trace ablations (-1 = all)")
    ap.add_argument("--eval_limit", type=int, help="dev questions for baselines + trace ablation (-1 = all)")
    ap.add_argument("--sweep_eval_limit", type=int, help="dev questions for size/rank sweeps")
    ap.add_argument("--mmlu_n", type=int)
    ap.add_argument("--gsm8k_n", type=int)
    ap.add_argument("--test_suite_dir", default=None, help="Spider test-suite databases (optional)")
    ap.add_argument("--forgetting", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_EXAMPLE_TOKENS)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--traces", default="data/teacher/traces.jsonl")
    ap.add_argument("--pool", default="data/processed/train_pool.jsonl")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = dict(PROFILES[args.profile])
    for k in ("groups", "seeds", "multi_seed_groups", "eval_seeds", "baseline_eval_seeds", "sizes", "ranks",
              "eval_limit", "sweep_eval_limit", "mmlu_n", "gsm8k_n"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg.update(profile=args.profile, rank_n=args.rank_n, forgetting=args.forgetting, epochs=args.epochs,
               lr=args.lr, batch_size=args.batch_size, max_seq_length=args.max_seq_length,
               test_suite_dir=args.test_suite_dir)
    runs = Path(args.runs_dir)
    R = Runner(cfg, args.dry_run)

    if "baselines" in cfg["groups"]:
        # Inference-only replicates for the base student only: the 8B teacher's thinking evals are
        # ~10 h each on the full dev set, and its variance isn't needed for any comparison.
        for key, model, forgetting, eval_seeds in (("base_student", STUDENT, True, cfg["baseline_eval_seeds"]),
                                                   ("teacher", TEACHER, False, cfg["eval_seeds"])):
            d = runs / key
            R.identity(d, {"kind": "baseline", "key": key, "model": model})
            R.evals(d, model, None, eval_seeds, cfg["eval_limit"])
            if forgetting:
                R.forgetting(d, None)

    for spec in lora_specs(cfg):
        data_dir = runs / "_data" / f"{spec['mode']}_n{tag(spec['n_train'])}"
        d = runs / f"{spec['key']}_s{spec['seed']}"
        adapter = d / "adapter"
        if args.dry_run:
            print(f"# build/validate {data_dir}")
        else:
            meta = build(args.traces, args.pool, data_dir, spec["mode"], spec["n_train"],
                         max_example_tokens=cfg["max_seq_length"])
            print("data", data_dir.name, meta["funnel"])
        R.sh([sys.executable, "-m", "src.train", "--data_dir", str(data_dir), "--adapter_path", str(adapter),
              "--model", STUDENT, "--rank", str(spec["rank"]), "--seed", str(spec["seed"]),
              "--epochs", str(cfg["epochs"]), "--lr", str(cfg["lr"]), "--batch_size", str(cfg["batch_size"]),
              "--max_seq_length", str(cfg["max_seq_length"])])
        R.identity(d, {"kind": "lora", "key": spec["key"], "mode": spec["mode"], "n_train": spec["n_train"],
                       "rank": spec["rank"], "seed": spec["seed"], "student": STUDENT, "teacher": TEACHER,
                       "dataset": "spider", "epochs": cfg["epochs"], "lr": cfg["lr"],
                       "batch_size": cfg["batch_size"], "max_seq_length": cfg["max_seq_length"],
                       "data_fingerprint": fpr.read_fingerprint(data_dir / "spec.json"),
                       "data_meta": read_json(data_dir / "meta.json")})
        R.evals(d, STUDENT, str(adapter), cfg["eval_seeds"], spec["eval_limit"])
        R.forgetting(d, str(adapter))

    print("done. Next: python -m src.summarize")


if __name__ == "__main__":
    main()
