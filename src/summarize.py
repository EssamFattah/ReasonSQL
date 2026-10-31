"""Collect every run into results/report.md (+ results/summary.json, per-DB CSV and figures).

Aggregation: per question, scores are averaged over eval seeds within a training run, then over
training seeds. EX is reported as mean ± std across TRAINING seeds (baselines: ± std across eval
seeds = inference-only variance), with a 95% CI over questions and a database-clustered 95% CI.
Comparisons are paired on the same questions: bootstrap CIs (question-level and DB-clustered)
and sign-flip permutation p-values.

  python -m src.summarize                 # execution accuracy on the Spider dev DBs
  python -m src.summarize --metric ts     # test-suite accuracy (if evaluated with --test_suite_dir)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.common import read_json, read_jsonl, write_json  # noqa: E402
from src.stats import accuracy_ci, contrast, mean_std, paired_test, seed_average  # noqa: E402

MODES = ("think", "nothink")
LABELS = {"base_student": "Qwen3-1.7B base", "teacher": "Qwen3-8B teacher",
          "student_fused_q4": "distilled 1.7B, fused 4-bit"}


def label(key):
    return LABELS.get(key, key)


# ------------------------------------------------------------------ loading
def _replicate(eval_dir: Path, metric: str):
    s = read_json(eval_dir / "sql_eval.json")
    if not s:
        return None
    preds = read_jsonl(eval_dir / "sql_preds.jsonl")
    field = "ts_correct" if metric == "ts" else "correct"
    per_q = {p["id"]: float(p[field]) for p in preds if p.get(field) is not None}
    if not per_q:
        return None
    return {"summary": s, "preds": preds, "per_q": per_q, "acc": float(np.mean(list(per_q.values())))}


def load_runs(runs_dir: Path, metric: str) -> dict[str, list[dict]]:
    by_key = defaultdict(list)
    for d in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        raw = read_json(d / "config.json")
        if not raw:
            continue
        cfg = raw.get("spec", raw)
        run = {"dir": d, "cfg": cfg, "evals": {}, "fq": {}}
        think = [r for e in sorted(d.glob("eval_think*")) if (r := _replicate(e, metric))]
        nothink = _replicate(d / "eval_nothink", metric)
        if think:
            run["evals"]["think"] = think
        if nothink:
            run["evals"]["nothink"] = [nothink]
        run["forgetting"] = read_json(d / "forgetting.json")
        for r in read_jsonl(d / "forgetting_preds.jsonl"):
            run["fq"].setdefault(r["id"].split("_")[0], {})[r["id"]] = float(r["correct"])
        run["train"] = read_json(d / "adapter" / "train_summary.json")
        run["curve"] = read_json(d / "checkpoint_curve.json")
        by_key[cfg["key"]].append(run)
    return by_key


def aggregate(runs: list[dict], mode: str) -> dict | None:
    trained = [r["evals"][mode] for r in runs if mode in r["evals"]]  # [training run][eval replicate]
    if not trained:
        return None
    run_pq = [seed_average([rep["per_q"] for rep in reps]) for reps in trained]
    pooled = seed_average(run_pq)
    meta = {p["id"]: p for p in trained[0][0]["preds"]}
    clusters = {q: meta[q]["db_id"] for q in pooled}
    ci = accuracy_ci(pooled, clusters)
    train_mean, train_std = mean_std([float(np.mean([rep["acc"] for rep in reps])) for reps in trained])
    infer_stds = [mean_std([rep["acc"] for rep in reps])[1] for reps in trained if len(reps) > 1]
    flat = [rep for reps in trained for rep in reps]
    cats = defaultdict(int)
    for rep in flat:
        for k, v in rep["summary"]["failure_categories"].items():
            cats[k] += v
    total = sum(cats.values())

    def by(field):
        g = defaultdict(list)
        for q, v in pooled.items():
            g[meta[q][field]].append(v)
        return {k: {"n": len(v), "ex": float(np.mean(v))} for k, v in sorted(g.items())}

    def avg(key):
        return mean_std([rep["summary"].get(key) for rep in flat])[0]

    return {
        "n_train_seeds": len(trained), "n_eval_seeds": max(len(r) for r in trained), "n": len(pooled),
        "ex_mean": train_mean, "ex_std": train_std, "infer_std": float(np.mean(infer_stds)) if infer_stds else None,
        "ci": ci["ci"], "ci_clustered": ci.get("ci_clustered"), "n_clusters": ci.get("n_clusters"),
        "pooled": pooled, "clusters": clusters,
        "valid": avg("valid_sql_rate"), "trunc": avg("truncation_rate"), "tokens": avg("avg_gen_tokens"),
        "latency": avg("median_latency_s"), "ts": avg("test_suite_accuracy"),
        "categories": {k: v / total for k, v in sorted(cats.items(), key=lambda x: -x[1])},
        "by_difficulty": by("difficulty"), "by_db": by("db_id"),
    }


def aggregate_forgetting(runs: list[dict]) -> dict:
    out = {}
    for bench in ("mmlu", "gsm8k"):
        per = [r["fq"][bench] for r in runs if bench in r["fq"]]
        if per:
            m, s = mean_std([float(np.mean(list(p.values()))) for p in per])
            out[bench] = {"mean": m, "std": s, "pooled": seed_average(per), "n_seeds": len(per)}
    return out


# ------------------------------------------------------------------ formatting
def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def ex_cell(a):
    if a is None:
        return "–"
    s = pct(a["ex_mean"])
    if a["ex_std"] is not None:
        s += f" ± {pct(a['ex_std'])}"
    elif a["infer_std"] is not None:
        s += f" ± {pct(a['infer_std'])}ⁱ"
    s += f" [{pct(a['ci'][0], 0)}, {pct(a['ci'][1], 0)}]"
    if a["ci_clustered"]:
        s += f" db:[{pct(a['ci_clustered'][0], 0)}, {pct(a['ci_clustered'][1], 0)}]"
    return s


def diff_cell(c):
    if not c or not c.get("n"):
        return "–"
    ci_main = c.get("ci_clustered") or c["ci"]
    sig = "**" if ci_main[0] > 0 or ci_main[1] < 0 else ""
    s = f"{sig}{100 * c['diff']:+.1f} pp{sig} [{100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}]"
    if c.get("ci_clustered"):
        s += f" db:[{100 * c['ci_clustered'][0]:+.1f}, {100 * c['ci_clustered'][1]:+.1f}]"
    s += f", p={c['p_value']:.3g}"
    if c.get("p_value_clustered") is not None:
        s += f", p_db={c['p_value_clustered']:.3g}"
    return s + f", n={c['n']}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--metric", choices=["ex", "ts"], default="ex", help="ex = dev DBs, ts = test-suite")
    ap.add_argument("--main_n", default="all", help="train-size tag of the headline runs")
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(Path(args.runs_dir), args.metric)
    agg = {(k, m): a for k, v in runs.items() for m in MODES if (a := aggregate(v, m))}
    fgt = {k: f for k, v in runs.items() if (f := aggregate_forgetting(v))}
    order = sorted(set(runs), key=lambda k: ({"base_student": 0, "teacher": 1}.get(k, 2), k))

    K = lambda mode: f"{mode}_n{args.main_n}_r{args.rank}"  # noqa: E731
    MAIN, SQL, R_UNV, S_UNV = K("reasoning"), K("sql_only"), K("reasoning_unverified"), K("sql_unverified")
    metric_name = "test-suite accuracy" if args.metric == "ts" else "execution accuracy (EX)"
    md = ["# Results report", "", f"Primary metric: **{metric_name}**.", ""]

    # --- teacher yield + data funnel
    ts = read_json(out / "teacher_stats.json")
    if ts:
        f = ts["funnel"]
        md += ["## Teacher data generation", "",
               table(["started", "finished", "teacher ever correct", "usable verified", *ts["pass_at"].keys(),
                      "never usable", "gain from resampling", "correct but unusable", "GPU-hours"],
                     [[f["questions_started"], f["questions_finished"], f["teacher_ever_correct"],
                       f["usable_verified"], *[pct(v) for v in ts["pass_at"].values()], pct(ts["never_usable"]),
                       pct(ts["gain_from_resampling"]), ts["correct_but_unusable_attempts"]["n"],
                       f"{ts['total_generation_hours']:.1f}"]]),
               "", f"Failed-attempt categories: {ts['failed_attempt_categories']}", ""]
    main_meta = (runs.get(MAIN, [{}])[0].get("cfg") or {}).get("data_meta") if MAIN in runs else None
    if main_meta and main_meta.get("funnel"):
        f = main_meta["funnel"]
        md += ["Dataset funnel for the headline runs: " + " → ".join(f"{k.replace('_', ' ')} {v}" for k, v in f.items()),
               ""]

    # --- main table
    rows = []
    for k in order:
        for m in MODES:
            a = agg.get((k, m))
            if a:
                n_tr = (runs[k][0]["cfg"].get("data_meta") or {}).get("n_train", "–")
                rows.append([label(k), m, n_tr, a["n_train_seeds"], a["n_eval_seeds"], a["n"], ex_cell(a),
                             pct(a["valid"]), pct(a["trunc"]), f"{a['tokens']:.0f}",
                             pct(a["ts"]) if args.metric == "ex" else "–"])
    md += [f"## {metric_name.capitalize()} on Spider dev", "",
           "Mean ± std across training seeds (ⁱ = across eval seeds, inference-only variance) "
           "[95% CI over questions] db:[95% CI clustered by database].", "",
           table(["model", "mode", "train ex.", "train seeds", "eval seeds", "dev q.", "accuracy %", "valid SQL %",
                  "truncated %", "avg tokens", "test-suite %"], rows), ""]

    # --- key comparisons
    def P(k, m):
        a = agg.get((k, m))
        return a["pooled"] if a else None

    clusters = next((a["clusters"] for a in agg.values()), None)
    comps = [
        ("Distillation gain, thinking", (MAIN, "think"), ("base_student", "think")),
        ("Distillation gain, answering directly", (MAIN, "nothink"), ("base_student", "nothink")),
        ("Reasoning vs SQL-only supervision, both answering directly", (MAIN, "nothink"), (SQL, "nothink")),
        ("Reasoning vs SQL-only supervision, both thinking", (MAIN, "think"), (SQL, "think")),
        ("Reasoning (thinking) vs SQL-only (direct), each in its natural mode", (MAIN, "think"), (SQL, "nothink")),
        ("Does thinking at inference help the distilled model?", (MAIN, "think"), (MAIN, "nothink")),
        ("Execution filtering, reasoning targets", (MAIN, "think"), (R_UNV, "think")),
        ("Execution filtering, SQL-only targets", (SQL, "nothink"), (S_UNV, "nothink")),
        ("Remaining gap: teacher vs distilled student", ("teacher", "think"), (MAIN, "think")),
        ("Quantisation cost: 4-bit fused vs bf16 + LoRA", ("student_fused_q4", "think"), (MAIN, "think")),
    ]
    comp_rows, comp_json = [], {}
    for name, a, b in comps:
        pa, pb = P(*a), P(*b)
        if pa and pb:
            c = paired_test(pa, pb, clusters)
            comp_json[name] = c
            comp_rows.append([name, f"{label(a[0])} ({a[1]})", f"{label(b[0])} ({b[1]})", diff_cell(c)])
    for m in MODES:  # 2x2 interaction: does verification matter more when reasoning is included?
        cells = [P(MAIN, m), P(SQL, m), P(R_UNV, m), P(S_UNV, m)]
        if all(cells):
            c = contrast(cells, [1, -1, -1, 1], clusters)
            name = f"Interaction ({m}): (reasoning − sql_only)_verified − (reasoning − sql_only)_unverified"
            comp_json[name] = c
            comp_rows.append([name, "2×2", "", diff_cell(c)])
    if comp_rows:
        md += ["## Key comparisons (paired, same questions)", "",
               "A − B. [question-level bootstrap CI] db:[database-clustered CI]; p = sign-flip permutation test "
               "(p_db flips whole databases). **Bold** = clustered CI excludes 0.", "",
               table(["question", "A", "B", "A − B"], comp_rows), ""]
    b, t, s = agg.get(("base_student", "think")), agg.get(("teacher", "think")), agg.get((MAIN, "think"))
    if b and t and s and t["ex_mean"] != b["ex_mean"]:
        md += [f"**Gap closed (thinking mode):** {(s['ex_mean'] - b['ex_mean']) / (t['ex_mean'] - b['ex_mean']):.0%} "
               "of the base → teacher gap.", ""]

    # --- breakdowns
    show = [(k, m) for k, m in [("base_student", "think"), (MAIN, "think"), (MAIN, "nothink"), (SQL, "nothink"),
                                ("student_fused_q4", "think"), ("teacher", "think")] if (k, m) in agg]
    if show:
        diffs = sorted({d for km in show for d in agg[km]["by_difficulty"]},
                       key=lambda d: ["easy", "medium", "hard", "extra", "unparsed"].index(d))
        md += ["## Accuracy by query difficulty", "",
               table(["difficulty", "n"] + [f"{label(k)} ({m})" for k, m in show],
                     [[d, agg[show[0]]["by_difficulty"].get(d, {}).get("n", "–")]
                      + [pct(agg[km]["by_difficulty"].get(d, {}).get("ex")) for km in show] for d in diffs]), ""]
        cats = sorted({c for km in show for c in agg[km]["categories"]})
        md += ["## Outcome categories (% of dev questions)", "",
               "Structural diffs (in `sql_preds.jsonl`) are heuristics: they show *where* a wrong query differs "
               "from gold, not *why* the model got it wrong.", "",
               table(["category"] + [f"{label(k)} ({m})" for k, m in show],
                     [[c] + [pct(agg[km]["categories"].get(c, 0)) for km in show] for c in cats]), ""]
        dbs = sorted({d for km in show for d in agg[km]["by_db"]})
        with (out / "ex_by_db.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["db_id", "n"] + [f"{k}|{m}" for k, m in show])
            for d in dbs:
                w.writerow([d, agg[show[0]]["by_db"].get(d, {}).get("n")]
                           + [agg[km]["by_db"].get(d, {}).get("ex") for km in show])
        md += [f"Per-database accuracy: `{out / 'ex_by_db.csv'}`", ""]

    # --- forgetting
    base_f = fgt.get("base_student", {})
    f_rows = []
    for k in order:
        f = fgt.get(k)
        if not f:
            continue
        cells = [label(k)]
        for bench in ("mmlu", "gsm8k"):
            if bench not in f:
                cells.append("–")
                continue
            v = pct(f[bench]["mean"]) + (f" ± {pct(f[bench]['std'])}" if f[bench]["std"] else "")
            if k != "base_student" and bench in base_f:
                v += " (" + diff_cell(paired_test(f[bench]["pooled"], base_f[bench]["pooled"])) + ")"
            cells.append(v)
        f_rows.append(cells)
    if f_rows:
        md += ["## Catastrophic forgetting", "", "Accuracy % (difference vs base on identical questions).", "",
               table(["model", "MMLU", "GSM8K"], f_rows), ""]

    # --- training cost (val loss next to accuracy: they measure different things)
    t_rows = []
    for k in order:
        tr = [r["train"] for r in runs[k] if r.get("train")]
        if tr:
            a = agg.get((k, "think"))
            t_rows.append([k, len(tr), tr[0]["iters"], f"{np.mean([x['wall_time_s'] for x in tr]) / 3600:.2f}",
                           f"{np.mean([x.get('peak_mem_gb') or 0 for x in tr]):.1f}",
                           f"{np.mean([x.get('final_val_loss') or np.nan for x in tr]):.3f}",
                           f"{np.mean([x.get('best_val_loss') or np.nan for x in tr]):.3f}", pct(a["ex_mean"]) if a else "–"])
    if t_rows:
        md += ["## Training cost and validation loss", "",
               "Validation loss is token-level likelihood of teacher targets; it need not track execution accuracy.",
               "", table(["run", "seeds", "iters", "hours", "peak mem GB", "final val loss", "best val loss",
                          "accuracy % (think)"], t_rows), ""]

    # --- benchmark
    bm = read_json(out / "benchmark.json")
    if bm:
        md += ["## Deployment benchmark", "",
               f"{bm['results'][0]['n_prompts']} dev prompts per model, thinking={bm['results'][0]['thinking']}. "
               "Accuracy on this small subset is indicative only; use the full-dev rows above for quality claims.", "",
               table(["model", "disk GB", "load s", "peak mem GB", "TTFT s", "gen tok/s", "median latency s",
                      "latency by output length", "acc. (bench prompts)"],
                     [[r["name"], f"{r['disk_gb']:.2f}" if r["disk_gb"] else "–", f"{r['load_s']:.1f}",
                       f"{r['peak_mem_gb']:.1f}", f"{r['median_ttft_s']:.2f}", f"{r['mean_gen_tps']:.0f}",
                       f"{r['median_latency_s']:.1f}",
                       "; ".join(f"{b}: {v['median_latency_s']:.1f}s (n={v['n']})"
                                 for b, v in r.get("latency_by_output_tokens", {}).items()),
                       pct(r["ex_on_bench_prompts"])] for r in bm["results"]]), ""]

    plots = make_plots(runs, agg, fgt, out, args)
    if plots:
        md += ["## Figures", ""] + [f"![{p}]({p})" for p in plots]

    (out / "report.md").write_text("\n".join(md) + "\n")
    strip = lambda a: {k: v for k, v in a.items() if k not in ("pooled", "clusters")}  # noqa: E731
    write_json(out / "summary.json", {
        "metric": args.metric,
        "runs": {f"{k}|{m}": strip(a) for (k, m), a in agg.items()},
        "comparisons": comp_json,
        "forgetting": {k: {b: strip(x) for b, x in f.items()} for k, f in fgt.items()},
    })
    print("\n".join(md))
    print(f"\nwrote {out}/report.md, summary.json and figures")


def make_plots(runs, agg, fgt, out: Path, args) -> list[str]:
    files = []
    base = {m: agg.get(("base_student", m)) for m in MODES}
    teacher = agg.get(("teacher", "think"))

    def lora(k):
        return runs[k][0]["cfg"].get("kind") == "lora"

    def n_of(c):
        return (c.get("data_meta") or {}).get("n_train") or 1

    def sweep(filter_fn, xfn, fname, xlabel):
        keys = [k for k in runs if lora(k) and filter_fn(runs[k][0]["cfg"])]
        if len(keys) < 2:
            return
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for m, style in (("think", "-o"), ("nothink", "--s")):
            pts = sorted((xfn(runs[k][0]["cfg"]), agg[(k, m)]) for k in keys if (k, m) in agg)
            if pts:
                ax.errorbar([p[0] for p in pts], [100 * p[1]["ex_mean"] for p in pts],
                            yerr=[[100 * (p[1]["ex_mean"] - p[1]["ci"][0]) for p in pts],
                                  [100 * (p[1]["ci"][1] - p[1]["ex_mean"]) for p in pts]],
                            fmt=style, capsize=3, label=f"distilled ({m})")
            if base[m]:
                ax.axhline(100 * base[m]["ex_mean"], ls=":", c="gray" if m == "think" else "silver",
                           label=f"base ({m})")
        if teacher:
            ax.axhline(100 * teacher["ex_mean"], ls="-.", c="k", label="teacher (think)")
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Accuracy (%), 95% CI")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
        files.append(fname)

    sweep(lambda c: c["mode"] == "reasoning" and c["rank"] == args.rank, n_of, "ex_vs_data_size.png",
          "Training examples")
    sweep(lambda c: c["mode"] == "reasoning" and f"n{args.main_n}_" in c["key"], lambda c: c["rank"],
          "ex_vs_rank.png", "LoRA rank")

    modes = ["reasoning", "sql_only", "reasoning_unverified", "sql_unverified"]
    keys = [f"{m}_n{args.main_n}_r{args.rank}" for m in modes]
    if any((k, m) in agg for k in keys for m in MODES):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        x = np.arange(len(keys))
        for j, m in enumerate(MODES):
            vals = [agg.get((k, m)) for k in keys]
            ax.bar(x + (j - .5) * .38, [100 * v["ex_mean"] if v else 0 for v in vals], .38,
                   yerr=[[100 * (v["ex_mean"] - v["ci"][0]) if v else 0 for v in vals],
                         [100 * (v["ci"][1] - v["ex_mean"]) if v else 0 for v in vals]], capsize=3, label=m)
        ax.set_xticks(x, [m.replace("_", "\n") for m in modes], fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("2×2 trace ablation (95% CI)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "trace_ablation.png", dpi=150)
        plt.close(fig)
        files.append("trace_ablation.png")

    base_mmlu = fgt.get("base_student", {}).get("mmlu", {}).get("mean")
    pts = [(runs[k][0]["cfg"], f["mmlu"]["mean"]) for k, f in fgt.items()
           if "mmlu" in f and lora(k) and runs[k][0]["cfg"]["mode"] == "reasoning"]
    if base_mmlu is not None and len(pts) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, (xf, xl, flt) in zip(axes, [(n_of, "Training examples", lambda c: c["rank"] == args.rank),
                                            (lambda c: c["rank"], "LoRA rank",
                                             lambda c: f"n{args.main_n}_" in c["key"])]):
            sel = sorted((xf(c), 100 * (v - base_mmlu)) for c, v in pts if flt(c))
            if sel:
                ax.plot([s[0] for s in sel], [s[1] for s in sel], "-o")
            ax.axhline(0, c="gray", ls=":")
            ax.set_xscale("log", base=2)
            ax.set_xlabel(xl)
            ax.set_ylabel("MMLU change vs base (pp)")
            ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "forgetting.png", dpi=150)
        plt.close(fig)
        files.append("forgetting.png")

    for k, rs in runs.items():
        curve = next((r["curve"] for r in rs if r.get("curve")), None)
        if curve:
            p = curve["points"]
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.plot([q["iter"] for q in p], [100 * q["ex"] for q in p], "-o", label="Spider accuracy")
            ax2 = ax.twinx()
            ax2.plot([q["iter"] for q in p], [100 * q["mmlu"] for q in p], "--s", c="C1", label="MMLU")
            ax.set_xlabel("Training step")
            ax.set_ylabel("Spider accuracy (%)")
            ax2.set_ylabel("MMLU (%)")
            fig.legend(loc="lower right", fontsize=8)
            ax.set_title(f"Specialisation vs forgetting: {k}", fontsize=9)
            fig.tight_layout()
            fname = f"curve_{k}.png"
            fig.savefig(out / fname, dpi=150)
            plt.close(fig)
            files.append(fname)
    return files


if __name__ == "__main__":
    main()
