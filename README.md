# Reasoning Distillation for Text-to-SQL

**Can a 1.7B model learn SQL reasoning from an 8B teacher, and what does that transfer cost?**

This project distils the step-by-step reasoning of **Qwen3-8B** (teacher) into **Qwen3-1.7B** (student)
for text-to-SQL. It uses **execution-verified synthetic data** and **LoRA fine-tuning**, all on Apple Silicon with **MLX**.
It then measures what transferred, what it cost in general ability, and what the small model buys you at deployment time.

## Key findings

> _Fill in after running. Keep each claim tied to a paired comparison in `results/report.md`._
>
> 1. **Distillation:** the student goes from __% to __% execution accuracy (EX) on Spider dev (+__ pp, 95% CI [__, __]),
>    closing __% of the gap to the teacher (__%).
> 2. **Reasoning supervision:** at the same SQL and the same questions, training on reasoning gives +__ pp over SQL-only
>    targets, **even when the student answers without thinking** (+__ pp [__, __]).
> 3. **Execution filtering:** checking teacher targets by execution is worth +__ pp, mostly on __ queries.
> 4. **Forgetting:** MMLU changes by __ pp [__, __] and GSM8K by __ pp; the cost grows / doesn't grow with LoRA rank.
> 5. **Deployment:** the fused 4-bit student is __× smaller and __× faster than the teacher, at __ pp EX.

## Research questions and how each is answered

| # | Question | Experiment | Metric |
|---|---|---|---|
| 1 | Does distillation work, and how much of the teacher gap does it close? | base vs distilled vs teacher, 3 seeds | EX, paired bootstrap, gap closed |
| 2 | Is the *reasoning* doing the work? | `reasoning` vs `sql_only` (same questions, same SQL) | EX in both inference modes |
| 3 | Does reasoning supervision help even without thinking at inference? | both students answering directly | EX, no-think mode |
| 4 | Does execution verification matter? | verified vs first-attempt targets (same questions) | EX |
| 5 | How data-efficient is it, and does LoRA rank matter? | size 250 → all; rank 4 → 64 | EX curves |
| 6 | What does specialising cost? | MMLU + GSM8K for every adapter, plus checkpoint curves | Δ vs base, paired |
| 7 | What does the small model buy you? | teacher vs bf16 vs fused 8/4-bit | memory, size, TTFT, tok/s, EX |

## Pipeline

```
 Spider train (7k questions + SQLite DBs)
          │
          ▼
 ┌────────────────────┐  up to k=3 attempts / question  ┌─────────────────────────────────┐
 │ Teacher Qwen3-8B   │ ──────────────────────────────▶ │ Run pred + gold SQL on the real │
 │ (thinking mode)    │     <think>…</think> + SQL      │ DB (read-only, SELECT-only,     │
 └────────────────────┘                                 │ timeout). Log every attempt     │
                                                        └────────────────┬────────────────┘
                                                                         ▼
                    traces.jsonl ──▶ pass@k stats ──▶ 2×2 datasets on the SAME questions
                                                      {reasoning, sql_only} × {verified, unverified}
                                                                         │
                                                                         ▼
                                        LoRA SFT Qwen3-1.7B (loss on assistant turn only)
                                                                         │
            ┌──────────────────────────┬─────────────────────────────────┼───────────────────────────┐
            ▼                          ▼                                 ▼                           ▼
   Spider dev EX, thinking   Spider dev EX, direct        MMLU / GSM8K (forgetting)    fuse → quantise → benchmark
            └───────────── paired bootstrap CIs, seeds, failure analysis → results/report.md ────────┘
```

## Software components

```
 src/common.py ─── prompts · <think>/SQL parsing · sqlglot features · safe SQLite exec · scoring · MLX helpers
   ▲     ▲
   │     └──── src/stats.py (bootstrap, paired bootstrap, seed averaging)
   │                 ▲
 prepare_data ─▶ generate_teacher ─▶ teacher_stats              summarize ──▶ results/report.md, figures
                       │                                            ▲
                       ▼                                            │ reads runs/*/config.json + eval files
                   build_sft ─▶ train ─▶ evaluate_sql, eval_forgetting, eval_checkpoints
                       ▲          ▲           ▲
                       └──── run_ablations (profiles, seeds, resumable) ────┘
                                                     benchmark (fuse/quantise) · predict (CLI demo)
```

| File | Purpose |
|---|---|
| `src/common.py` | Prompt format, output parsing, SQL structure features, safe execution, failure categories, MLX streaming generation |
| `src/prepare_data.py` | Spider JSON + SQLite → prompt-ready `train_pool.jsonl` / `dev.jsonl` (pre-shuffled) |
| `src/generate_teacher.py` | Teacher rejection sampling with execution verification; resumable, deterministic per-attempt seeds |
| `src/teacher_stats.py` | pass@1..k, never-solved rate, failure categories, trace lengths, generation hours |
| `src/build_sft.py` | The 2×2 datasets, nested subsets, a fixed validation set, and data hashes |
| `src/train.py` | mlx_lm LoRA config, log parsing (loss curves, peak memory), `DONE` marker |
| `src/evaluate_sql.py` | EX, valid SQL, truncation, failure categories, EX by difficulty and database, latency, throughput, memory; resumable |
| `src/eval_forgetting.py` | MMLU (logit scoring, tokenisation-checked) and GSM8K (answer-format tracked), per-question outputs |
| `src/eval_checkpoints.py` | Spider EX + MMLU at every saved checkpoint (specialisation vs forgetting curve) |
| `src/benchmark.py` | Fuse, quantise to 8/4-bit, and benchmark each model in an isolated process |
| `src/run_ablations.py` | Orchestration with `pilot` / `full` profiles; separate training and evaluation seeds |
| `src/fingerprint.py` | Spec fingerprints: safe resumption, refuses to mix configurations |
| `src/stats.py` | Bootstrap and database-clustered CIs, sign-flip permutation tests, contrasts (2×2 interaction) |
| `src/summarize.py` | Tables with CIs, paired comparisons, data funnel, breakdowns, figures |
| `src/predict.py` | Ask any SQLite DB a question from the command line |
| `tests/` | 81 unit tests: parsing, SQL safety, ordering, test-suite scoring, dataset controls, fingerprints, statistical calibration, reporting |

## Setup

You need an Apple Silicon Mac. 16 GB RAM is comfortable; with 8 GB, lower `--max_seq_length` and `--batch_size`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 81 tests, no MLX/GPU needed
```

### Why MLX rather than PyTorch

The whole study runs on one laptop. MLX is Apple's array framework for Apple Silicon. It uses unified memory
(no host↔GPU copies), so an 8B 4-bit teacher and a bf16 1.7B student fit in 16 GB. `mlx-lm` provides LoRA
training, fusing and quantisation out of the box. PyTorch's MPS backend works, but it has weaker quantised-inference
support and more memory overhead on Macs. The trade-off is portability: to reproduce on NVIDIA you'd swap the MLX
helpers in `common.py` and `train.py` for Hugging Face `transformers` + `peft`. The data, verification,
statistics and reporting code is framework-independent.

**Data:** download `spider_data.zip` from the official Spider page (https://yale-lily.github.io/spider) and unzip it into `data/`
so you have `data/spider_data/{train_spider.json, dev.json, database/<db_id>/<db_id>.sqlite}`.
Spider's dev databases don't appear in training, so dev EX measures generalisation to **unseen schemas**.

## Running it

**Phase 1, pilot (do this first).** 500 teacher questions, one seed, 300 dev questions:

```bash
bash scripts/run_pilot.sh
```

Before committing days of compute, check the pilot's `results/teacher_stats.json` (yield, trace length, seconds per attempt) and `results/report.md`.

**Phase 2–3, full study:**

```bash
N_QUESTIONS=3000 bash scripts/run_full.sh
```

This runs the teacher on 3,000 questions, the `full` ablation profile, the checkpoint curve, the deployment
benchmark, a full-dev evaluation of the 4-bit model, the environment freeze and the report. Every step is
resumable: re-run the same command after an interruption.

**Optional, test-suite evaluation (recommended before publishing numbers).** Download the test-suite databases from
[taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) and pass the folder:

```bash
TEST_SUITE_DIR=data/test_suite_database N_QUESTIONS=3000 bash scripts/run_full.sh
```

Every prediction is then also checked on all database variants for its schema, and `results/test_suite/report.md`
reports the same tables using test-suite accuracy. Each eval also writes `official/pred.sql` and
`official/gold.sql`, so you can confirm headline numbers with the reference script
(`python evaluation.py --gold gold.sql --pred pred.sql --db <test_suite_database> --etype exec`).

**Resuming and changing settings.** Every stage writes a `spec.json` fingerprint next to its outputs. Re-running
with identical settings skips or resumes. Re-running with *different* settings (learning rate, rank, data, eval
protocol, more teacher traces, …) stops with a list of what changed rather than silently reusing old results.
For a new configuration, use a new `--runs_dir`.

Useful individual commands:

```bash
python -m src.run_ablations --profile full --dry_run                 # print the plan only
python -m src.run_ablations --profile full --seeds 0 1 --eval_limit 500   # cheaper variant
python -m src.predict --db data/spider_data/database/concert_singer/concert_singer.sqlite \
    --question "How many singers are older than 30?" --model artifacts/student_fused_q4 --show_reasoning
```

### What the profiles run

| | pilot | full |
|---|---|---|
| Groups | baselines, 2×2 trace ablation | + data size (250, 500, 1k, 2k, all), LoRA rank (4–64) |
| Training seeds | 0 | 0, 1, 2 for the 2×2; 0 for sweeps |
| Eval seeds (thinking) | 0 for every model | 0 for every model; base student also 0, 1, 2 (inference-only variance; not the teacher, ~10 h per eval) |
| Dev questions | 300 | all 1,034 for headline runs; first 300 for sweeps (same shuffled order, so still paired) |
| Forgetting | MMLU 300, GSM8K 100 | MMLU 1,000, GSM8K 300 |
| Training runs | 4 | 20 |

### Compute budget (rough, M-series MacBook Air)

| Step | Approximate cost |
|---|---|
| Teacher 8B-4bit, thinking, ~500–1,500 tokens/attempt | 20–60 s/attempt → 3k questions ≈ 1–2 days |
| LoRA 1.7B, 2k examples × 2 epochs | 1–3 h per run |
| Student eval, full dev, thinking / direct | ~3–4 h / ~0.5 h |
| Teacher eval, full dev, thinking | ~10 h |
| **Full profile total** | **several days of machine time**; use `--seeds 0 1` or `--eval_limit 500` to halve it |

Use `caffeinate -i` so the Mac doesn't sleep. Check measured timings in the pilot before planning the full run.

## Method

**Prompt.** A system instruction, the `CREATE TABLE` schema with 3 example rows per table, and the question.
The answer goes in a ` ```sql ` block.

**Execution verification.** The predicted and gold queries both run against the real SQLite file. The connection is
read-only, and a SQLite authorizer allows only `SELECT`/`READ`/functions/recursive CTEs, so `ATTACH`, `PRAGMA`, writes
and temp tables are all denied. There's a 10 s timeout. Result sets are compared as ordered lists only when the
**outermost** gold query has `ORDER BY` (checked with `sqlglot`, so an ordered subquery doesn't count), and as
multisets otherwise, with floats rounded to 4 decimal places. SQL extraction cuts at the first `;` outside string
literals, quoted identifiers and comments, and keeps whitespace inside literals.

**Usable trace: one definition, used everywhere.** `common.is_usable` = correct by execution, not truncated,
non-empty SQL, and the full tokenised training example (prompt + reasoning + SQL) fits in `max_seq_length` (4,096).
The teacher keeps sampling after a correct-but-unusable attempt. `teacher_stats` reports yield on this definition,
and `build_sft` trains only on it. Because length is checked in tokens, not characters, mlx_lm never truncates
an example and silently cuts off its SQL target.

**Failure categories.** Each prediction gets one of: `correct`, `syntax_error`, `missing_table`, `missing_column`,
`timeout`, `blocked`, `execution_error`, `no_sql`, `truncated`, `wrong_columns`, `empty_result`, `wrong_row_count`,
`wrong_values`. For wrong answers that still run, a `sqlglot` structural diff against the gold query flags which
aspect differs: `joins`, `filtering`, `aggregation`, `grouping`, `nesting`, `set_ops`, `ordering` or `distinct`.
This is heuristic: it says *where* the query differs, not *why*.

**Difficulty.** A Spider-style hardness proxy from the gold query's parsed structure. It's a weighted count of
joins, subqueries, set operations, GROUP BY/HAVING, aggregation and ORDER BY, bucketed into easy, medium, hard
and extra.

**Rejection sampling.** Up to 3 teacher attempts per question at Qwen3's thinking settings (T=0.6, top-p=0.95,
top-k=20), stopping at the first verified one. Each attempt has its own deterministic seed. Every attempt is logged,
which is what enables pass@k analysis and the unverified ablations.

**Training.** The target is `<think>\n{reasoning}\n</think>\n\n```sql\n{SQL}\n```` in Qwen3's chat template. SQL-only
targets use an empty think block, which is exactly Qwen3's non-thinking format. `mask_prompt: true` puts loss on the
assistant turn only. LoRA is on all layers with a bf16 base, cosine LR with warmup and gradient checkpointing.
Checkpoints are saved on the same steps as validation, and a `DONE` marker tells finished runs apart from interrupted ones.

### Controls

- **The 2×2 trace ablation uses the same questions in every cell.**
  - The reasoning rows differ only in whether the reasoning is included; the SQL is identical.
  - The verification columns differ only in whether targets were checked. "Unverified" takes the teacher's first
    attempt that passes cheap format filters, which is what you'd get without execution.
  - `frac_targets_correct` and `frac_targets_differ_from_verified` in each `data/meta.json` show exactly how much changed.
- **Fixed splits.** The validation set and question order are fixed by `--split_seed` and shared by every run.
  Training subsets are nested, and the train/valid ID hashes are recorded.
- **Training seeds and evaluation seeds are separate.**
  - A training seed changes LoRA initialisation and data order, never the data.
  - Thinking-mode inference uses its own `--eval_seed`, and every model is evaluated with the *same* eval seeds.
    So the spread across training seeds reflects training, not sampling luck.
  - The base student is also evaluated with 3 eval seeds, which measures inference-only variance (ⁱ in the
    tables) for comparison.
  - The same seed doesn't make two different models draw identical samples, but it does share the random-number stream.
- **Inference protocol.** Thinking mode samples at T=0.6 (greedy decoding causes repetition loops when Qwen3 thinks);
  direct mode is greedy. The protocol is identical for every model, and **every adapter is evaluated in both modes**,
  so models are only compared within the same mode. Per-question seeds make evaluation resumable and order-independent.
- **Configuration safety.** Fingerprints chain upstream to downstream: traces → dataset → adapter → evaluation.
  Changing anything upstream invalidates everything that depends on it, and a stage refuses to reuse outputs made
  with different settings.
  Hugging Face models are loaded from the exact snapshot whose commit is recorded (pin one with `org/name@<commit>`), local models, adapter weights and test-suite databases are
  content-hashed, and fused/quantised benchmark models are rebuilt whenever their adapter or quantisation
  settings change.
- **LoRA scale is fixed across ranks.** mlx_lm has no 1/r factor, so this keeps alpha/r constant.
- **Forgetting** uses the same questions (fixed shuffle seed) and the same scoring for every model. MMLU letter tokens
  are checked to tokenise as single tokens after the prompt, and GSM8K tracks how often the requested `Answer:` format
  was used versus a last-number fallback.

### Statistics

- **Accuracy** is reported as mean ± std across training seeds, a 95% bootstrap CI over questions, and a
  **database-clustered** 95% CI that resamples whole Spider databases. Questions sharing a schema aren't independent,
  and the claim is generalisation to unseen schemas, so the clustered interval is the honest one.
- **Comparisons** are paired on the same dev questions:
  - Bootstrap CIs of the difference, at question level and clustered by database.
  - p-values from a paired **sign-flip permutation test**. Under H0 each question's difference is equally likely to
    be positive or negative. The clustered version flips whole databases. With one seed and binary outcomes, the
    question-level test is a Monte Carlo exact McNemar test.
  - A test checks that the p-values are calibrated under the null (about 5% false positives at α = 0.05).
- **The 2×2 interaction** (does verification matter more when reasoning is included?) is the per-question contrast
  (R_ver − S_ver) − (R_unv − S_unv), with the same CIs and test.
- **Caveat:** Spider dev has only 20 databases, so the clustered bootstrap is itself noisy. Treat it as a robustness
  check alongside the question-level interval, not a replacement. Bold in the report means the clustered CI excludes 0.
- **Sample size:** a 300-question eval has roughly ±5.5 pp CI width at 60% accuracy; the full 1,034 gives about ±3 pp
  before clustering. Paired comparisons are much tighter than two independent CIs.

## Outputs

| File | Contents |
|---|---|
| `results/report.md` | Every table below, plus figures |
| `results/teacher_stats.json` | Data funnel, pass@1/2/3 on usable traces, never usable, correct-but-unusable, failure categories, token-length percentiles, GPU-hours |
| `results/summary.json` | Machine-readable aggregates and comparisons |
| `results/ex_by_db.csv` | Per-database accuracy for the headline models |
| `results/benchmark.json` | Size, memory, TTFT, throughput, latency per deployment variant |
| `results/*.png` | EX vs data size, EX vs rank (both modes, seed error bars), 2×2 trace ablation (CIs), forgetting vs size/rank, checkpoint curve |
| `runs/<run>/config.json` | Fingerprinted run identity (spec, hyperparameters, data fingerprint, data funnel) |
| `runs/<run>/env.json`, `*/spec.json` | Environment (packages, chip, RAM, git commit); per-stage fingerprints |
| `runs/<run>/eval_*/official/` | `pred.sql` / `gold.sql` for the reference Spider evaluation scripts |

A run's `config.json` looks like this (abridged):

```json
{
  "fingerprint": "3f9c1a7e02bd",
  "spec": {
    "kind": "lora", "key": "reasoning_nall_r16", "mode": "reasoning", "n_train": -1, "rank": 16, "seed": 1,
    "student": "Qwen/Qwen3-1.7B", "teacher": "mlx-community/Qwen3-8B-4bit", "dataset": "spider",
    "epochs": 2.0, "lr": 2e-05, "batch_size": 2, "max_seq_length": 4096,
    "data_fingerprint": "a81e5c09d4f2",
    "data_meta": {"n_train": 2050, "n_valid": 100,
                  "funnel": {"questions_with_traces": 3000, "teacher_ever_correct": 2310,
                             "usable_verified": 2150, "valid": 100, "train": 2050}}
  }
}
```

_(Numbers are illustrative; they show the structure, not results.)_

## Results

_Generated in `results/report.md`; copy the headline tables here._

**Hardware:** _e.g. MacBook Air M3, 16 GB, macOS 15.x, mlx 0.x, mlx-lm 0.x (copy from `results/environment.json`)_

**Data funnel:** _3,000 questions → __ teacher ever correct → __ usable verified → __ train / 100 valid_

| Model | Mode | Spider dev EX (95% CI) | MMLU Δ | GSM8K Δ | Peak mem | tok/s |
|---|---|---|---|---|---|---|
| Qwen3-1.7B base | think / direct | | — | — | | |
| Qwen3-8B teacher (4-bit) | think | | — | — | | |
| **Distilled 1.7B, reasoning** | think / direct | | | | | |
| Distilled 1.7B, SQL-only | direct | | | | | |
| Distilled 1.7B, fused 4-bit | think | | | | | |

### Examples and failure analysis

Pick real cases from `runs/*/eval_*/sql_preds.jsonl` (each record has the question, raw output, predicted and gold SQL,
outcome category and structural diff). `predict.py --show_reasoning` reproduces one interactively. Useful ones:

1. A question the base student gets wrong and the distilled student gets right, with both reasoning traces.
2. A remaining failure, with its category, e.g. `wrong_values` + struct diff `["joins"]`: the query runs but joins
   the wrong table.
3. A teacher trace whose reasoning is flawed but whose SQL passed verification (the "correct SQL ≠ correct
   reasoning" limitation).
4. A prediction that's correct on the dev DB but wrong under the test suite (an execution-match false positive).

The structural diff shows *where* a wrong query differs from gold (joins, filtering, aggregation…), not *why* the
model made the mistake. Read the trace before drawing conclusions.

## Limitations

- **Execution match ≠ semantic equivalence.** Two queries can agree on one database instance and differ on another.
  Test-suite scoring (`--test_suite_dir`) reduces this for *evaluation*. Our built-in check requires matching
  results on every variant but doesn't reproduce the reference script's value/column-order normalisation, so
  confirm final numbers with the exported `official/` files. Teacher *verification* still uses the single training
  DB, so some kept traces are false positives.
- **Correct SQL doesn't mean correct reasoning.** Verification checks the final query, not the trace, so some kept
  traces will contain flawed reasoning that happened to reach the right SQL.
- **The unverified cells use matched questions.** They answer "does filtering wrong targets help?", not "does
  unfiltered data help because there's *more* of it?". That's a separate experiment.
- **Seeds:** 3 seeds on headline runs; the size and rank sweeps use one seed and should be read as trends.
- **Structural diff and difficulty** are heuristics built on `sqlglot`; Spider's official hardness uses its own parser.
- **One benchmark (Spider 1.0) and one model family.** BIRD would test harder, messier schemas.

## Extensions (after the core results are done)

1. **Trace-length buckets**, the most interesting open question: is the student learning from the reasoning
   content, or just from more training tokens? Compare short, medium and long verified traces at equal example counts.
2. **DPO** on the logged correct/incorrect teacher attempts. Build pairs carefully: separate syntax errors,
   execution errors and semantically wrong results, and compare SFT vs SFT+DPO.
3. A coding benchmark (MBPP/HumanEval subset) for forgetting, and BIRD for harder SQL.

## Interview prep: things to be able to explain

LoRA and what rank controls · why alpha/r matters · teacher-generated data vs human labels · execution accuracy vs
exact match · why verification still has false positives · why a model can get the right SQL with wrong reasoning ·
why catastrophic forgetting happens and why LoRA limits it · paired vs unpaired tests · how quantisation changes
memory, speed and accuracy.

## Licence notes

Spider is CC BY-SA 4.0. Qwen3 models are Apache 2.0.
# ReasonSQL
