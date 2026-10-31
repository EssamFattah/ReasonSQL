# Changelog

## v3.2.1

- **Python 3.9 support.** `benchmark.py` and `summarize.py` used `X | None` type hints, which Python 3.9 can't
  evaluate. That broke on macOS's built-in Python 3.9 (Command Line Tools). Both now use
  `from __future__ import annotations`. The full test suite passes on 3.9.25 and 3.12.

## v3.2: pinning is enforced, not just recorded

- **Model revisions are now enforced.** Every loader receives the exact local snapshot whose commit is recorded:
  `load_model`, `mlx_lm.lora` (via the training config) and `mlx_lm.fuse`. Previously the commit was recorded but
  `mlx_lm.load` was given the repo name, so the two could disagree. A snapshot is resolved once per process. A model
  can be pinned explicitly with `org/name@<commit>`. An unresolvable model fails loudly instead of silently
  loading something else.
- **Adapter identity hashes the weights.** `adapter_identity()` now combines the training-spec fingerprint with
  hashes of `adapters.safetensors` and `adapter_config.json`. SQL and forgetting evals are therefore invalidated if
  the weights change while the spec stays the same.
- The machine-specific model path is excluded from the training spec (the model identity stands in for it).
- 81 tests (from 77), including one asserting the path given to `mlx_lm.load` is the snapshot whose commit is recorded.

## v3.1: close the last reproducibility gaps before the pilot

- **Stale benchmark artifacts.** Fused and quantised models were reused whenever `config.json` existed, so a new
  adapter could be benchmarked against a model fused from an old one. Each artifact now stores a fingerprint of its
  inputs (base model revision, adapter spec + weights hash, quantisation bits and group size) and is **rebuilt** if
  they don't match. Derived artifacts are rebuilt rather than refused.
- **Official export changed string literals.** `export_official` collapsed all whitespace, so
  `'New  York'` became `'New York'`. The new `common.sql_one_line` collapses whitespace only outside literals and
  quoted identifiers, and drops comments (a `-- comment` on one line would swallow the rest of the query).
- **Test-suite databases are content-hashed** in the eval spec, limited to the databases the eval touches.
  Replacing the files under the same path now invalidates old results.
- **Model revisions pinned.** Hugging Face models are identified by repo + resolved commit hash (from the local
  cache, downloading with mlx_lm's file patterns if needed) in teacher, training and eval specs. Local model
  directories are identified by a hash of their weights.
- **Teacher eval cost.** In the full profile the teacher got the base student's 3 inference replicates
  (~30 h of 8B thinking evals). It now gets one, and the base student keeps 3.
- Training spec factored into `train.training_spec()`. Tests stub the HF revision lookup so they never need the network.
- 77 tests (from 67).

## v3: make the full run trustworthy

### Must-fix before the full run
- **One definition of a usable trace** (`common.is_usable`): correct, not truncated, and the full tokenised example
  fits `max_seq_length`. Previously the teacher stopped at the first *correct* attempt, even one that `build_sft`
  later dropped as truncated or too long, so usable data could be lower than the stats suggested. Generation
  (stop and resume), `teacher_stats` and `build_sft` now share the definition.
- **Token-length filtering** replaces the 12,000-character cap. Examples are measured in tokens, including the
  prompt, so mlx_lm can't silently truncate away a SQL target. `train.py` refuses a `--max_seq_length` below the
  dataset's limit.
- **Configuration fingerprints** (`src/fingerprint.py`) for traces, datasets, adapters, SQL evals and forgetting
  evals, chained upstream to downstream. Resuming with different settings now stops with a diff of what changed,
  instead of silently reusing results produced under another configuration. Run `config.json` is a fingerprinted
  identity and environment goes to `env.json`. The orchestrator always invokes each stage, so every stage validates itself.
- **Training seeds and evaluation seeds are separated.** Thinking-mode inference uses `--eval_seed`, shared by every
  model. Baselines no longer have fake "training seeds"; the base student gets 3 eval seeds to measure
  inference-only variance.
- **p-values** now come from a paired sign-flip permutation test (the old bootstrap-crossing-zero value wasn't a
  proper null-hypothesis p-value), and their calibration under the null is tested.

### Also fixed / added
- Database-clustered bootstrap CIs and cluster permutation p-values (Spider questions share schemas).
- 2×2 interaction contrast: (reasoning − sql_only)_verified − (reasoning − sql_only)_unverified.
- Ordered comparison only when the **outermost** gold query has `ORDER BY` (sqlglot), not any nested one.
- Optional test-suite scoring (`--test_suite_dir`), `summarize --metric ts`, and `official/pred.sql`/`gold.sql`
  export for the reference evaluator.
- Data funnel reported at every stage (teacher stats, dataset meta, report).
- Validation loss shown next to accuracy in the training table.
- Benchmark: whole-directory disk size, load time in the table, latency by output length, 100 prompts by default.
- `run_full.sh` is safe with macOS's bash 3.2. A missing `sqlglot` now fails with a clear install message.
- README: why MLX, sample config, hardware/funnel placeholders, failure-analysis guide, clustered-CI caveat.
- 67 tests (from 47).

### Checked and unchanged
- `run_full.sh` already passed `--limit "$N_QUESTIONS"` to `generate_teacher`; no change needed.

## v2

### Bug fixes
- **Interrupted training looked complete.** mlx_lm overwrites `adapters.safetensors` at every checkpoint, so a crashed
  run was treated as trained. Runs now finish with a `DONE` marker; runs without it are wiped and retrained.
- **SQL extraction** cut at the first `;` even inside string literals, and collapsed whitespace inside literals
  (`'New  York'` → `'New York'`). It also mis-parsed ```` ```sqlite ```` blocks. All three are fixed.
- **SQL safety.** Added a SQLite authorizer (SELECT/READ/functions only). `ATTACH`, `PRAGMA`, writes and temp tables are denied.
- **Confounded "unverified" ablation.** It used different questions from the verified set. It's now a 2×2 on matched questions.
- **Confounded inference settings.** SQL-only models were only evaluated without thinking and reasoning models only
  with thinking. Every model is now evaluated in both modes under one fixed protocol.
- **Order-dependent sampling.** Seeds are now per question and per attempt, so resuming doesn't change results.
- **Partial results looked complete.** Result files are written atomically and act as completion markers.

### New
- Teacher pass@k / never-solved / failure-category analysis (`teacher_stats.py`).
- Failure categories and `sqlglot` structural diffs for every prediction; difficulty buckets; per-database EX.
- Bootstrap CIs, paired bootstrap comparisons, seed aggregation (`stats.py`); 3-seed headline runs.
- `pilot` / `full` profiles; full dev set for headline runs.
- Latency, TTFT, throughput and peak memory in every eval; loss curves and peak memory from training.
- Forgetting: per-question outputs for paired CIs, MMLU tokenisation check, GSM8K answer-format tracking.
- Checkpoint curve: Spider EX and MMLU over training steps (`eval_checkpoints.py`).
- Deployment benchmark: fuse, quantise to 8/4-bit, and compare against the teacher (`benchmark.py`); full-dev eval of the 4-bit model.
- `predict.py` CLI demo.
- Full `config.json` per run (spec, hyperparameters, data hashes, packages, chip, RAM, git commit); `freeze_env.sh` lock file.
- 47 unit tests, including an end-to-end reporting test on synthetic runs.
