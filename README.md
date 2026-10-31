# Reasoning Distillation for Text-to-SQL

## Overview

I built this project to investigate whether a smaller language model can learn to solve Text-to-SQL problems by learning from the reasoning traces of a larger model.

The idea is to use **Qwen3-8B** as a teacher to generate reasoning traces and SQL queries, then use those examples to fine-tune **Qwen3-1.7B**.

I use the Spider dataset for training and evaluation, and check generated SQL by actually running it against the corresponding SQLite databases. This helps filter out incorrect answers before using them for training.

The student is fine-tuned using LoRA on Apple Silicon with MLX.

Beyond just measuring whether the student improves, I also wanted to investigate a few other things:
- Does training on reasoning actually help compared to training on SQL answers alone?
- Does checking the teacher's answers before training make a difference?
- How much training data and LoRA capacity does the student need?
- Does specialising the model for SQL affect its general abilities?
- How much faster and smaller is the student compared to the teacher?

## Results

*This section will be updated once the full experiments have been completed.*

The main results I'm interested in are:

1. **Distillation:** How much does the student improve compared to its original version, and how close does it get to the teacher?
2. **Reasoning:** Does including the teacher's reasoning lead to better SQL performance than training on SQL answers alone?
3. **Verification:** Does filtering training examples using SQL execution improve the student?
4. **Forgetting:** Does fine-tuning for Text-to-SQL affect general knowledge and mathematical reasoning?
5. **Deployment:** How much memory and inference speed do we gain by using a smaller, quantised model?

All comparisons are based on the same evaluation questions wherever possible, with confidence intervals and statistical tests used to assess the results.

## How it works

The project follows this general pipeline:

```text
Spider dataset
      |
      v
Prepare questions, schemas and databases
      |
      v
Qwen3-8B generates reasoning + SQL
      |
      v
Execute generated SQL against SQLite
      |
      v
Save and analyse teacher attempts
      |
      v
Build different training datasets
      |
      v
Fine-tune Qwen3-1.7B using LoRA
      |
      v
Evaluate SQL accuracy and general abilities
      |
      v
Fuse and quantise models
      |
      v
Compare performance, speed and memory
```

The teacher gets up to three attempts per question. Each generated SQL query is checked against the gold query by executing both on the actual database. Every attempt is saved, including unsuccessful ones, so they can also be used for analysis.

## Experiments

I organised the experiments around a few main questions.

### 1. Does reasoning distillation work?

Compare the original Qwen3-1.7B model, the fine-tuned student and the Qwen3-8B teacher.

The main metric is execution accuracy (EX), which measures how often the generated SQL returns the correct result.

### 2. Does reasoning supervision help?

I compare two training approaches:

- `reasoning`: the student learns from the teacher's reasoning and SQL.
- `sql_only`: the student learns from the final SQL answer only.

Both use the same questions and verified SQL answers. This makes it possible to isolate the effect of including reasoning in the training data.

I also evaluate both students with and without thinking enabled, to see whether learning from reasoning helps even when the student generates SQL directly.

### 3. Does execution verification matter?

I compare verified and unverified teacher outputs.

Verified examples must pass execution checks. Unverified examples use the teacher's first attempt that passes basic formatting and length checks, even if the SQL is incorrect.

Together with the reasoning experiment, this gives four training conditions:

| Mode | Reasoning | Execution verified |
|---|---|---|
| `reasoning` | Yes | Yes |
| `sql_only` | No | Yes |
| `reasoning_unverified` | Yes | No |
| `sql_unverified` | No | No |

The same questions are used across all four conditions.

### 4. Training data and LoRA experiments

I also investigate how the amount of training data and LoRA rank affect performance.

- Training data sizes: 250, 500, 1,000, 2,000 and the full available dataset.
- LoRA ranks: 4, 8, 16, 32 and 64.

### 5. Catastrophic forgetting

Fine-tuning a model for one specific task could affect its other abilities.

To investigate this, I evaluate the base and fine-tuned models on:

- **MMLU:** general knowledge across different subjects.
- **GSM8K:** mathematical reasoning.

I also evaluate intermediate checkpoints to see how these abilities change during training.

### 6. Deployment and efficiency

Finally, I compare different model configurations:

- Qwen3-8B (4-bit)
- Qwen3-1.7B (BF16)
- Fine-tuned student
- Fused student (BF16, 8-bit and 4-bit)

I measure model size, memory usage, loading time, time to first token (TTFT), generation speed and SQL execution accuracy.

## Project structure

| File | Description |
|---|---|
| `src/common.py` | Shared functions for prompts, model generation, SQL parsing, execution and verification. |
| `src/prepare_data.py` | Processes Spider data and extracts database schemas and example rows. |
| `src/generate_teacher.py` | Generates teacher reasoning traces and SQL, verifies answers and saves attempts. |
| `src/teacher_stats.py` | Analyses teacher performance, success rates and generation statistics. |
| `src/build_sft.py` | Creates the different training datasets and validation splits. |
| `src/train.py` | Configures and runs LoRA fine-tuning using MLX. |
| `src/evaluate_sql.py` | Evaluates SQL execution accuracy and generation performance. |
| `src/eval_forgetting.py` | Evaluates general knowledge and mathematical reasoning. |
| `src/eval_checkpoints.py` | Evaluates intermediate training checkpoints. |
| `src/benchmark.py` | Fuses, quantises and benchmarks the models. |
| `src/run_ablations.py` | Runs and coordinates the different experiments. |
| `src/fingerprint.py` | Tracks experiment configurations and prevents incompatible results from being mixed. |
| `src/stats.py` | Statistical tests, bootstrap confidence intervals and comparisons. |
| `src/summarize.py` | Collects experiment results and generates tables and figures. |
| `src/predict.py` | Allows users to ask questions about SQLite databases through the model. |
| `tests/` | Unit tests covering SQL execution, parsing, data preparation and experiment controls. |

## Setup

This project is designed to run on an Apple Silicon Mac using MLX.

16 GB of unified memory is recommended. Some settings may need to be reduced on machines with less memory.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Run the tests
python -m pytest -q
```

### Why MLX?

I chose MLX because the experiments are designed to run locally on Apple Silicon.

It uses unified memory, which makes it possible to run the quantised 8B teacher and fine-tune the smaller student on a Mac.

MLX also provides the tools needed for LoRA training, model fusion and quantisation.

The main trade-off is portability. Running the same experiments on NVIDIA hardware would require replacing the MLX-specific parts with something like Hugging Face Transformers and PEFT. Most of the data processing, SQL verification and statistical analysis is independent of the framework.

## Dataset

The project uses the [Spider dataset](https://yale-lily.github.io/spider).

Download `spider_data.zip` and extract it into `data/`:

```text
data/
└── spider_data/
    ├── train_spider.json
    ├── dev.json
    └── database/
        └── ...
```

Spider contains natural-language questions, SQL queries and the corresponding SQLite databases.

One important detail is that the development set uses databases that are not present in the training set. This means evaluation also tests whether the model can work with previously unseen database schemas.

## Running the experiments

### 1. Pilot run

I recommend starting with the pilot before running the full experiment.

```bash
bash scripts/run_pilot.sh
```

This runs a smaller version of the pipeline so that generation, training and evaluation can be checked before committing to longer runs.

Check the teacher statistics and generated report before moving on.

### 2. Full experiment

```bash
N_QUESTIONS=3000 bash scripts/run_full.sh
```

This runs the larger experiment, including teacher generation, the different training conditions, checkpoint evaluation and deployment benchmarks.

The scripts are designed to resume after interruptions. Running the same command again should continue from the existing results when the configuration hasn't changed.

### 3. Optional test-suite evaluation

For more reliable SQL evaluation, the project also supports the Spider test-suite databases.

Download them from [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval), then run:

```bash
TEST_SUITE_DIR=data/test_suite_database \
N_QUESTIONS=3000 bash scripts/run_full.sh
```

This checks predictions against multiple database variations rather than relying only on the original database contents.

### Useful commands

Preview the full experiment without running it:

```bash
python -m src.run_ablations --profile full --dry_run
```

Run a smaller experiment:

```bash
python -m src.run_ablations \
    --profile full \
    --seeds 0 1 \
    --eval_limit 500
```

Try the model on a database:

```bash
python -m src.predict \
    --db data/spider_data/database/concert_singer/concert_singer.sqlite \
    --question "How many singers are older than 30?" \
    --model artifacts/student_fused_q4 \
    --show_reasoning
```

## Some implementation details

### SQL verification

Generated SQL is executed against the actual SQLite database and compared with the gold query.

The database connection is read-only, and an SQLite authorizer restricts execution to permitted operations such as `SELECT` and `READ`. Writes, `ATTACH`, `PRAGMA` and temporary tables are blocked. Queries also have a 10-second timeout.

Results are compared as ordered lists when the outermost gold query contains `ORDER BY`. Otherwise, row order is ignored. Floating-point values are rounded to four decimal places.

### Selecting training examples

A teacher answer is considered usable only when:

- Its SQL executes correctly.
- The output isn't truncated.
- The SQL is not empty.
- The complete training example fits within the student's maximum sequence length.

This definition is shared between teacher generation, teacher statistics and dataset construction. It prevents examples from being accepted just because their SQL is correct when the full reasoning trace is too long for the student.

### Reproducibility

Each stage saves a fingerprint of its configuration and relevant inputs.

This helps prevent results generated with different settings from accidentally being combined. For example, changing the teacher model, training data, LoRA rank or evaluation settings will invalidate the relevant downstream outputs.

Training and evaluation seeds are also kept separate. This allows us to distinguish variation caused by training from variation caused by model generation.

### Statistical analysis

Results are reported using confidence intervals and paired comparisons because the models are evaluated on the same questions.

The project includes:
- Bootstrap confidence intervals.
- Database-clustered confidence intervals.
- Paired sign-flip permutation tests.
- Comparisons of the reasoning and verification conditions.

Database clustering is particularly relevant because questions sharing the same database schema aren't fully independent.

## Limitations

There are a few limitations worth keeping in mind.

- **Execution match isn't the same as semantic equivalence.** Two queries can return the same results on one database while behaving differently on another.
- **Correct SQL doesn't guarantee correct reasoning.** Verification checks the final SQL, not whether the teacher's explanation is logically sound.
- **Teacher verification uses the original training database.** Some incorrect queries may therefore pass by coincidence.
- **The unverified experiment uses the same questions.** It tests the effect of filtering incorrect targets, not the effect of simply having more training data.
- **The size and rank experiments use fewer training seeds**, so these results should be treated more as trends.
- **The project currently focuses on Spider and one model family.** Testing on a more challenging dataset such as BIRD would be a useful next step.

## Possible extensions

Some things I'd like to investigate further:

1. **Reasoning trace length:** Does the student actually benefit from the reasoning content, or does it simply benefit from seeing more training tokens?
2. **DPO:** Use the saved correct and incorrect teacher attempts to investigate preference-based training.
3. **More difficult SQL benchmarks:** Evaluate on BIRD to see how well the approach transfers to more complex databases.
4. **Additional forgetting benchmarks:** Include coding tasks to investigate whether SQL fine-tuning affects programming abilities.

## Acknowledgements and licences

- Spider dataset: CC BY-SA 4.0.
- Qwen3 models: Apache 2.0.

