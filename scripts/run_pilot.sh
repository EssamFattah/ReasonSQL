#!/usr/bin/env bash
# Phase 1: small, end-to-end, one seed. Shows real yield, trace lengths and timings before you commit
# days of compute. Every step is resumable: re-run the script after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

N_QUESTIONS=${N_QUESTIONS:-500}

python -m pytest -q
python -m src.prepare_data
python -m src.generate_teacher --limit "$N_QUESTIONS"
python -m src.teacher_stats
python -m src.build_sft --out_dir runs/_preview --show | tail -n 30   # eyeball one training example
python -m src.run_ablations --profile pilot
python -m src.summarize
echo "Pilot done -> results/report.md"
