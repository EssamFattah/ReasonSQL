#!/usr/bin/env bash
# Phase 2 + 3: full study. Days of machine time on a laptop; see the compute table in the README.
# Resumable: re-run after any interruption. Every stage validates its settings against what's on
# disk and stops (instead of silently mixing results) if anything changed.
#
#   N_QUESTIONS=3000 bash scripts/run_full.sh
#   TEST_SUITE_DIR=data/test_suite_database N_QUESTIONS=3000 bash scripts/run_full.sh   # + test-suite scoring
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results

N_QUESTIONS=${N_QUESTIONS:-3000}
MAIN_RUN=${MAIN_RUN:-runs/reasoning_nall_r16_s0}
TS_ARGS=()
if [[ -n "${TEST_SUITE_DIR:-}" ]]; then TS_ARGS=(--test_suite_dir "$TEST_SUITE_DIR"); fi

python -m src.generate_teacher --limit "$N_QUESTIONS"
python -m src.teacher_stats
python -m src.run_ablations --profile full ${TS_ARGS[@]+"${TS_ARGS[@]}"}
python -m src.eval_checkpoints --run_dir "$MAIN_RUN"
python -m src.benchmark --adapter_path "$MAIN_RUN/adapter"

# Full-dev quality check of the deployable 4-bit model (appears in the report as student_fused_q4)
python - <<'EOF'
from src import fingerprint as fpr
ident = {"kind": "deployed", "key": "student_fused_q4", "model": "artifacts/student_fused_q4"}
fpr.check("runs/student_fused_q4/config.json", ident, "run student_fused_q4")
fpr.save("runs/student_fused_q4/config.json", ident)
EOF
python -m src.evaluate_sql --model artifacts/student_fused_q4 --thinking --eval_seed 0 \
    --out_dir runs/student_fused_q4/eval_think_e0 ${TS_ARGS[@]+"${TS_ARGS[@]}"}
python -m src.evaluate_sql --model artifacts/student_fused_q4 --no-thinking \
    --out_dir runs/student_fused_q4/eval_nothink ${TS_ARGS[@]+"${TS_ARGS[@]}"}

bash scripts/freeze_env.sh
python -m src.summarize
if [[ -n "${TEST_SUITE_DIR:-}" ]]; then python -m src.summarize --metric ts --out_dir results/test_suite; fi
echo "Done -> results/report.md"
