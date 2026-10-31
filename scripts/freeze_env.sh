#!/usr/bin/env bash
# Pin the exact environment that produced your results.
set -euo pipefail
cd "$(dirname "$0")/.."
pip freeze --exclude-editable > requirements.lock
python -m src.env_info > results/environment.json
echo "wrote requirements.lock and results/environment.json"
