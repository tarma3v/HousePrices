#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/download_data.sh
poetry run python -m src.train_baseline
poetry run python -m src.train_ablations
poetry run jupyter nbconvert --execute --to notebook \
    --output-dir=outputs \
    notebooks/EDA.ipynb notebooks/baseline_review.ipynb notebooks/baseline_ablations.ipynb