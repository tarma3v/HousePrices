#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/download_data.sh
poetry run python -m src.train
poetry run jupyter nbconvert --execute --to notebook \
    --output-dir=outputs \
    notebooks/EDA.ipynb notebooks/baseline_review.ipynb