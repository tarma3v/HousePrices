#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f data/raw/data_description.txt && -f data/raw/train.csv && -f data/raw/test.csv && -f data/raw/sample_submission.csv ]]
then
    echo "Data already present"
    exit 0
fi

mkdir -p data/raw
poetry run kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw
poetry run python -m zipfile -e data/raw/house-prices-advanced-regression-techniques.zip data/raw
rm "data/raw/house-prices-advanced-regression-techniques.zip"
echo "Data is downloaded successfully"
