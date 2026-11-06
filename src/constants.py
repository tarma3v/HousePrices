from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"
DESCRIPTION_PATH = RAW_DIR / "data_description.txt"

ID_COL = "Id"
TARGET = "SalePrice"
HOLDOUT_SHARE = 0.15
HOLDOUT_SEED = 42