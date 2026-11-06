import pandas as pd
from sklearn.model_selection import train_test_split
from src.constants import (
    DESCRIPTION_PATH, HOLDOUT_SEED, HOLDOUT_SHARE, ID_COL, TEST_PATH, TRAIN_PATH,
)

def load_full_df(path = TRAIN_PATH) -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH, index_col=ID_COL)

def split_train_holdout() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_full_df()
    return train_test_split(df, test_size=HOLDOUT_SHARE, random_state=HOLDOUT_SEED)

def load_closed_test_df() -> pd.DataFrame:
    load_full_df(TEST_PATH)

def load_train_df() -> pd.DataFrame:
    return split_train_holdout()[0]

def load_description() -> str:
    with open(DESCRIPTION_PATH, 'r') as file:
        file_content = file.read()
    return file_content