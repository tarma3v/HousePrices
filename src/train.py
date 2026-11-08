import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.constants import PROJECT_ROOT, TARGET
from src.utils import load_train_df

def median_predictions_dummy(train, val):
    median_default = train[TARGET].median()
    return np.full(len(val), median_default)

def median_predictions_by_hood(train, val):
    median_default = train[TARGET].median()
    by_area = train.groupby('Neighborhood')[TARGET].median()
    local = val['Neighborhood'].map(by_area).fillna(median_default)
    return local

NUM = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'LotFrontage', 'Age']
CAT = ['Neighborhood', 'KitchenQual']

def linear_predictions(train, val):
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT),
    ])
    model = make_pipeline(prep, LinearRegression())
    model.fit(train[NUM + CAT], np.log(train[TARGET]))
    return np.exp(model.predict(val[NUM + CAT]))

def get_predictions(df, cv):
    rows = []
    for tr, va in cv.split(df):
        train, val = df.iloc[tr], df.iloc[va]
        pred = pd.DataFrame({
            'linear': linear_predictions(train, val),
            'median': median_predictions_dummy(train, val),
            'hood': median_predictions_by_hood(train, val),
        }, index=val.index)
        pred['fold'] = len(rows) + 1
        rows.append(pred)
    return pd.concat(rows)

def get_config_cv():
    config_path = PROJECT_ROOT / 'configs' / 'baseline.yaml'
    return yaml.safe_load(config_path.read_text())['cv']

if __name__ == '__main__':
    df = load_train_df()
    df['Age'] = df['YrSold'] - df['YearBuilt']

    config_cv=get_config_cv()
    preds = get_predictions(df, KFold(**config_cv))

    output_path = PROJECT_ROOT / 'outputs'
    output_path.mkdir(exist_ok=True)
    preds.to_csv(output_path / 'baseline.csv')
