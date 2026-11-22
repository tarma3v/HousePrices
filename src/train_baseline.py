import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor

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
KNN_NUM = ['LogArea', 'OverallQual', 'Age']
KNN_CAT = ['Neighborhood']

def linear_predictions(train, val, num=NUM, cat=CAT):
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
    ])
    model = make_pipeline(prep, LinearRegression())
    model.fit(train[num + cat], np.log(train[TARGET]))
    return np.exp(model.predict(val[num + cat]))

def knn_predictions(train, val, k=5):
    prep = ColumnTransformer([
        ('num', StandardScaler(), KNN_NUM),
        ('cat', OneHotEncoder(handle_unknown='ignore'), KNN_CAT),
    ])
    model = make_pipeline(prep, KNeighborsRegressor(n_neighbors=k))
    cols = KNN_NUM + KNN_CAT
    model.fit(train[cols], np.log(train[TARGET]))
    return np.exp(model.predict(val[cols]))

def get_predictions(df, cv):
    rows = []
    log_num = ['LogArea' if c == 'GrLivArea' else c for c in NUM]
    no_garage_area = [c for c in NUM if c != 'GarageArea']
    for tr, va in cv.split(df):
        train, val = df.iloc[tr], df.iloc[va]
        pred = pd.DataFrame({
            'linear': linear_predictions(train, val),
            'median': median_predictions_dummy(train, val),
            'hood': median_predictions_by_hood(train, val),
        }, index=val.index)
        pred['log_area'] = linear_predictions(train, val, num=log_num)
        pred['no_garage_area'] = linear_predictions(train, val, num=no_garage_area)
        pred['ordinal'] = linear_predictions(
            train, val, num=NUM + ['KitchenScore'], cat=['Neighborhood']
        )
        for k in (3, 5, 10, 20, 30):
            pred[f'knn_{k}'] = knn_predictions(train, val, k)
        pred['linear_with_knn_features'] = linear_predictions(train, val, KNN_NUM, KNN_CAT)
        pred['fold'] = len(rows) + 1
        rows.append(pred)
    return pd.concat(rows)

def get_config_cv():
    config_path = PROJECT_ROOT / 'configs' / 'baseline.yaml'
    return yaml.safe_load(config_path.read_text())['cv']

if __name__ == '__main__':
    df = load_train_df()
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['LogArea'] = np.log(df['GrLivArea'])
    df['KitchenScore'] = df['KitchenQual'].map(
    {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
)
    config_cv=get_config_cv()
    preds = get_predictions(df, KFold(**config_cv))

    output_path = PROJECT_ROOT / 'outputs'
    output_path.mkdir(exist_ok=True)
    preds.to_csv(output_path / 'baseline.csv')
