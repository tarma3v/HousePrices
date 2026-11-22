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

NUM = ['OverallQual', 'LogArea', 'GarageCars', 'GarageArea', 'LotFrontage', 'Age', 
       'TotalBsmtSF', 'RemodelYears']
CAT = ['Neighborhood', 'KitchenQual']
KNN_NUM = ['LogArea', 'OverallQual', 'Age', 'TotalBsmtSF', 'RemodelYears']
KNN_CAT = ['Neighborhood']

def linear_predictions(train, val, num=NUM, cat=CAT):
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
    ])
    model = make_pipeline(prep, LinearRegression())
    model.fit(train[num + cat], np.log(train[TARGET]))
    return np.exp(model.predict(val[num + cat]))

def knn_predictions(train, val, num=KNN_NUM, cat=KNN_CAT, k=20):
    prep = ColumnTransformer([
        ('num', StandardScaler(), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
    ])
    model = make_pipeline(prep, KNeighborsRegressor(n_neighbors=k))
    cols = num + cat
    model.fit(train[cols], np.log(train[TARGET]))
    return np.exp(model.predict(val[cols]))

def get_predictions(df, cv):
    rows = []
    for tr, va in cv.split(df):
        train, val = df.iloc[tr], df.iloc[va]
        pred = pd.DataFrame(index=val.index)
        pred["linear"] = linear_predictions(train, val, NUM, CAT)
        for feature in NUM+CAT:
            linear_num = [x for x in NUM if x != feature]
            linear_cat = [x for x in CAT if x != feature]
            pred[f"linear_no_{feature}"] = linear_predictions(train, val, linear_num, linear_cat)

        for k in (3, 5, 10, 20, 30):
            pred[f"knn_k={k}"] = knn_predictions(train, val, KNN_NUM, KNN_CAT, k)
            for feature in KNN_NUM + KNN_CAT:
                knn_num = [x for x in KNN_NUM if x != feature]
                knn_cat = [x for x in KNN_CAT if x != feature]
                pred[f"knn_no_{feature}_k={k}"] = knn_predictions(train, val, knn_num, knn_cat, k)
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
    remodel_year = df['YearRemodAdd']
    remodel_correct = remodel_year.between(df['YearBuilt'], df['YrSold'])
    df['RemodelYears'] = (remodel_year - df['YearBuilt']).where(remodel_correct, 0)
    df['KitchenScore'] = df['KitchenQual'].map(
        {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    )
    config_cv=get_config_cv()
    preds = get_predictions(df, KFold(**config_cv))

    output_path = PROJECT_ROOT / 'outputs'
    output_path.mkdir(exist_ok=True)
    preds.to_csv(output_path / 'ablations.csv')
