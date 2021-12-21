import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor


#__file__ = Path('submissions') /  'my_submission' /  'estimator.py'


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    X['workingday'] = X['weekday'].item==5 or X['weekday'].item==6 or X['isHoliday']==True
    X['rainingday'] = X['ww']>19
    X = X.drop(columns=["ww"])

    # Finally we can drop the original columns from the dataframe
    return X

def _merge_external_data(X):
    
    ext_cols = ['date', 't', 'ww', 'isHoliday', 'td', 'season', 'u', 'ff']
    
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    # Là on peut choisir plus de données dans la base de données météo
    X = pd.merge_asof(X.sort_values('date'), df_ext[ext_cols].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    
    pca=PCA(n_components=1)
    X['temperature']=pca.fit_transform(X[['t', 'td']])
    return(X)

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    merge_data = FunctionTransformer(_merge_external_data, validate=False)
    #data_selector = FunctionTransformer(_select_data)
    date_cols = ['year', 'day']
    numeric_cols = ['t', 'td', 'u', 'ff', 'longitude', 'latitude']

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    #categorical_encoder = OrdinalEncoder()

    categorical_cols = ["counter_id", "site_id", "rainingday", "isHoliday", "workingday", "year"]

    
    preprocessor = ColumnTransformer(
        [
        ("date", OneHotEncoder(handle_unknown="ignore", sparse=False), date_cols),
        ("month_sin", sin_transformer(12), ["month"]),
        ("month_cos", cos_transformer(12), ["month"]),
        ("weekday_sin", sin_transformer(7), ["weekday"]),
        ("weekday_cos", cos_transformer(7), ["weekday"]),
        ("hour_sin", sin_transformer(24), ["hour"]),
        ("hour_cos", cos_transformer(24), ["hour"]),
        ("cat", categorical_encoder, categorical_cols),
        ('numeric', 'passthrough', numeric_cols)
        ]
    )
    regressor = RandomForestRegressor(max_depth=50)
    #regressor = GradientBoostingRegressor(learning_rate=0.09, n_estimators=500, max_depth=4)
    #regressor = RidgeCV()
    #regressor = DecisionTreeRegressor(max_depth=47, max_features='sqrt', max_leaf_nodes=150000)
    '''regressor = LGBMRegressor(boosting_type='gbdt', class_weight=None,
              colsample_bytree=0.6746393485503049, importance_type='split',
              learning_rate=0.03158974434726661, max_bin=55, max_depth=-1,
              min_child_samples=159, min_child_weight=0.001, min_split_gain=0.0,
              n_estimators=1458, n_jobs=-1, num_leaves=196, objective=None,
              random_state=18, reg_alpha=0.23417614793823338,
              reg_lambda=0.33890027779706655,
              subsample=0.5712459474269626, subsample_for_bin=200000,
              subsample_freq=1)
'''
    pipe = make_pipeline(merge_data, date_encoder, preprocessor, regressor)

    return pipe
