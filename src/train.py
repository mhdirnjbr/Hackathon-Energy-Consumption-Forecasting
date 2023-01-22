from datetime import datetime
from pathlib import Path

import imblearn
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import gc
import tqdm
import re
from collections import defaultdict

from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import Ridge
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import Pool, CatBoostRegressor

from sklearn.model_selection import KFold
from sklearn import (
    linear_model,
    metrics,
    model_selection,
)
from sklearn.metrics import explained_variance_score

import matplotlib.pyplot as plt
import pickle
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def create_folds(df, target, n_s=5, n_grp=10000):
    skf = StratifiedKFold(n_splits=n_s)
    grp_target = pd.cut(target, n_grp, labels=False)
    return skf.split(grp_target, grp_target)

train_labels = pd.read_csv('/kaggle/input/hi-paris-2023/train/train_labels_sent.csv').energy_consumption_per_annum

MODEL_NAMES = ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']
STACKING_MODEL = 'Ridge'
NFOLDS = 5
NESTIMATORS = 100
NESTIMATORS_STACK = 1000

df = pd.read_csv('../data/train.csv', nrows=10000) ## UNCOMMENT FOR FULL DATA
test_df = pd.read_csv('../data/test.csv', nrows=10000) ## UNCOMMENT FOR FULL DATA

X_train = df[(train_labels< 1200) & (train_labels >=0)]
ys = train_labels[(train_labels< 1200) & (train_labels >=0)]
splits = list(create_folds(X_train, ys, n_s=NFOLDS, n_grp=1000))
X_test = test_df

all_y_preds = []
all_y_oof = []
train_scores = []

for MODEL_NAME in MODEL_NAMES:
    print(f"Training {MODEL_NAME}")

    for fold_n in range(NFOLDS):
        train_index, valid_index = splits[fold_n]
        print(f"Fold {fold_n}")
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_tr, y_val = ys.iloc[train_index], ys.iloc[valid_index]  
        
        file_to_save = '../models/'+MODEL_NAME+'_fold'+str(fold_n)+'.sav'
        
        if MODEL_NAME=='XGBRegressor':
            clf = str_to_class(MODEL_NAME)(tree_method="hist", enable_categorical=True, n_estimators=NESTIMATORS)
            clf.fit(X_tr, y_tr)
            
        elif MODEL_NAME=='LGBMRegressor': 
            
            dtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
            lgb_params = {
                    'n_jobs': -1,
                    'verbosity': -1,
                    'n_estimators': NESTIMATORS,
                }
            clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=200)
            
        elif MODEL_NAME=='CatBoostRegressor':
            clf = str_to_class(MODEL_NAME)(n_estimators=NESTIMATORS)
            clf.fit(X_tr.to_numpy(), y_tr.to_numpy(), verbose=0)
        pickle.dump(clf, open(file_to_save, 'wb'))