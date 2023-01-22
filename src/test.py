from datetime import datetime
from pathlib import Path

# import imblearn
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
# import seaborn as sns
import sklearn
import gc
# import tqdm
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

def create_folds(df, target, n_s=5, n_grp=10000):
    skf = StratifiedKFold(n_splits=n_s)
    grp_target = pd.cut(target, n_grp, labels=False)
    return skf.split(grp_target, grp_target)

MODEL_NAMES = ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']
NFOLDS = 5

test_df = pd.read_csv('../data/test.csv', nrows=50000) ## UNCOMMENT FOR FULL DATA

all_y_preds = []

for MODEL_NAME in MODEL_NAMES:
    print(f"Testing {MODEL_NAME}")

    y_preds = np.zeros((NFOLDS, test_df.shape[0]))

    for fold_n in range(NFOLDS):
        print(f"Fold {fold_n}")
        file_to_load = '../models/'+MODEL_NAME+'_fold'+str(fold_n)+'.sav'
        loaded_model = pickle.load(open(file_to_load, 'rb'))
        y_preds[fold_n, :] =  loaded_model.predict(test_df)

        gc.collect()
    all_y_preds.append(y_preds)


sub = pd.read_csv('/home/jovyan/hfactory_magic_folders/hi__paris_hackathon/building_energy_efficiency/datasets/sample_submission_sent.csv', nrows=50000)
sub.energy_consumption_per_annum = np.mean(np.mean(all_y_preds, axis=0),axis=0)
sub.to_csv('submission_6.csv', index=False)