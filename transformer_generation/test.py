import random

import lightgbm
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold

from feature_generate_memory import *
import numpy as np
import pandas as pd


def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    rae = np.sum([np.abs(y_hat[i] - y[i]) for i in range(len(y))]) / np.sum(
        [np.abs(y_mean - y[i]) for i in range(len(y))])
    res = 1 - rae
    return res
name = 'PimaIndian'
path = fr"D:/python files/pythonProject3/DQN_my2/data/{name}.csv"
data = pd.read_csv(path)

OPS = ["add", "subtract", "multiply", "divide", "abss", 'square', 'inverse', 'log', 'sqrt', 'power3', 'pass']
score_list = []
for num in tqdm(range(200)):
    new_fe = data.copy()
    pass_list = []
    count_list = np.zeros(data.shape[1])
    for i in range(6):
        for j in range(data.shape[1] - 1):
            if j in pass_list:
                continue
            a = random.randint(0, 10)
            op = OPS[a]
            if op in ['pass']:
                pass_list.append(j)
                continue
            if op in ["add", "subtract", "multiply", "divide"]:
                b = random.randint(0, data.shape[1] - 2)

                new_feature = globals()[op](new_fe.iloc[:, j].values, data.iloc[:, b].values).reshape(-1)
                new_name = new_fe.iloc[:, j].name + '_' + op + '_' + data.iloc[:, b].name
                x = (abs(new_fe.iloc[:, j].values - data.iloc[:, b].values)).sum()
                if abs(x) < 0.1 and op in ['subtract', 'divide']:
                    continue
                new_fe[new_fe.iloc[:, j].name] = new_feature
                new_fe = new_fe.rename(columns={new_fe.iloc[:, j].name: new_name})
                count_list[j] += 1
            else:
                new_feature = globals()[op](new_fe.iloc[:, j].values).reshape(-1)
                new_name = new_fe.iloc[:, j].name + '_' + op
                new_fe[new_fe.iloc[:, j].name] = new_feature
                new_fe = new_fe.rename(columns={new_fe.iloc[:, j].name: new_name})
                count_list[j] += 1
    new_fe = new_fe.astype(np.float32).apply(np.nan_to_num)
    # model = RandomForestClassifier(n_estimators=10, random_state=0)
    model = LogisticRegression()
    # my_cv = StratifiedKFold(n_splits=5)
    # my_cv = KFold(n_splits=5)
    X = new_fe.iloc[:, :-1]
    # X = pd.concat([X],axis=1)
    y = new_fe.iloc[:, -1]
    # X = data.iloc[:,:-1]
    # y = data.iloc[:,-1]
    rae_score1 = make_scorer(sub_rae, greater_is_better=True)
    scores = cross_val_score(model, X, y, scoring='f1_micro', cv=5, error_score="raise")
    # scores = cross_val_score(model, X, y, scoring=rae_score1, cv=5)
    score_list.append(np.mean(scores))
print(score_list)
print(max(score_list))
