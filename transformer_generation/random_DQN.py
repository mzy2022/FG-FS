import random

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import warnings

from feature_generate_memory import *
from config_pool import configs
import pandas as pd
import numpy as np
from tqdm import tqdm
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
name = 'spambase'
path = fr"D:/python files/pythonProject3/DQN_my2/data/{name}.csv"
data = pd.read_csv(path)
ori_X = data.iloc[:, :-1]

y = data.iloc[:, -1]


def get_binning_df(df, label,c_columns, d_columns, mode):
    new_df = pd.DataFrame()
    new_c_columns = []
    new_d_columns = []
    if mode == 'classify':
        for col in c_columns:
            new_df[col] = df[col]
            new_c_columns.append(col)
        for col in c_columns:
            ori_fe = np.array(df[col])
            label = np.array(label)
            new_fe = binning_with_tree(ori_fe, label)
            new_name = 'bin_' + col
            new_df[new_name] = new_fe
            new_d_columns.append(new_name)
        for col in d_columns:
            new_df[col] = df[col]
            new_d_columns.append(col)

    else:
        for col in c_columns:
            new_df[col] = df[col]
            new_c_columns.append(col)
        for col in d_columns:
            new_df[col] = df[col]
            new_d_columns.append(col)

    return new_df, new_c_columns, new_d_columns


def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)

    # 计算每个预测值与实际值之间的绝对误差
    absolute_errors = np.abs(y_hat - y)

    # 计算每个实际值与均值之间的绝对误差
    mean_errors = np.abs(y_mean - y)

    # 计算RAE，然后计算其补值
    rae = np.sum(absolute_errors) / np.sum(mean_errors)
    res = 1 - rae

    return res


D_columns = configs[name]['d_columns']
V_columns = configs[name]['c_columns']
mode = configs[name]['mode']
ori_X, new_c_columns, new_d_columns = get_binning_df(ori_X,y,V_columns,D_columns,mode)
n_c_features = len(new_c_columns)
n_d_features = len(new_d_columns)

value_convert = ["abss", 'square', 'inverse', 'log', 'sqrt', 'power3']
random.seed(1)
np.random.seed(1)
epochs = 1000
steps = 6
best = 0
for epoch in tqdm(range(epochs)):
    new_X = ori_X.copy()
    kkkk = 0
    for step in range(steps):

        for num, feature in enumerate(ori_X.columns):
            kkkk += 1
            b = random.randint(0, 2)
            if 0 <= num < n_c_features:
                a = random.randint(0, 4 * n_c_features + 5)
                if 0 <= a < n_c_features:
                    x = a % n_c_features
                    new = globals()['add'](ori_X.iloc[:, x].values, new_X.iloc[:, num].values)

                elif n_c_features <= a < (2 * n_c_features):
                    x = a % n_c_features
                    new = globals()['subtract'](ori_X.iloc[:, x].values, new_X.iloc[:, num].values)
                elif (2 * n_c_features) <= a < (3 * n_c_features):
                    x = a % n_c_features
                    new = globals()['multiply'](ori_X.iloc[:, x].values, new_X.iloc[:, num].values)
                elif (3 * n_c_features) <= a < (4 * n_c_features):
                    x = a % n_c_features
                    new = globals()['divide'](ori_X.iloc[:, x].values, new_X.iloc[:, num].values)
                else:
                    x = a - n_c_features * 4
                    new = globals()[value_convert[x]](new_X.iloc[:, num].values)
                new_name = kkkk
                if b == 0:
                    new_X[new_name] = new.reshape(-1)
                elif b == 1:
                    new_X[new_X.iloc[:, num].name] = new.reshape(-1)


            else:

                a = random.randint(0, 2 * n_d_features - 1)

                if 0 <= a < n_d_features:
                    x = a % n_d_features + n_c_features
                    feasible_values = {}
                    cnt = 0
                    ori_fe1 = ori_X.iloc[:, x]
                    ori_fe2 = new_X.iloc[:, num]
                    for aaa in np.unique(ori_fe1):
                        for bbb in np.unique(ori_fe2):
                            feasible_values[str(int(aaa)) + str(int(bbb))] = cnt
                            cnt += 1

                    new = globals()['generate_combine_fe'](ori_fe1, ori_fe2,feasible_values)
                    new_name = kkkk
                else:
                    x = a % n_d_features + n_c_features
                    new = globals()['get_nunique_feature'](ori_X.iloc[:, x], new_X.iloc[:, num])
                    new_name = kkkk
                if b == 0:
                    new_X[new_name] = new.reshape(-1)
                elif b == 1:
                    new_X[new_X.iloc[:, num].name] = new.reshape(-1)
    new_X.columns = new_X.columns.astype(str)
    new_X[new_X > 1e15] = 0
    new_X = new_X.apply(np.nan_to_num)

    # clf = RandomForestClassifier(n_estimators=10, random_state=0)
    # scores = cross_val_score(clf, new_X, y, scoring='f1_micro', cv=5)

    model = RandomForestRegressor(n_estimators=10, random_state=0)
    rae_score1 = make_scorer(sub_rae, greater_is_better=True)
    scores = cross_val_score(model, new_X, y, cv=5, scoring=rae_score1)
    score = np.mean(scores)
    if best < score:
        best = score
    print(score)

print(f"{best}&&&")
