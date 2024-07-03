from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.special import expit
from sklearn import linear_model
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.svm import LinearSVC


def cube(x):
    return x ** 3


# 转换操作集为np.的形式
def justify_operation_type(op):
    if op == 'sqrt':
        op = np.sqrt
    elif op == 'square':
        op = np.square
    elif op == 'sin':
        op = np.sin
    elif op == 'cos':
        op = np.cos
    elif op == 'tanh':
        op = np.tanh
    elif op == 'reciprocal':
        op = np.reciprocal
    elif op == '+':
        op = np.add
    elif op == '-':
        op = np.subtract
    elif op == '/':
        op = np.divide
    elif op == '*':
        op = np.multiply
    elif op == 'stand_scaler':
        op = StandardScaler()
    elif op == 'minmax_scaler':
        op = MinMaxScaler(feature_range=(-1, 1))
    elif op == 'quan_trans':
        op = QuantileTransformer(random_state=0)
    elif op == 'exp':
        op = np.exp
    elif op == 'cube':
        op = cube
    elif op == 'sigmoid':
        op = expit
    elif op == 'log':
        op = np.log
    else:
        print('Please check your operation!')
    return op


# 合并两个dataframe
def insert_generated_feature_to_original_feas(feas, f):
    y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
    y_label.columns = [feas.columns[len(feas.columns) - 1]]
    feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data


# 函数的目的是生成特征状态矩阵。它首先将输入 X 转换为浮点型（np.float64），然后使用describe()方法计算统计描述信息。接着对结果使用iloc[i, :]进行索引操作，表示提取第 i 行的数据，然后再次使用describe()计算统计描述信息。
def feature_state_generation(X):
    return _feature_state_generation_des(X)


def _feature_state_generation_des(X):
    feature_matrix = []
    for i in range(8):
        feature_matrix += list(X.astype(np.float64).describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix


# 这段代码定义了一个名为downstream_task_new的函数，它用于执行下游任务（即基于数据进行分类或回归预测）并计算评估指标。
def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(y_test) - y_test))
    return error


def downstream_task_new(data, task_type):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        reg = RandomForestRegressor(random_state=0)
        rae_list = []
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    else:
        return -1


def test_task_new(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(int)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list = [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list = [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list)
    else:
        return -1


# 这是一个名为cluster_features的函数，用于对特征进行聚类分析，并返回聚类结果。
def cluster_features(features, y, cluster_num=2, mode=''):
    if mode == 'c':
        return _wocluster_features(features, y, cluster_num)
    else:
        return _cluster_features(features, y, cluster_num)


# 用于将每个特征列作为一个独立的聚类，并返回聚类结果
def _wocluster_features(features, y, cluster_num=2):
    clusters = defaultdict(list)
    for ind, item in enumerate(range(features.shape[1])):
        clusters[item].append(ind)
    return clusters


# 用于将特征进行聚类，并返回聚类结果
def _cluster_features(features, y, cluster_num=2):
    k = int(np.sqrt(features.shape[1]))
    features = feature_distance(features, y)
    features = features.reshape(features.shape[0], -1)
    clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='single').fit(features)
    labels = clustering.labels_
    clusters = defaultdict(list)
    for ind, item in enumerate(labels):
        clusters[item].append(ind)
    return clusters


def feature_distance(feature, y):
    return mi_feature_distance(feature, y)


def mi_feature_distance(features, y):
    dis_mat = []
    for i in range(features.shape[1]):
        tmp = []
        for j in range(features.shape[1]):
            tmp.append(np.abs(mutual_info_regression(features[:, i].reshape(-1, 1), y) - mutual_info_regression(
                features[:, j].reshape(-1, 1), y))[0] / (
                               mutual_info_regression(features[:, i].reshape(-1, 1), features[:, j].reshape(-1, 1))[
                                   0] + 1e-05))
        dis_mat.append(np.array(tmp))
    dis_mat = np.array(dis_mat)
    return dis_mat
