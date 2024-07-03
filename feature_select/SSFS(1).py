# coding=UTF-8
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from construct_W import construct_W
from numpy import linalg as LA
from numpy.random import seed
from sklearn.metrics import accuracy_score
import lightgbm as lgb

eps = 2.2204e-16


def SSFS(X, Y, select_nub, alpha, beta, gamma):
    num, dim = X.shape
    num, label_num = Y.shape

    seed(4)

    options = {'metric': 'euclidean', 'neighbor_mode': 'knn', 'k': 5, 'weight_mode': 'heat_kernel', 't': 1.0}
    Sx = construct_W(X, **options)
    Sx = Sx.A
    Ax = np.diag(np.sum(Sx, 0))
    Lx = Ax - Sx

    k = 10

    V = np.random.rand(num, k)
    B = np.random.rand(dim, k)
    W = np.random.rand(k, label_num)

    iter = 0
    obj = []
    obji = 1

    while 1:
        Btmp = np.sqrt(np.sum(np.multiply(B, B), 1) + eps)
        d1 = 0.5 / Btmp
        D = np.diag(d1.flat)

        V = np.multiply(V, np.true_divide(np.dot(X, B) + alpha * np.dot(Y, W.T) + beta * np.dot(Sx, V),
                                          np.dot(np.dot(V, B.T), B) + alpha * np.dot(np.dot(V, W), W.T) + beta * np.dot(
                                              Ax, V) + eps))

        W = np.multiply(W, np.true_divide(np.dot(V.T, Y), np.dot(np.dot(V.T, V), W) + eps))

        B = np.multiply(B, np.true_divide(np.dot(X.T, V), np.dot(np.dot(B, V.T), V) + gamma * np.dot(D, B) + eps))

        objectives = pow(LA.norm(X - np.dot(V, B.T), 'fro'), 2) + alpha * pow(LA.norm(Y - np.dot(V, W), 'fro'), 2) \
                     + beta * np.trace(np.dot(np.dot(V.T, Lx), V)) + 2 * gamma * np.trace(np.dot(np.dot(B.T, D), B))

        obj.append(objectives)
        cver = abs((objectives - obji) / float(obji))
        obji = objectives
        iter = iter + 1
        if (iter > 2 and (cver < 1e-3 or iter == 300)):
            break

    obj_value = np.array(obj)
    obj_function_value = []
    for i in range(iter):
        temp_value = float(obj_value[i])
        obj_function_value.append(temp_value)
    score = np.sum(np.multiply(B, B), 1)
    idx = np.argsort(-score, axis=0)
    idx = idx.T.tolist()
    l = [i for i in idx]
    n = 1
    F = [l[i:i + n] for i in range(0, len(l), n)]
    F = np.matrix(F)

    ll = [i for i in obj_function_value]
    n = 1
    F_value = [ll[i:i + n] for i in range(0, len(ll), n)]
    F_value = np.matrix(F_value).T
    F_value = np.array(F_value)

    record = dict()
    record['idx'] = F[0:select_nub, :]
    record['obj_value'] = F_value[:, :]
    return record, iter


if __name__ == "__main__":
    # X = np.array([[1, 0, 0, 1, 0, 1, 2, 3],
    #               [1, 0, 2, 0, 1, 1, 1, 2],
    #               [2, 1, 1, 0, 0, 2, 2, 1],
    #               [1, 0, 0, 1, 1, 1, 2, 1],
    #               [3, 2, 1, 1, 1, 1, 3, 3],
    #               [1, 1, 1, 1, 1, 2, 2, 3],
    #               [1, 0, 0, 1, 1, 2, 2, 2],
    #               [1, 1, 0, 1, 0, 2, 2, 0],
    #               [1, 1, 0, 0, 0, 0, 2, 2],
    #               [0, 1, 0, 1, 1, 2, 2, 2]])
    #
    # Y = np.array([[1, 0, 0, 1, 1],
    #               [1, 1, 0, 1, 0],
    #               [0, 0, 1, 1, 0],
    #               [0, 1, 0, 0, 1],
    #               [0, 1, 0, 1, 0],
    #               [1, 0, 0, 0, 0],
    #               [0, 1, 1, 0, 0],
    #               [0, 0, 1, 0, 1],
    #               [0, 1, 0, 0, 1],
    #               [1, 1, 0, 1, 0]])
    data = pd.read_csv('data/Birds.csv')
    data = data.apply(np.nan_to_num)
    X = data.iloc[:, :260]
    Y = data.iloc[:, 260:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    for i in range(126,127):
        aa, bb = SSFS(X_train.values, Y_train.values, select_nub=i, alpha=0.1, beta=0.1, gamma=0.3)
        list_x = aa['idx'].tolist()
        new_list = []
        for i in range(len(list_x)):
            x = list_x[i][0]
            new_list.append(x)

        new_list = sorted(new_list)
        print(new_list)
        # new_list = list(range(X_train.shape[1]))
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(X_train.iloc[:, new_list], Y_train)
        accuracy = model.score(X_test.iloc[:, new_list], Y_test)
        print(accuracy)

        # base_model = SVC(kernel='linear')
        # model = MultiOutputClassifier(base_model, n_jobs=-1)
        # model.fit(X_train.iloc[:, new_list], Y_train)
        # y_pred = model.predict(X_test.iloc[:, new_list])
        # accuracy = accuracy_score(y_pred, Y_test)
        # print(f"SVC{accuracy}")
        #
        # base_model = XGBClassifier(eval_metric='logloss')
        # model = MultiOutputClassifier(base_model)
        # model.fit(X_train.iloc[:, new_list], Y_train)
        # y_pred = model.predict(X_test.iloc[:, new_list])
        # accuracy = accuracy_score(y_pred, Y_test)
        # print(f"XGB{accuracy}")
        #
        # base_model = DecisionTreeClassifier(random_state=42)
        # model = MultiOutputClassifier(base_model)
        # model.fit(X_train.iloc[:, new_list], Y_train)
        # y_pred = model.predict(X_test.iloc[:, new_list])
        # accuracy = accuracy_score(y_pred, Y_test)
        # print(f"DT{accuracy}")
        #
        # base_model = lgb.LGBMClassifier()
        # model = MultiOutputClassifier(base_model)
        # model.fit(X_train.iloc[:, new_list], Y_train)
        # y_pred = model.predict(X_test.iloc[:, new_list])
        # accuracy = accuracy_score(y_pred, Y_test)
        # print(f"LGB{accuracy}")
