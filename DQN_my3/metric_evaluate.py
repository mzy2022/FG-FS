import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold


def rae(y_true: np.ndarray, y_pred: np.ndarray):
    up = np.abs(y_pred - y_true).sum()
    down = np.abs(y_true.mean() - y_true).sum()
    score = 1 - up / down
    return score


def xgb_f1(pred, xgbtrain):
    label = xgbtrain.get_label()
    pred = 1 / (1 + np.exp(-pred))
    y_pred = (pred >= 0.5).astype(float)
    f1 = f1_score(label, y_pred)
    return 'xgb_f1', -f1


def f1_metric(model, x_test, y_test, y_train):
    y_pred = model.predict(x_test)
    score = f1_score(y_test, y_pred, average="micro")
    return score

# def f1_metric(model,X,y,y1):
#     X = pd.DataFrame(X)
#     y = pd.DataFrame(y)
#     clf = RandomForestClassifier(random_state=0)
#     f1_list = []
#     skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
#     for train, test in skf.split(X, y):
#         X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
#         clf.fit(X_train, y_train)
#         y_predict = clf.predict(X_test)
#         f1_list.append(f1_score(y_test, y_predict, average='weighted'))
#     return f1_list


def auc_metric(model, x_test, y_test, y_train):
    y_pred = model.predict_proba(x_test)
    score = roc_auc_score(y_test, y_pred, average="macro", multi_class="ovo")
    return score


def ks_metric(model, x_test, y_test, y_train):
    y_pred = model.predict_proba(x_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    score = max(tpr - fpr)
    return score


def r2_metric(model, x_test, y_test, y_train):
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    return score


def rae_metric(model, x_test, y_test, y_train):
    y_pred = model.predict(x_test)
    score = rae(y_test, y_pred)
    return score


def rae_score(model, x_test, y_test):
    y_pred = model.predict(x_test)
    # score = rae(y_test, y_pred)
    up = np.abs(y_pred - y_test).sum()
    down = np.abs(y_test.mean() - y_test).sum()
    score = 1 - up / down
    return score

    # y_predict = model.predict(x_test)
    # y_test = np.array(y_test)
    # y_predict = np.array(y_predict)
    # error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(y_test) - y_test))
    # return 1 - error



def mse_metric(model, x_test, y_test, y_train):
    y_pred = model.predict(x_test)
    # 负的MSE，因为排序是从大到小排
    score = -mean_squared_error(y_test, y_pred)
    return score


def mae_metric(model, x_test, y_test, y_train):
    y_pred = model.predict(x_test)
    score = mean_absolute_error(y_test, y_pred)
    return score


metric_fuctions = {"f1": f1_metric, "auc": auc_metric, "ks": ks_metric, "r2": r2_metric, "rae": rae_metric,
                   "mse": mse_metric, "mae": mae_metric}
