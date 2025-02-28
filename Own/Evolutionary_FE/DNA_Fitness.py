import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from Own.model_based import ModelBase

def get_rf_model(hyper_param):
    model = ModelBase.rf_classify(0)

    if hyper_param is not None and model is not None:
        model.set_params(**hyper_param)
    return model
def fitness_score(X,y, task_type):
    # score_list = cross_val_score(get_rf_model(None), X, y ,scoring="f1_micro", cv=5)
    # return np.mean(score_list)
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

def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(y_test) - y_test))
    return error

def rae_score(model,x_test,y_test):
    # reg = RandomForestRegressor(random_state=0)
    y_predict = model.predict(x_test)
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(y_test) - y_test))
    return error
