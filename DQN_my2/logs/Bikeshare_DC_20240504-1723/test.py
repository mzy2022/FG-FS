import catboost
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC, SVR

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

def rae(X, y):
    reg = SVR(kernel='linear')
    rae_list = []
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in kf.split(X):
        X_train, y_train, X_test, y_test = X[train, :], y[train], X[test, :], y[test]
        reg.fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        rae_list.append(1 - relative_absolute_error(y_test, y_predict))
    return np.mean(np.array(rae_list))


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(y_test) - y_test))
    return error

data = pd.read_csv('test.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
mode = 'reg'
if mode == 'classify':
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=2)
    svm_classifier = SVC(kernel='linear')
    log_cls = LogisticRegression()
    ridge_cls = RidgeClassifier(alpha=1.0)
    model_xgb = xgboost.XGBClassifier(n_estimators=10, random_state=0)
    model_light = lightgbm.LGBMClassifier(n_estimators=10, random_state=0)
    model_cat = catboost.CatBoostClassifier(n_estimators=10, random_state=0)

    scores_rf = cross_val_score(rf, X, y, scoring='f1_micro', cv=5).mean()
    scores_knn = cross_val_score(knn, X,y, scoring='f1_micro', cv=5).mean()
    scores_svm = cross_val_score(svm_classifier, X, y, scoring='f1_micro', cv=5).mean()
    scores_log = cross_val_score(log_cls,X,y, scoring='f1_micro', cv=5).mean()
    scores_ridge = cross_val_score(ridge_cls, X, y, scoring='f1_micro', cv=5).mean()
    scores_xgb = cross_val_score(model_xgb, X,y, scoring='f1_micro', cv=5).mean()
    scores_light = cross_val_score(model_light, X,y, scoring='f1_micro', cv=5).mean()
    scores_cat = cross_val_score(model_cat, X,y, scoring='f1_micro', cv=5).mean()


else:
    rf = RandomForestRegressor(n_estimators=10, random_state=0)
    knn_reg = KNeighborsRegressor(n_neighbors=3)
    ridge_cls = Ridge(alpha=1.0)
    svm_regressor = SVR(kernel='linear')
    log_reg = LinearRegression()
    model_xgb = xgboost.XGBRegressor(n_estimators=10, random_state=0)
    model_light = lightgbm.LGBMRegressor(n_estimators=10, random_state=0)
    model_cat = catboost.CatBoostRegressor(n_estimators=10, random_state=0)

    rae_score1 = make_scorer(sub_rae, greater_is_better=True)

    scores_rf = cross_val_score(rf, X, y, cv=5, scoring=rae_score1).mean()
    scores_knn = cross_val_score(knn_reg, X,y, cv=5, scoring=rae_score1).mean()
    # scores_svm = cross_val_score(svm_regressor, X,y, cv=5, scoring=rae_score1).mean()
    scores_svm = rae(X,y)
    scores_log = cross_val_score(log_reg, X,y, cv=5, scoring=rae_score1).mean()
    scores_ridge = cross_val_score(ridge_cls, X,y, cv=5, scoring=rae_score1).mean()
    scores_xgb = cross_val_score(model_xgb, X,y, cv=5, scoring=rae_score1).mean()
    scores_light = cross_val_score(model_light, X,y, cv=5, scoring=rae_score1).mean()
    scores_cat = cross_val_score(model_cat, X,y, cv=5, scoring=rae_score1).mean()


print(scores_rf,scores_knn,scores_log,scores_ridge,scores_xgb,scores_light,scores_cat,scores_svm)
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