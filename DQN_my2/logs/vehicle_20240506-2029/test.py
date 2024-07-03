import catboost
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
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

data = pd.read_csv('train.csv')
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
mode = 'classify'
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
    # svm_regressor = SVR(kernel='linear')
    log_reg = LinearRegression()
    model_xgb = xgboost.XGBRegressor(n_estimators=10, random_state=0)
    model_light = lightgbm.LGBMRegressor(n_estimators=10, random_state=0)
    model_cat = catboost.CatBoostRegressor(n_estimators=10, random_state=0)

    rae_score1 = make_scorer(sub_rae, greater_is_better=True)

    scores_rf = cross_val_score(rf, X, y, cv=5, scoring=rae_score1).mean()
    scores_knn = cross_val_score(knn_reg, X,y, cv=5, scoring=rae_score1).mean()
    # scores_svm = cross_val_score(svm_regressor, X,y, cv=5, scoring=rae_score1)
    scores_log = cross_val_score(log_reg, X,y, cv=5, scoring=rae_score1).mean()
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