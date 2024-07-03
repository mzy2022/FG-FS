import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from Own.Evolutionary_FE.DNA_Fitness import relative_absolute_error

data = pd.read_csv("test.csv")
y = data.iloc[:,0]
X = data.iloc[:,1:]

new_pd = pd.concat([X,y],axis=1)
new_pd.to_csv('lymphography.csv',index=False)
label_encoder = LabelEncoder()

# 使用LabelEncoder进行标签编码
encoded_labels = label_encoder.fit_transform(y)
X['label'] = encoded_labels
data1= X.copy()
data1.to_csv("133.csv", index=False)



def downstream_task_new(f_data, target, task_type):
    X = f_data.iloc[:,1:-1]
    y = target
    # selector = SelectKBest(score_func=mutual_info_classif, k=k)
    # X = selector.fit_transform(X, y)
    # X = pd.DataFrame(X)
    if task_type == 'cls':
        clf = RandomForestClassifier(random_state=0)
        # clf = xgb.XGBClassifier(random_state=0)
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





data = pd.read_csv('test.csv')
data2 = pd.read_csv('train.csv')


list = []
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

for k in range(X.shape[1],X.shape[1]+1):
    m = downstream_task_new(X,y,'cls')
    list.append(m)
    print(k,m)

print(max(list),list.index(max(list)))


# print(data1.shape)
# X = data2.iloc[:,:-1]
# y = data2.iloc[:,-1]
# m = downstream_task_new(X,y,'cls')
# print(m)

