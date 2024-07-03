import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

path = os.path.dirname(os.path.realpath(__file__))
file_name = 'CHD_49'
dataset_path = fr"{path}/data/{file_name}.csv"

data = pd.read_csv(dataset_path)
feature_nums = 49


X = data.iloc[:, :feature_nums]
y = data.iloc[:, feature_nums:]


def split_train_test(X, y, train_size, val_size, seed):
    rng = np.random.default_rng(seed)
    inds = np.arange(len(X))
    rng.shuffle(inds)
    n_train = int(train_size * len(X))
    n_val = int(val_size * len(X))
    train_inds = inds[:n_train]
    val_inds = inds[n_train:(n_train + n_val)]
    test_inds = inds[(n_train + n_val):]
    train_data_x = X.iloc[train_inds, :]
    train_data_y = y.iloc[train_inds, :]
    val_data_x = X.iloc[val_inds, :]
    val_data_y = y.iloc[val_inds, :]
    test_data_x = X.iloc[test_inds, :]
    test_data_y = y.iloc[test_inds, :]
    return train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y


train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y = split_train_test(X, y,
                                                                                                0.6,
                                                                                                0.2,
                                                                                                0)
best = 0
f_list = []
for j in range(1000):
    action_list = np.random.randint(2, size=feature_nums)
    i = 0
    while sum(action_list) < 2:
        np.random.seed(i)
        action_list = np.random.randint(2, size=feature_nums)
        i += 1
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    rf_classifier.fit(train_data_x.iloc[:, action_list == 1], train_data_y)
    y_pred = rf_classifier.predict(val_data_x.iloc[:, action_list == 1])
    accuracy = accuracy_score(val_data_y, y_pred)
    print("分类准确性：", accuracy)
    if accuracy >= best:
        best = accuracy
        f_list = action_list
        print(f"{best}()()")



model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
accuracy = accuracy_score(y_pred, test_data_y)
print(f"最终的RF{accuracy}")

base_model = SVC(kernel='linear')
model1 = MultiOutputClassifier(base_model, n_jobs=-1)
model1.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
y_pred = model1.predict(test_data_x.iloc[:, f_list == 1])
accuracy = accuracy_score(y_pred, test_data_y)
print(f"最终的SVC{accuracy}")


base_model = XGBClassifier(eval_metric='logloss')
model = MultiOutputClassifier(base_model)
model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
accuracy = accuracy_score(y_pred, test_data_y)
print(f"XGB{accuracy}")


base_model = DecisionTreeClassifier(random_state=42)
model = MultiOutputClassifier(base_model)
model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
accuracy = accuracy_score(y_pred, test_data_y)
print(f"DT{accuracy}")


base_model = lgb.LGBMClassifier()
model = MultiOutputClassifier(base_model)
model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
accuracy = accuracy_score(y_pred, test_data_y)
print(f"LGB{accuracy}")

print(f_list)