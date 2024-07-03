import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/forest.csv')


X = data.iloc[:, :54]
y = data.iloc[:, 54:]
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


def get_reward(X_train, Y_train, X_val, Y_val,actioms):

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X_train.iloc[:, actions], Y_train)
    y_pred = model.predict(X_val.iloc[:, actions])
    Y_val = Y_val.values
    accuracy = accuracy_score(y_pred, Y_val)

    return accuracy

best =0
train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y = split_train_test(X,y,0.6,0.2,0)
for i in range(1000):
    actions = random.sample(range(0, 54), 34)
    reward = get_reward(train_data_x,train_data_y,val_data_x,val_data_y,actions)
    if reward > best:
        best = reward
        print(f"{best}))))")