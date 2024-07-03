import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, mutual_info_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold


def sample(args, i, x, y, dqn_ops, worker, device, steps_done, ori_x):
    init_x_state = torch.from_numpy(x.values).float().transpose(0, 1).to(device)
    init_y_state = torch.from_numpy(y.values).float().transpose(0, 1).to(device)
    for_next = False
    steps_done += 1
    action, state = dqn_ops.choose_action_ops(init_x_state, init_y_state, for_next, steps_done)
    worker.states_x = init_x_state
    worker.states_y = init_y_state
    worker.states_ops = state
    worker.actions_ops = action
    worker.action_list[i] = action
    columns_to_keep = []
    for num, i in enumerate(worker.action_list):
        if i != 0:
            columns_to_keep.append(num)
    # columns_to_keep = [col for col in worker.action_list if worker.action_list[col] != 0]

    worker.new_x = ori_x.iloc[:, columns_to_keep]

    return worker


def sample_update(args, x, y, dqn_ops, worker, device, steps_done):
    init_x_state = torch.from_numpy(x.values).float().transpose(0, 1).to(device)
    init_y_state = torch.from_numpy(y.values).float().transpose(0, 1).to(device)
    for_next = True
    steps_done += 1
    action, state = dqn_ops.choose_action_ops(init_x_state, init_y_state, for_next, steps_done)
    worker.states_ops_ = state
    return worker


def multiprocess_reward(args, worker, X_train, Y_train, X_val, Y_val):
    action_list = worker.action_list
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    # model = RandomForestRegressor(n_estimators=10, random_state=0)
    if sum(action_list) == 0:
        worker.reward_1 = 0
    else:
        model.fit(X_train.iloc[:, action_list == 1], Y_train)
        y_pred = model.predict(X_val.iloc[:, action_list == 1])
        Y_val = Y_val.values
        accuracy = 0
        accuracy = accuracy_score(y_pred, Y_val)
        # for i in range(y_pred.shape[1]):
        #     accuracy += accuracy_score(y_pred[:, i], Y_val[:, i])
        # accuracy /= y_pred.shape[1]
        worker.reward_1 = accuracy
    return worker
