import logging

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler

from DQN_my3.metric_evaluate import rae_score
from DQN_my3.model_evaluate import model_fuctions
from DQN_my3.process_data.pipeline_data import Pipeline
from DQN_my3.fe_parsers import parse_actions
from DQN_my3.process_data.update_data import Update_data


def sample(args, dqn_ops, dqn_features, dqn_otp, pipline_data, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops,
           c_features, d_features, i, workers, device,
           steps_done):
    states = []
    ff = []
    pipline_ffs = []
    con_or_diss = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    df_encode = pd.concat(
        [df_c_encode.drop(df_c_encode.columns[-1], axis=1), df_d_encode.drop(df_d_encode.columns[-1], axis=1),
         df_t_norm], axis=1)
    init_state = torch.from_numpy(df_encode.values).float().transpose(0, 1).to(device)
    init_con_or_dis = [1] * len(pipline_data['continuous_columns']) + [-1] * len(pipline_data['discrete_columns']) + [0]
    for_next = False
    steps_done += 1
    if i == 0:
        for _ in range(args.episodes):
            states.append(init_state)
            pipline_ffs.append(Pipeline(pipline_data))
            con_or_diss.append(init_con_or_dis)
    else:
        for j in range(args.episodes):
            states.append(workers[j].c_d)
            pipline_ffs.append(Update_data(workers[j].features_c, workers[j].features_d, pipline_data))
            con_or_diss.append(workers[j].con_or_dis)

    for i in range(args.episodes):
        x_c = pd.DataFrame()
        x_d = pd.DataFrame()
        actions_ops, states_ops = dqn_ops.choose_action_ops(states[i], for_next, steps_done, con_or_diss[i])
        new_actions_ops = []
        new_actions_ops = actions_ops
        # for j in actions_ops:
        #     if j >= 4:
        #         new_actions_ops.append(10)
        #     else:
        #         new_actions_ops.append(j)
        # TODO:需要将actions_ops进行掩码？states是否需要？
        actions_features, states_features = dqn_features.choose_action_features(new_actions_ops, states_ops, for_next,
                                                                                steps_done)
        new_actions_features = []
        for num, (action, feature) in enumerate(zip(actions_ops, actions_features)):
            if num < n_c_features:
                feature = feature % n_c_features
                if action >= 4:
                    new_actions_features.append(n_c_features * 4 + action - 4)
                elif action == 0:
                    new_actions_features.append(feature)
                elif action == 1:
                    new_actions_features.append(n_c_features * 1 + feature)
                elif action == 2:
                    new_actions_features.append(n_c_features * 2 + feature)
                elif action == 3:
                    new_actions_features.append(n_c_features * 3 + feature)
            else:
                feature = feature % n_d_features
                action = action % 2
                if action == 0:
                    new_actions_features.append(n_c_features * 4 + 6 + feature)
                else:
                    new_actions_features.append(n_c_features * 4 + 6 + n_d_features + feature)

        actions_otp, states_otp = dqn_otp.choose_action_otp(new_actions_features, states_features, for_next, steps_done)

        fe = parse_actions(actions_ops, actions_features, actions_otp, c_ops, d_ops, df_c_encode, df_d_encode,
                           n_c_features, n_d_features, c_features, d_features)
        df_c_encode, df_d_encode = pipline_ffs[i].process_data(fe)
        if df_c_encode.shape[0] != 0:
            df_c_encode = pd.concat([df_c_encode, df_t_norm], axis=1)
            x_c = df_c_encode.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
        if df_d_encode.shape[0] != 0:
            df_d_encode = pd.concat([df_d_encode, df_t_norm], axis=1)
            x_d = df_d_encode.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)

        if x_c.shape[0] == 0:
            x_encode_c = np.hstack((x_d, df_t_norm.values.reshape(-1, 1)))
            x_c_d = x_d
        elif x_d.shape[0] == 0:
            x_encode_c = np.hstack((x_c, df_t_norm.values.reshape(-1, 1)))
            x_c_d = x_c
        else:
            x_encode_c = np.hstack((x_c, x_d, df_t_norm.values.reshape(-1, 1)))
            x_c_d = pd.concat([x_c, x_d], axis=1)
        x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
        con_or_dis = get_con_or_dis(x_c_d, pipline_data)
        workers[i].c_d = x_encode_c
        workers[i].states_ops = states_ops
        workers[i].states_features = states_features
        workers[i].states_otp = states_otp
        workers[i].actions_ops = actions_ops
        workers[i].actions_features = actions_features
        workers[i].actions_otp = actions_otp
        workers[i].features_c = x_c
        workers[i].features_d = x_d
        workers[i].x_c_d = x_c_d
        workers[i].ff.append(fe)
        workers[i].con_or_dis = con_or_dis

    return workers


def sample_update(args, pipline_data, dqn_ops, dqn_features, dqn_otp, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops,
                  c_features, d_features, workers,
                  steps_done, device):
    states_ = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    for i in range(args.episodes):
        states_.append(workers[i].c_d)
    for_next = True
    steps_done += 1
    for i in range(args.episodes):
        features_c = workers[i].features_c
        features_d = workers[i].features_d
        pipline_ff = Update_data(features_c, features_d, pipline_data)
        actions_ops, states_ops = dqn_ops.choose_action_ops(states_[i], for_next, steps_done, workers[i].con_or_dis)
        actions_features, states_features = dqn_features.choose_action_features(actions_ops, states_ops, for_next,
                                                                                steps_done)
        actions_otp, states_otp = dqn_otp.choose_action_otp(actions_features, states_features, for_next, steps_done)

        fe = parse_actions(actions_ops, actions_features, actions_otp, c_ops, d_ops, df_c_encode, df_d_encode,
                           n_c_features, n_d_features, c_features, d_features)
        df_c_encode, df_d_encode = pipline_ff.process_data(fe)
        df_c_encode = pd.concat([df_c_encode, df_t_norm], axis=1)
        df_d_encode = pd.concat([df_d_encode, df_t_norm], axis=1)
        x_c = df_c_encode.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
        x_d = df_d_encode.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
        if x_d.shape[0] == 0:
            x_encode_c = np.hstack((x_c, df_t_norm.values.reshape(-1, 1)))
        elif x_c.shape[0] == 0:
            x_encode_c = np.hstack((x_d, df_t_norm.values.reshape(-1, 1)))
        else:
            x_encode_c = np.hstack((x_c, x_d, df_t_norm.values.reshape(-1, 1)))
        x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)

        workers[i].states_ = states_
        workers[i].states_otp_ = states_otp
        workers[i].states_features_ = states_features
        workers[i].states_ops_ = states_ops
        workers[i].actions_ops_ = actions_ops
        workers[i].actions_features_ = actions_features
        workers[i].actions_otp_ = actions_otp
        workers[i].features_c_ = x_c
        workers[i].features_d_ = x_d
        workers[i].ff_ = fe
        workers[i].c_d_ = x_encode_c
    return workers


def multiprocess_reward(args, workers, scores_b, mode, model, metric, y):
    for i in range(args.episodes):
        x_c = workers[i].features_c
        x_d = workers[i].features_d
        if x_c.shape[0] != 0:
            x_c, _ = remove_duplication(x_c)
        if x_d.shape[0] != 0:
            x_d, _ = remove_duplication(x_d)
        if x_c.shape[0] == 0:
            x = np.array(x_d)
        elif x_d.shape[0] == 0:
            x = np.array(x_c)
        else:
            x = np.concatenate((x_c, x_d), axis=1)
        y = np.array(y)
        # scaler = MinMaxScaler()
        # x = scaler.fit_transform(x)
        # logging.info(f"x.shape{x.shape} ")
        score = get_reward(x, y, args, mode, model, metric)

        workers[i].scores = score
        workers[i].scores_b = score
        workers[i].reward = np.mean(score) - np.mean(scores_b)
    return workers


def get_reward(x, y, args, mode, model, metric):
    model = model_fuctions[f"{model}_{mode}"]()
    if not args.shuffle: args.seed = None
    if args.cv == 1:
        if mode == "classify":
            my_cv = StratifiedShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size,
                                           random_state=args.seed)
        else:
            my_cv = ShuffleSplit(n_splits=args.cv, train_size=args.cv_train_size, random_state=args.seed)
    else:
        if mode == "classify":
            my_cv = StratifiedKFold(n_splits=args.cv, shuffle=args.shuffle, random_state=args.seed)
        else:
            my_cv = KFold(n_splits=args.cv, shuffle=args.shuffle, random_state=args.seed)

    if mode == "classify":
        if metric == 'f1':
            scores = cross_val_score(model, x, y, scoring='f1_micro', cv=my_cv, error_score="raise")
            # scores = f1(x, y)
        elif metric == 'auc':
            auc_scorer = make_scorer(roc_auc_score, needs_proba=True, average="macro", multi_class="ovo")
            scores = cross_val_score(model, x, y, scoring=auc_scorer, cv=my_cv, error_score="raise")
    else:
        if metric == 'mae':
            scores = cross_val_score(model, x, y, cv=my_cv, scoring='neg_mean_absolute_error')
        elif metric == 'mse':
            scores = cross_val_score(model, x, y, cv=my_cv, scoring='neg_mean_squared_error')
        elif metric == 'r2':
            scores = cross_val_score(model, x, y, cv=my_cv, scoring='r2')
        elif metric == 'rae':
            rae_score1 = make_scorer(sub_rae, greater_is_better=True)
            scores = cross_val_score(model, x, y, cv=my_cv, scoring=rae_score1)
            # scores = rae(x,y)

    return scores


def f1(X, y):
    X = pd.DataFrame(X)
    y = pd.Series(y)
    clf = RandomForestClassifier(random_state=0)
    f1_list = []
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        f1_list.append(f1_score(y_test, y_predict, average='weighted'))
    return np.array(f1_list)


def rae(X, y):
    X = pd.DataFrame(X)
    y = pd.Series(y)
    reg = RandomForestRegressor(random_state=0)
    rae_list = []
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in kf.split(X):
        X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
        reg.fit(X_train, y_train)
        y_predict = reg.predict(X_test)
        rae_list.append(1 - relative_absolute_error(y_test, y_predict))
    return np.array(rae_list)


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(y_test) - y_test))
    return error


def remove_duplication(data):
    _, idx = np.unique(data, axis=1, return_index=True)
    y = data.iloc[:, np.sort(idx)]
    return y, np.sort(idx)


def get_con_or_dis(data, pipline_data):
    con_or_dis = []
    features = data.columns.values
    for feature_name in features:
        flag = 1
        for con_name in pipline_data["discrete_columns"]:
            if con_name in feature_name:
                flag = -1
                break
        con_or_dis.append(flag)
    con_or_dis.append(0)
    return con_or_dis


def sub_rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    rae = np.sum([np.abs(y_hat[i] - y[i]) for i in range(len(y))]) / np.sum(
        [np.abs(y_mean - y[i]) for i in range(len(y))])
    res = 1 - rae
    return res
