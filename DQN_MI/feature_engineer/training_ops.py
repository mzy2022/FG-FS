import logging

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler

from DQN_MI.metric_evaluate import rae_score
from DQN_MI.model_evaluate import model_fuctions
from DQN_MI.process_data.pipeline_data import Pipeline
from DQN_MI.fe_parsers import parse_actions
from DQN_MI.process_data.update_data import Update_data


def sample(args, dqn_ops, pipline_data, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, step, workers, device,
           steps_done):
    states = []
    ff = []
    con_or_diss = []
    pipline_ffs = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    df_encode = pd.concat(
        [df_c_encode.drop(df_c_encode.columns[-1], axis=1), df_d_encode.drop(df_d_encode.columns[-1], axis=1),
         df_t_norm], axis=1)
    init_state = torch.from_numpy(df_encode.values).float().transpose(0, 1).to(device)
    init_con_or_dis = [1] * len(pipline_data['continuous_columns']) + [-1] * len(pipline_data['discrete_columns']) + [0]
    for_next = False
    steps_done += 1
    if step == 0:
        for _ in range(args.episodes):
            states.append(init_state)
            pipline_ffs.append(
                Update_data(pipline_data['continuous_data'].drop(pipline_data['continuous_data'].columns[-1], axis=1),
                            pipline_data['discrete_data'].drop(pipline_data['discrete_data'].columns[-1], axis=1),
                            pipline_data))
            con_or_diss.append(init_con_or_dis)
    else:
        for j in range(args.episodes):
            states.append(workers[j].c_d)
            pipline_ffs.append(Update_data(workers[j].past_x_c, workers[j].past_x_d, pipline_data))
            con_or_diss.append(workers[j].con_or_dis)

    for i in range(args.episodes):
        if step == 0:
            past_x_c = pipline_data['continuous_data'].drop(pipline_data['continuous_data'].columns[-1], axis=1)
            past_x_d = pipline_data['discrete_data'].drop(pipline_data['discrete_data'].columns[-1], axis=1)
        else:
            past_x_c = workers[i].past_x_c
            past_x_d = workers[i].past_x_d
        actions_ops, states_ops = dqn_ops.choose_action_ops(states[i], for_next, steps_done,con_or_diss[i])
        fe = parse_actions(actions_ops, c_ops, d_ops, df_c_encode, df_d_encode, n_c_features, n_d_features)
        df_c_encode, df_d_encode = pipline_ffs[i].process_data(fe)
        past_x_c = pd.concat([past_x_c, df_c_encode], axis=1)
        past_x_d = pd.concat([past_x_d, df_d_encode], axis=1)
        # df_c_encode = pd.concat([df_c_encode, df_t_norm], axis=1)
        # df_d_encode = pd.concat([df_d_encode, df_t_norm], axis=1)

        x_c = df_c_encode.astype(np.float32).apply(np.nan_to_num)
        x_d = df_d_encode.astype(np.float32).apply(np.nan_to_num)
        if x_d.shape[0] == 0:
            x_encode_c = np.hstack((past_x_c, df_t_norm.values.reshape(-1, 1)))
            x_c_d = pd.concat([past_x_c],axis=1)
        elif x_c.shape[0] == 0:
            x_encode_c = np.hstack((past_x_d, df_t_norm.values.reshape(-1, 1)))
            x_c_d = pd.concat([past_x_d], axis=1)

        else:
            x_encode_c = np.hstack((past_x_c, past_x_d, df_t_norm.values.reshape(-1, 1)))
            x_c_d = pd.concat([past_x_c, past_x_d], axis=1)
        x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
        con_or_dis = get_con_or_dis(x_c_d, pipline_data)
        workers[i].c_d = x_encode_c
        workers[i].states_ops = states_ops
        workers[i].actions_ops = actions_ops
        workers[i].features_c = x_c
        workers[i].features_d = x_d
        workers[i].ff = fe
        workers[i].past_x_c = past_x_c
        workers[i].past_x_d = past_x_d
        workers[i].con_or_dis = con_or_dis

    return workers


def sample_update(args, pipline_data, dqn_ops, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, workers, steps_done,
                  device):
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
        actions_ops, states_ops = dqn_ops.choose_action_ops(states_[i], for_next, steps_done,workers[i].con_or_dis)
    #     fe = parse_actions(actions_ops, c_ops, d_ops, df_c_encode, df_d_encode, n_c_features, n_d_features)
    #     df_c_encode, df_d_encode = pipline_ff.process_data(fe)
    #     df_c_encode = pd.concat([df_c_encode, df_t_norm], axis=1)
    #     df_d_encode = pd.concat([df_d_encode, df_t_norm], axis=1)
    #     x_c = df_c_encode.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
    #     x_d = df_d_encode.iloc[:, :-1].astype(np.float32).apply(np.nan_to_num)
    #     if x_d.shape[0] == 0:
    #         x_encode_c = np.hstack((x_c, df_t_norm.values.reshape(-1, 1)))
    #     elif x_c.shape[0] == 0:
    #         x_encode_c = np.hstack((x_d, df_t_norm.values.reshape(-1, 1)))
    #     else:
    #         x_encode_c = np.hstack((x_c, x_d, df_t_norm.values.reshape(-1, 1)))
    #     x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
    #
        workers[i].states_ = states_
        workers[i].states_ops_ = states_ops
        workers[i].actions_ops_ = actions_ops
    #     workers[i].features_c_ = x_c
    #     workers[i].features_d_ = x_d
    #     workers[i].ff_ = fe
    #     workers[i].c_d_ = x_encode_c
    return workers


def multiprocess_reward(args, workers, scores_b, mode, model, metric, y):
    for i in range(args.episodes):
        x_c = workers[i].past_x_c
        x_d = workers[i].past_x_d
        # if x_c.shape[1] > 0:
        #     x_c = x_c.drop(x_c.columns[-1], axis=1)
        # if x_d.shape[1] > 0:
        #     x_d = x_d.drop(x_d.columns[-1], axis=1)
        if x_c.shape[1] == 0:
            x = np.array(x_d)
        elif x_d.shape[1] == 0:
            x = np.array(x_c)
        else:
            x = np.concatenate((x_c, x_d), axis=1)
        y = np.array(y)
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
        logging.info(f"x.shape{x.shape} ")
        acc, cv, score = get_reward(x, y, args, scores_b, mode, model, metric)

        workers[i].accs = acc
        workers[i].cvs = cv
        workers[i].scores = score
        workers[i].scores_b = score
        workers[i].reward = np.mean(score) - np.mean(scores_b)
    return workers


def get_reward(x, y, args, scores_b, mode, model, metric, repeat_ratio=0):
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
            # scores = f1(x,y)
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
            scores = cross_val_score(model, x, y, cv=my_cv, scoring=rae_score)
            # scores = rae(x,y)

    values = np.array(scores) - np.array(scores_b)
    mask = values < 0
    negative = values[mask]
    negative_sum = negative.sum()
    reward = np.array(scores).mean() + negative_sum
    return round(reward, 4), values, scores


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