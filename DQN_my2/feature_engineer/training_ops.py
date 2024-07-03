import logging

import lightgbm
import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, mutual_info_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from DQN_my2.metric_evaluate import rae_score
from DQN_my2.model_evaluate import model_fuctions
from DQN_my2.process_data.pipeline_data import Pipeline
from DQN_my2.fe_parsers import parse_actions
from DQN_my2.process_data.update_data import Update_data


def sample(args, dqn_ops, dqn_otp, pipline_data, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, i, workers, device,
           steps_done):
    states = []
    ff = []
    pipline_ffs = []
    con_or_diss = []
    ori_x_c_ds = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    df_encode = pd.concat(
        [df_c_encode.drop(df_c_encode.columns[-1], axis=1), df_d_encode.drop(df_d_encode.columns[-1], axis=1),
         df_t_norm], axis=1)
    ori_x_c_d = pd.concat(
        [df_c_encode.drop(df_c_encode.columns[-1], axis=1), df_d_encode.drop(df_d_encode.columns[-1], axis=1)], axis=1)
    init_state = torch.from_numpy(df_encode.values).float().transpose(0, 1).to(device)
    init_con_or_dis = [1] * len(pipline_data['continuous_columns']) + [-1] * len(pipline_data['discrete_columns']) + [0]
    for_next = False
    steps_done += 1
    if i == 0:
        for _ in range(args.episodes):
            states.append(init_state)
            pipline_ffs.append(Pipeline(pipline_data))
            con_or_diss.append(init_con_or_dis)
            ori_x_c_ds.append(ori_x_c_d)

    else:
        for j in range(args.episodes):
            states.append(workers[j].c_d)
            pipline_ffs.append(Update_data(workers[j].features_c, workers[j].features_d, pipline_data))
            con_or_diss.append(workers[j].con_or_dis)
            ori_x_c_ds.append(workers[j].x_c_d)

    for i in range(args.episodes):
        x_c = pd.DataFrame()
        x_d = pd.DataFrame()
        actions_ops, states_ops = dqn_ops.choose_action_ops(states[i], for_next, steps_done, con_or_diss[i])
        actions_otp, states_otp = dqn_otp.choose_action_otp(actions_ops, states_ops, for_next, steps_done)
        fe = parse_actions(actions_ops, actions_otp, c_ops, d_ops, df_c_encode, df_d_encode, n_c_features, n_d_features)
        df_c_encode, df_d_encode, specials = pipline_ffs[i].process_data(fe)
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
        workers[i].states_otp = states_otp
        workers[i].actions_ops = actions_ops
        workers[i].actions_otp = actions_otp
        workers[i].features_c = x_c
        workers[i].features_d = x_d
        workers[i].x_c_d = x_c_d
        workers[i].ff.append(fe)
        workers[i].con_or_dis = con_or_dis
        workers[i].ori_x_c_d = ori_x_c_ds[i]
        workers[i].specials = specials
    return workers


def sample_update(args, pipline_data, dqn_ops, dqn_otp, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, workers,
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
        actions_otp, states_otp = dqn_otp.choose_action_otp(actions_ops, states_ops, for_next, steps_done)
        fe = parse_actions(actions_ops, actions_otp, c_ops, d_ops, df_c_encode, df_d_encode, n_c_features, n_d_features)
        df_c_encode, df_d_encode, special = pipline_ff.process_data(fe)
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
        workers[i].states_ops_ = states_ops
        workers[i].actions_ops_ = actions_ops
        workers[i].actions_otp_ = actions_otp
        workers[i].features_c_ = x_c
        workers[i].features_d_ = x_d
        workers[i].ff_ = fe
        workers[i].c_d_ = x_encode_c
    return workers


def multiprocess_reward(args, pipline_data,workers, scores_b, mode, model, metric, y):
    for i in range(args.episodes):
        x_c = workers[i].features_c
        x_d = workers[i].features_d
        if x_c.shape[0] != 0:
            x_c, _ = remove_duplication(x_c)
        if x_d.shape[0] != 0:
            x_d, _ = remove_duplication(x_d)
        if x_c.shape[0] == 0:
            x = np.array(x_d)
            x_c_d = x_d
        elif x_d.shape[0] == 0:
            x = np.array(x_c)
            x_c_d = x_c
        else:
            x = np.concatenate((x_c, x_d), axis=1)
            x_c_d = pd.concat([x_c, x_d], axis=1)
        # x = np.concatenate((x, pipline_data['ori_continuous_data']), axis=1)
        y = np.array(y)
        a = args.a
        b = args.b
        c = args.c
        d = args.d
        e = args.e
        f = args.f
        g = args.g
        # scaler = MinMaxScaler()
        # x = scaler.fit_transform(x)
        # logging.info(f"x.shape{x.shape} ")
        score = get_reward(x, y, args, mode, model, metric)

        ### Rv
        Rv = 0
        # mutual_info_list = mutual_info_regression(x_c_d, y).tolist()
        # total_mutual = 0.0
        # for mutual in mutual_info_list:
        #     total_mutual += mutual
        # Rv = total_mutual / x_c_d.shape[1]


        # for j in range(x_c_d.shape[1]):
        #     x = mutual_info_score(np.array(x_c_d.iloc[:,j]), np.array(y))
        #     Rv += x
        # Rv = np.sum(Rv) / x_c_d.shape[1]

        #### Rd
        Rd = 0
        x_c_d_features = x_c_d.columns
        totle_redundancy = 0
        #计算每对特征之间的互信息
        # correlation_matrix = x_c_d.corr()
        # correlation_matrix = np.array(correlation_matrix)
        # for num,col in enumerate(correlation_matrix):
        #     for num2,row in enumerate(col):
        #         if num2 != num:
        #             totle_redundancy += row
        # for col in range(x_c_d.shape[1]):
        #     mutual_info_list = mutual_info_regression(x_c_d, x_c_d.iloc[:,col]).tolist()
        #     for mutual in mutual_info_list:
        #         totle_redundancy += mutual
        # Rd = totle_redundancy / x_c_d.shape[1] / x_c_d.shape[1]


        # for feature1 in x_c_d_features:
        #     for feature2 in x_c_d_features:
        #         if feature1 != feature2:
        #             # 计算特征之间的互信息
        #             mi_score = mutual_info_score(x_c_d.loc[:,feature2], x_c_d.loc[:,feature1])
        #             mi_between_features += mi_score
        # Rd = mi_between_features / (x_c_d.shape[1] * x_c_d.shape[1])

        Rre = 0
        Rde = 0
        Rco = 0
        num_Rre = 0
        num_Rde = 0
        num_Rco = 0

        for k, v in workers[i].specials.items():
            if k == 'replace':
                for double in v:
                    double[1] = np.where(double[1] > 1e15, 0, double[1])
                    q = mutual_info_score(double[1], y)
                    w = mutual_info_score(double[0], y)
                    # q = mutual_info_classif(double[1].reshape(-1,1), y.reshape(-1,1))
                    # w = mutual_info_classif(double[0].reshape(-1,1), y.reshape(-1,1))
                    x = q - w
                    x = float(x)
                    Rre += x
                    num_Rre += 1
                if Rre != 0:
                    Rre /= num_Rre
            elif k == 'delete':
                for double in v:
                    double[1] = np.where(double[1] > 1e15, 0, double[1])
                    q = mutual_info_score(double[0], y)
                    w = mutual_info_score(double[1], y)
                    # q = mutual_info_classif(double[1].reshape(-1, 1), y.reshape(-1, 1))
                    # w = mutual_info_classif(double[0].reshape(-1, 1), y.reshape(-1, 1))
                    x = q - w
                    x = float(x)
                    Rde += x
                    num_Rde += 1
                if num_Rde != 0:
                    Rde /= num_Rde
            else:
                for double in v:
                    double[1] = np.where(double[1] > 1e15, 0, double[1])
                    double[1] = np.where(double[1] < -1e15, 0, double[1])
                    x = mutual_info_score(double[0], double[1])
                    # x = 0
                    x = float(x)
                    Rco += x
                    num_Rco += 1
                if num_Rco != 0:
                    Rco /= num_Rco

        workers[i].scores = score
        workers[i].scores_b = score
        # if np.mean(score) > workers[i].best_score:
        #     workers[i].best_score = np.mean(score)
        # w = (np.mean(score) - workers[i].best_score)

        # w = np.mean(score)
        w = (np.mean(score) - np.mean(scores_b))
        workers[i].reward_1 = w
        # a * w + b * Rv - c * Rd
        workers[i].reward_2 = d * w + e * Rre + f * Rde - g * Rco
        # d * w + e * Rre + f * Rde - g * Rco
        workers[i].x_c_d = x_c_d
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
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            # clf = KNeighborsClassifier(n_neighbors=3)
            scores = cross_val_score(clf, x, y, scoring='f1_micro', cv=5)
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
            model1 = RandomForestRegressor(n_estimators=10, random_state=0)
            rae_score1 = make_scorer(sub_rae, greater_is_better=True)
            scores = cross_val_score(model1, x, y, cv=5, scoring=rae_score1)
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


# def sub_rae(y, y_hat):
#     y = np.array(y).reshape(-1)
#     y_hat = np.array(y_hat).reshape(-1)
#     y_mean = np.mean(y)
#     rae = np.sum([np.abs(y_hat[i] - y[i]) for i in range(len(y))]) / np.sum(
#         [np.abs(y_mean - y[i]) for i in range(len(y))])
#     res = 1 - rae
#     return res


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