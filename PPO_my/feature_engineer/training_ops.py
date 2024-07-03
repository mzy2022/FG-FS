import numpy as np
import logging

import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler

from .worker import Worker
from PPO_my.fe_parsers import parse_actions
from PPO_my.metric_evaluate import rae_score
from PPO_my.model_evaluate import model_fuctions
from PPO_my.process_data import Pipeline
from PPO_my.process_data.utils_memory import remove_duplication
from PPO_my.process_data.update_data import Update_data
def sample(args, ppo, pipline_args_train, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, epoch, i, device):
    logging.debug(f'Get in pipline_ff_c for episode {i}...')

    logging.debug(f'End pipline_ff_c')
    worker = Worker(args)
    emb_states = []
    states = []
    actionss_ops = []
    actionss_otp = []
    log_ops_probss = []
    log_otp_probss = []
    features_c = []
    features_d = []
    ff = []
    steps = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    df_encode = pd.concat([df_c_encode.drop(df_c_encode.columns[-1], axis=1) ,df_d_encode.drop(df_d_encode.columns[-1], axis=1),df_t_norm],axis=1)
    df_c_encode = df_c_encode.drop(df_c_encode.columns[-1], axis=1)
    df_d_encode = df_d_encode.drop(df_d_encode.columns[-1], axis=1)
    init_state = torch.from_numpy(df_encode.values).float().transpose(0, 1).to(device)
    steps_num = args.steps_num
    con_or_diss = []
    logging.debug(f'Start sample episode {i}...')
    init_con_or_dis = [1] * len(pipline_args_train['continuous_columns']) + [-1] * len(pipline_args_train['discrete_columns']) + [0]
    con_or_diss.append(init_con_or_dis)
    for step in range(steps_num):
        if step != 0:
            pipline_ff = Update_data(x_c,x_d,pipline_args_train)
        else:
            pipline_ff = Update_data(df_c_encode, df_d_encode, pipline_args_train)
        steps.append(step)
        state = init_state
        actions_ops, log_ops_probs, emb_state = ppo.choose_action_ops(state, step, epoch,c_ops,init_con_or_dis)
        actions_otp, log_otp_probs = ppo.choose_action_otp(actions_ops, emb_state)
        fe = parse_actions(actions_ops,actions_otp, c_ops,d_ops,df_c_encode,df_d_encode,n_c_features,n_d_features)
        ff.append(fe)
        df_c_encode, df_d_encode = pipline_ff.process_data(fe)
        df_c_encode = pd.concat([df_c_encode,df_t_norm],axis=1)
        df_d_encode = pd.concat([df_d_encode, df_t_norm], axis=1)

        x_c = df_c_encode.iloc[:,:-1].astype(np.float32).apply(np.nan_to_num)
        x_d = df_d_encode.iloc[:,:-1].astype(np.float32).apply(np.nan_to_num)
        features_d.append(x_d)
        features_c.append(x_c)
        if x_d.shape[0] == 0:
            x_encode_c = np.hstack((x_c,df_t_norm.values.reshape(-1, 1)))
            x_c_d = x_c
        elif x_c.shape[0] == 0:
            x_encode_c = np.hstack((x_d, df_t_norm.values.reshape(-1, 1)))
            x_c_d = x_d
        else:
            x_encode_c = np.hstack((x_c,x_d, df_t_norm.values.reshape(-1, 1)))
            x_c_d = pd.concat([x_c, x_d], axis=1)
        x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
        init_con_or_dis = get_con_or_dis(x_c_d, pipline_args_train)
        con_or_diss.append(init_con_or_dis)
        init_state = x_encode_c
        states.append(state.cpu())
        actionss_ops.append(actions_ops)
        log_ops_probss.append(log_ops_probs)
        actionss_otp.append(actions_otp)
        log_otp_probss.append(log_otp_probs)
        emb_states.append(emb_state.detach().cpu())

    dones = [False for i in range(steps_num)]
    dones[-1] = True

    worker.steps = steps
    worker.states = states
    worker.actions_ops = actionss_ops
    worker.actions_otp = actionss_otp
    worker.log_ops_probs = log_ops_probss
    worker.log_otp_probs = log_otp_probss
    worker.dones = dones
    worker.features_c = features_c
    worker.features_d = features_d
    worker.ff = ff
    worker.emb_states = emb_states
    worker.con_or_diss = con_or_diss
    return worker


def multiprocess_reward(args, worker, c_columns, d_columns, scores_b, mode, model, metric, x_d_onehot, y,df_d_labelencode):
    accs = []
    cvs = []
    scores = []
    new_fe_nums = []
    x_cs = []
    x_ds = []
    for step in range(args.steps_num):
        x_c = worker.features_c[step]
        x_d = worker.features_d[step]
        if x_c.shape[0] == 0:
            x = x_d
        elif x_d.shape[0] == 0:
            x = x_c
        else:
            # x_c, _ = remove_duplication(x_c)
            # x_d, _ = remove_duplication(x_d)
            x = np.concatenate((x_c, x_d), axis=1)
        x_cs.append(x_c)
        x_ds.append(x_d)
        # scaler = MinMaxScaler()
        # x = scaler.fit_transform(x)
        # logging.info(f"x.shape{x.shape} ")
        acc, cv, score = get_reward(x, y, args, scores_b, mode, model, metric, step)
        accs.append(acc)
        cvs.append(cv)
        scores.append(score)
    worker.fe_nums = new_fe_nums
    worker.accs = accs
    worker.cvs = cvs
    worker.scores = scores
    # worker.repeat_fe_nums = repeat_fe_nums
    return worker


def get_reward(x, y, args, scores_b, mode, model, metric, step, repeat_ratio=0):
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

    values = np.array(scores) - np.array(scores_b)
    mask = values < 0
    negative = values[mask]
    negative_sum = negative.sum()
    reward = np.array(scores).mean() + negative_sum
    if step == (args.steps_num - 1):
        reward = reward - repeat_ratio
    return round(reward, 4), values, scores

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