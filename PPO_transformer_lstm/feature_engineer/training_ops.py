import numpy as np
import logging

import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from .worker import Worker
from PPO_transformer_lstm.fe_parsers import parse_actions
from PPO_transformer_lstm.metric_evaluate import rae_score
from PPO_transformer_lstm.model_evaluate import model_fuctions
from PPO_transformer_lstm.process_data import Pipeline
from PPO_transformer_lstm.process_data.utils_memory import remove_duplication

def sample(args, ppo, pipline_args_train, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, epoch, i, device):
    logging.debug(f'Get in pipline_ff_c for episode {i}...')
    pipline_ff = Pipeline(pipline_args_train)
    logging.debug(f'End pipline_ff_c')
    worker = Worker(args)
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
    init_state = torch.from_numpy(df_encode.values).float().transpose(0, 1).to(device)
    steps_num = args.steps_num
    logging.debug(f'Start sample episode {i}...')
    for step in range(steps_num):
        steps.append(step)
        if step == 0:
            res_h_c_t_list = None
        state = init_state
        actions_ops, log_ops_probs, actions_otp, log_otp_probs,res_h_c_t_list = ppo.choose_action(state, step, epoch,c_ops,res_h_c_t_list)
        fe = parse_actions(actions_ops, actions_otp, c_ops,d_ops,df_c_encode,df_d_encode,n_c_features,n_d_features)
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
        elif x_c.shape[0] == 0:
            x_encode_c = np.hstack((x_d, df_t_norm.values.reshape(-1, 1)))
        else:
            x_encode_c = np.hstack((x_c, x_d, df_t_norm.values.reshape(-1, 1)))
        x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
        init_state = x_encode_c
        states.append(state.cpu())
        actionss_ops.append(actions_ops)
        actionss_otp.append(actions_otp)
        log_ops_probss.append(log_ops_probs)
        log_otp_probss.append(log_otp_probs)

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
    return worker


def multiprocess_reward(args, worker, c_columns, d_columns, scores_b, mode, model, metric, x_d_onehot, y,df_d_labelencode):
    accs = []
    cvs = []
    scores = []
    new_fe_nums = []
    repeat_fe_nums = []
    # repeat_ratio = cal_repeat_actions(len(c_columns), worker_c.ff)
    # repeat_fe_nums.append(repeat_ratio)
    for step in range(args.steps_num):
        x_c = worker.features_c[step]
        x_d = worker.features_d[step]
        if x_c.shape[0] == 0:
            x = x_d
        elif x_d.shape[0] == 0:
            x = x_c
        else:
            x_c, _ = remove_duplication(x_c)
            x_d, _ = remove_duplication(x_d)
            x = np.concatenate((x_c, x_d), axis=1)

        acc, cv, score = get_reward(x, y, args, scores_b, mode, model, metric, step)
        accs.append(acc)
        cvs.append(cv)
        scores.append(score)
    worker.fe_nums = new_fe_nums
    worker.accs = accs
    worker.cvs = cvs
    worker.scores = scores
    worker.features_c = x_c
    worker.features_d = x_d
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