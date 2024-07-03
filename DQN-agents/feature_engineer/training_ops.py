import numpy as np
import pandas as pd
import torch
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold

from .metric_evaluate import rae_score
from ..model_evaluate import model_fuctions
from ..process_data.pipeline_data import Pipeline
from ..fe_parsers import parse_actions
from .worker import Worker


def sample(args, dqn_ops, dqn_otp,pipline_data, df_c_encode, df_d_encode, df_t_norm, c_ops,d_ops,epoch, device,steps_done):
    pipline_ff = Pipeline(pipline_data)
    worker = Worker(args)
    states = []
    actionss_ops = []
    actionss_otp = []
    features_c = []
    features_d = []
    ff = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    df_encode = pd.concat([df_c_encode.drop(df_c_encode.columns[-1], axis=1) ,df_d_encode.drop(df_d_encode.columns[-1], axis=1),df_t_norm],axis=1)
    init_state = torch.from_numpy(df_encode.values).float().transpose(0, 1).to(device)
    steps_num = args.steps_num
    for_next = False
    for step in range(steps_num):
        steps_done += 1
        state = init_state
        actions_ops = dqn_ops.choose_action_ops(state, for_next,steps_done)
        actions_otp = dqn_otp.choose_action_otp(actions_ops, for_next,steps_done)
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


    worker.states = states
    worker.actions_ops = actionss_ops
    worker.actions_otp = actionss_otp
    worker.features_c = features_c
    worker.features_d = features_d
    worker.ff = ff
    return worker


def sample_update(args, pipline_data,dqn_ops, dqn_otp,df_c_encode, df_d_encode, df_t_norm, c_ops,d_ops,worker):
    pipline_ff = Pipeline(pipline_data)
    worker = Worker(args)
    states_ = []
    states_ops_ = []
    states_otp_ = []
    actionss_ops_ = []
    actionss_otp_ = []
    features_c_ = []
    features_d_ = []
    ff_ = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    steps_num = args.steps_num
    init_state = worker.states[0]
    for_next = True
    steps_done = 0
    for step in range(steps_num):
        state = init_state
        steps_done += 1
        actions_ops,states_ops = dqn_ops.choose_action_ops(state, for_next,steps_done)
        actions_otp,states_otp = dqn_otp.choose_action_otp(actions_ops,for_next,steps_done)
        fe = parse_actions(actions_ops, actions_otp, c_ops,d_ops,df_c_encode,df_d_encode,n_c_features,n_d_features)
        ff_.append(fe)
        df_c_encode, df_d_encode = pipline_ff.process_data(fe)
        df_c_encode = pd.concat([df_c_encode,df_t_norm],axis=1)
        df_d_encode = pd.concat([df_d_encode, df_t_norm], axis=1)
        x_c = df_c_encode.iloc[:,:-1].astype(np.float32).apply(np.nan_to_num)
        x_d = df_d_encode.iloc[:,:-1].astype(np.float32).apply(np.nan_to_num)
        features_d_.append(x_d)
        features_c_.append(x_c)
        if x_d.shape[0] == 0:
            x_encode_c = np.hstack((x_c,df_t_norm.values.reshape(-1, 1)))
        elif x_c.shape[0] == 0:
            x_encode_c = np.hstack((x_d, df_t_norm.values.reshape(-1, 1)))
        else:
            x_encode_c = np.hstack((x_c, x_d, df_t_norm.values.reshape(-1, 1)))
        x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
        init_state = x_encode_c
        states_.append(state.cpu())
        states_otp_.append(states_otp)
        states_ops_.append(states_ops)
        actionss_ops_.append(actions_ops)
        actionss_otp_.append(actions_otp)


    worker.states_ = states_
    worker.states_otp_ = states_otp_
    worker.states_ops_ = states_ops_
    worker.actions_ops_ = actionss_ops_
    worker.actions_otp_ = actionss_otp_
    worker.features_c_ = features_c_
    worker.features_d_ = features_d_
    worker.ff_ = ff_
    return worker


def multiprocess_reward(args, worker, scores_b, mode, model, metric,y):
    accs = []
    cvs = []
    scores = []
    new_fe_nums = []

    for step in range(args.steps_num):
        x_c = worker.features_c[step]
        x_d = worker.features_d[step]
        x = np.concatenate((x_c, x_d), axis=1)
        acc, cv, score = get_reward(x, y, args, scores_b, mode, model, metric, step)
        accs.append(acc)
        cvs.append(cv)
        scores.append(score)
    worker.fe_nums = new_fe_nums
    worker.accs = accs
    worker.cvs = cvs
    worker.scores = scores
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