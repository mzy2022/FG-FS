import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold

from DQN_my.metric_evaluate import rae_score
from DQN_my.model_evaluate import model_fuctions
from DQN_my.process_data.pipeline_data import Pipeline
from DQN_my.fe_parsers import parse_actions
from DQN_my.feature_engineer.worker import Worker
from DQN_my.process_data.update_data import Update_data

def sample(args, dqn_ops, dqn_otp,pipline_data, df_c_encode, df_d_encode, df_t_norm, c_ops,d_ops,epoch, device,steps_done):
    pipline_ff = Pipeline(pipline_data)
    worker = Worker(args)
    states = []
    c_d = []
    statess_ops = []
    statess_otp = []
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
        actions_ops,states_ops = dqn_ops.choose_action_ops(state, for_next,steps_done)
        actions_otp,states_otp = dqn_otp.choose_action_otp(actions_ops,states_ops, for_next,steps_done)
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
        statess_otp.append(states_otp)
        statess_ops.append(states_ops)
        c_d.append(x_encode_c)


    worker.states = states
    worker.states_ops = statess_ops
    worker.states_otp = statess_otp
    worker.actions_ops = actionss_ops
    worker.actions_otp = actionss_otp
    worker.features_c = features_c
    worker.features_d = features_d
    worker.ff = ff
    worker.c_d = c_d
    return worker


def sample_update(args, pipline_data,dqn_ops, dqn_otp,df_c_encode, df_d_encode, df_t_norm, c_ops,d_ops,worker,device):

    states_ = []
    states_ops_ = []
    states_otp_ = []
    actionss_ops_ = []
    actionss_otp_ = []
    features_c_ = []
    features_d_ = []
    ff_ = []
    x_encodes = []
    n_c_features = df_c_encode.shape[1] - 1
    n_d_features = df_d_encode.shape[1] - 1
    steps_num = args.steps_num
    for_next = True
    steps_done = 0
    for step in range(steps_num):
        pipline_ff = Update_data(worker.features_c[step],worker.features_d[step],pipline_data)
        state = worker.c_d[step]
        steps_done += 1
        actions_ops,states_ops = dqn_ops.choose_action_ops(state, for_next,steps_done)
        actions_otp,states_otp = dqn_otp.choose_action_otp(actions_ops,states_ops,for_next,steps_done)
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
        states_.append(state.cpu())
        states_otp_.append(states_otp)
        states_ops_.append(states_ops)
        actionss_ops_.append(actions_ops)
        actionss_otp_.append(actions_otp)
        x_encodes.append(x_encode_c)


    worker.states_ = states_
    worker.states_otp_ = states_otp_
    worker.states_ops_ = states_ops_
    worker.actions_ops_ = actionss_ops_
    worker.actions_otp_ = actionss_otp_
    worker.features_c_ = features_c_
    worker.features_d_ = features_d_
    worker.ff_ = ff_
    worker.c_d_ = x_encodes
    return worker


def multiprocess_reward(args, worker, scores_b, mode, model, metric,y):
    accs = []
    cvs = []
    scores = []
    new_fe_nums = []
    real_pds = []
    rewards = []
    for step in range(args.steps_num):
        x_c = worker.features_c[step]
        x_d = worker.features_d[step]
        if x_c.shape[0] == 0:
            x = np.array(x_d)
        elif x_d.shape[0] == 0:
            x = np.array(x_c)
        else:
            x = np.concatenate((x_c, x_d), axis=1)
        y = np.array(y)
        acc, cv, score = get_reward(x, y, args, scores_b, mode, model, metric, step)
        accs.append(acc)
        cvs.append(cv)
        scores.append(score)
        rewards.append(np.mean(scores_b) - np.mean(scores))
    worker.fe_nums = new_fe_nums
    worker.accs = accs
    worker.cvs = cvs
    worker.scores = scores
    worker.rewards = rewards
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




    values = np.array(scores) - np.array(scores_b)
    mask = values < 0
    negative = values[mask]
    negative_sum = negative.sum()
    reward = np.array(scores).mean() + negative_sum
    if step == (args.steps_num - 1):
        reward = reward - repeat_ratio
    return round(reward, 4), values, scores


def f1(X,y):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    clf = RandomForestClassifier(random_state=0)
    f1_list = []
    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        f1_list.append(f1_score(y_test, y_predict, average='weighted'))
    return np.array(f1_list)