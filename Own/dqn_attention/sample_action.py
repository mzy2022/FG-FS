import logging

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold

from feature_engineer.attention_searching.worker import Worker
from feature_engineer.fe_parsers import parse_actions
from metrics.metric_evaluate import rae_score
from models import model_fuctions
from process_data import Pipeline
from process_data.feature_process import remove_duplication, label_encode_to_onehot


def sample(args, ppo, pipline_args_train, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops, epoch, i, device):
    logging.debug(f'Get in pipline_ff_c for episode {i}...')
    pipline_ff_c = Pipeline(pipline_args_train)
    logging.debug(f'End pipline_ff_c')
    worker_c = Worker(args)
    states_c = []
    actions_c = []
    log_probs_c = []
    features_c = []
    ff_c = []
    steps = []

    worker_d = Worker(args)
    states_d = []
    actions_d = []
    log_probs_d = []
    features_d = []
    ff_d = []

    n_features_c = df_c_encode.shape[1] - 1
    n_features_d = df_d_encode.shape[1] - 1
    init_state_c = torch.from_numpy(df_c_encode.values).float().transpose(0, 1).to(device)
    init_state_d = torch.from_numpy(df_d_encode.values).float().transpose(0, 1).to(device)

    steps_num = args.steps_num
    if i < args.episodes // 2:
        sample_rule = True
    else:
        sample_rule = False
    logging.debug(f'Start sample episode {i}...')
    for step in range(steps_num):
        steps.append(step)

        if df_c_encode.shape[0] > 1:
            state_c = init_state_c

            logging.debug(f'Start choose_action_c for step {step}, state_c: {state_c.shape}')
            actions, log_probs, m1_output, m2_output, m3_output, action_softmax = ppo.choose_action_c(state_c, step, epoch, c_ops,
                                                                                      sample_rule)
            logging.debug(f'Start parse_actions...')
            fe_c = parse_actions(actions, c_ops, n_features_c, continuous=True)
            ff_c.append(fe_c)

            logging.debug(f'Start process_continuous...')
            # x_c_encode, x_c_reward = pipline_ff_c.process_continuous(fe_c)
            x_c_encode, x_c_combine = pipline_ff_c.process_continuous(fe_c)

            logging.debug(f'Start astype, x_c_encode: {x_c_encode.shape}')
            # Process np.nan and np.inf in np.float32
            x_c_encode = x_c_encode.astype(np.float32).apply(np.nan_to_num)
            x_c_combine = x_c_combine.astype(np.float32).apply(np.nan_to_num)
            features_c.append(x_c_combine)
            logging.debug(f'Start hstack...')
            if x_c_encode.shape[0]:
                x_encode_c = np.hstack((x_c_encode, df_t_norm.values.reshape(-1, 1)))
                # x_encode_c = np.hstack((x_c_combine, df_t_norm.values.reshape(-1, 1)))
                x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
                # x_encode_c = torch.from_numpy(x_c_encode.values).float().transpose(0, 1)
                init_state_c = x_encode_c
                states_c.append(state_c.cpu())
                # states_c.append(state_c)
                actions_c.append(actions)
                log_probs_c.append(log_probs)
            else:
                states_c.append(state_c.cpu())
                # states_c.append(state_c)
                actions_c.append(actions)
                log_probs_c.append(log_probs)
            logging.debug(f'End append, state_c: {state_c.shape}')
        if args.combine:
            state_d = init_state_d

            logging.debug(f'Start choose_action_c for step {step}, state_d: {state_d.shape}')
            actions, log_probs, m1_output, m2_output, m3_output, action_softmax = ppo.choose_action_d(state_d, step, epoch, c_ops, sample_rule)
            logging.debug(f'Start parse_actions...')
            fe_d = parse_actions(actions, d_ops, n_features_d, continuous=False)
            ff_d.append(fe_d)
            logging.debug(f'Start process_discrete...')
            x_d_norm, x_d = pipline_ff_c.process_discrete(fe_d)
            # for ff_action in fe_d:
            #     logging.debug(f'Start process_discrete 2...')
            #     x_d_norm, x_d = pipline_ff_c.process_discrete(ff_action)

            # Process np.nan and np.inf in np.float32
            logging.debug(f'Start astype, x_d_norm: {x_d_norm.shape}')
            x_d_norm = x_d_norm.astype(np.float32).apply(np.nan_to_num)
            x_d = x_d.astype(np.float32).apply(np.nan_to_num)
            # x_d_norm = np.nan_to_num(x_d_norm.astype(np.float32))
            # x_d = np.nan_to_num(x_d.astype(np.float32))
            features_d.append(x_d)
            logging.debug(f'Start hstack...')
            try:
                x_encode_d = np.hstack((x_d_norm, df_t_norm.values.reshape(-1, 1)))
            except:
                breakpoint()
            x_encode_d = torch.from_numpy(x_encode_d).float().transpose(0, 1).to(device)
            init_state_d = x_encode_d
            states_d.append(state_d.cpu())
            actions_d.append(actions)
            log_probs_d.append(log_probs)
            logging.debug(f'End append, state_d: {state_d.shape}')
    dones = [False for i in range(steps_num)]
    dones[-1] = True

    worker_c.steps = steps
    worker_c.states = states_c
    worker_c.actions = actions_c
    worker_c.log_probs = log_probs_c
    worker_c.dones = dones
    worker_c.features = features_c
    worker_c.ff = ff_c

    worker_d.steps = steps
    worker_d.states = states_d
    worker_d.actions = actions_d
    worker_d.log_probs = log_probs_d
    worker_d.dones = dones
    worker_d.features = features_d
    worker_d.ff = ff_d
    return worker_c, worker_d