# -*- coding: utf-8 -*-
import logging
import time
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import feature_type_recognition
import reduce_scale
from config import dataset_config
from ppo_ori import PPO_ori
from ppo_psm import PPO_psm
from pipline_thread_2N_batch_singlevalue import Pipline
from utility.metrics import calculate_f1_score



def log_config(args):
    """
        log Configuration information, specifying the saving path of output log file, etc
        :return: None
    """
    dataset_path = Path(args.dataset_path)
    dataset_name = dataset_path.stem
    exp_dir = 'search_{}_{}'.format(dataset_name, time.strftime("%Y%m%d-%H%M%S"))
    exp_log_dir = Path('Catch_log') / exp_dir
    # save argss
    setattr(args,'exp_log_dir',exp_log_dir)
    if not os.path.exists(exp_log_dir):
        os.mkdir(exp_log_dir)
    log_format = '%(asctime)s - %(levelname)s : %(message)s'
    logging.basicConfig(filename=exp_log_dir / 'log.txt', level=logging.INFO,format=log_format, datefmt='%Y/%m/%d %H:%M:%S')
    fh = logging.FileHandler(exp_log_dir / 'log.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    return exp_log_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Catch')
    parser.add_argument('--data', type=str, default="spectf", help='dataset name')
    parser.add_argument('--model', type=str, default="rf")
    parser.add_argument('--cuda', type=int, default=0, help='which gpu to use')  # -1 represent cpu-only
    parser.add_argument('--coreset', type=int, default=0,help='whether to use coreset')  # 1 represent work with coreset
    parser.add_argument('--core_size', type=int, default=10000, help='size of coreset')  # m-->sample size
    parser.add_argument('--psm', type=int, default=1,help='whether to use policy-set-merge')  # >0 represent work with psm, and the value eauals the number of Policy-set
    parser.add_argument('--is_select', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    # init_dataset.py ---> args's arri.
    search_dataset_info = dataset_config[args.data]
    for key, value in search_dataset_info.items():
        setattr(args, key, value)
    return args


def main():
    warnings.filterwarnings('ignore')
    _args = parse_args()
    # setup environments
    # set_cuda(_args.cuda)
    log_config(_args)
    logging.info(f'args : {_args}')
    # read data
    # csv
    all_data = pd.read_csv(_args.dataset_path)
    pipline = Pipline(_args.continuous_col,_args.discrete_col)

    # fill num
    for x in _args.continuous_col:
        all_data[x].replace('?', np.nan, inplace=True)
        all_data[x].replace("NA", np.nan, inplace=True)
        all_data[x] = all_data[x].astype(float)
        mean = np.nanmean(all_data[x])
        all_data[x].fillna(mean, inplace=True)
    for x in _args.discrete_col:
        all_data[x].replace('?', np.nan, inplace=True)
        all_data[x].replace("NA", np.nan, inplace=True)
        all_data[x] = all_data[x].astype(float)
        mean = np.nanmean(all_data[x])
        all_data[x].fillna(mean, inplace=True)
        all_data[x] = all_data[x].astype(int)
        # all_data[x].fillna("*Unique", inplace=True)

    nlp = None

    # automatic recognition
    if (not _args.continuous_col) and (not _args.discrete_col):
        T = feature_type_recognition.Feature_type_recognition()
        T.fit(all_data)
        if _args.target_col in T.num:
            T.num.remove(_args.target_col)
        elif _args.target_col in T.cat:
            T.cat.remove(_args.target_col)
        _args.continuous_col = T.num
        _args.discrete_col = T.cat


    features = _args.continuous_col + _args.discrete_col
    label = _args.target_col
    all_data = all_data[features + [label]]
    # log settings

    # do coreset
    if _args.coreset:
        # find coreset
        new_data = reduce_scale.reduce_scale(all_data, _args)
        new_data.to_csv(f'data/coreset_{_args.data}_{_args.core_size}.csv', index=False)

        # policy-merge-ensemble
        if _args.psm > 0:
            ppo_psm = PPO_psm(_args, nlp_feature = nlp)
            ppo_psm.policy_nums = _args.psm
            ppo_psm.search_data = new_data
            # search
            ppo_psm.feature_search()
            actions = ppo_psm.final_action
            get_reward_ins = ppo_psm.get_reward_ins
            logging.info(f'final_action: {actions}\r')
            # base
            base_score = get_reward_ins.k_fold_score(
                all_data, [], is_base=True)
            logging.info(f'base_score:{base_score.mean()}')
            # apply the actions on original dataset
            new_score, new_columns, fe_num = get_reward_ins.k_fold_score(
                all_data, actions)
            logging.info(f'new_score:{new_score.mean()}')
            logging.info(f'fe_num:{fe_num}')

        else:
            ppo_ori = PPO_ori(_args, nlp_feature = nlp)
            ppo_ori.search_data = new_data
            ppo_ori.feature_search()

            features = _args.continuous_col + _args.discrete_col
            all_data = all_data[features + [label]]

            get_reward_ins = ppo_ori.get_reward_ins
            base_score = get_reward_ins.k_fold_score(
                all_data, [], is_base=True)
            logging.info(f'base_score:{np.mean(base_score)}')

            # apply the actions on original dataset
            new_score, new_columns, fe_num = get_reward_ins.k_fold_score(
                all_data, ppo_ori.best_trans)
            logging.info(f'new_score:{np.mean(new_score)}')
            logging.info(f'fe_num:{fe_num}')

    else:
        if _args.psm > 0:
            ppo_psm = PPO_psm(_args, nlp_feature = nlp)
            ppo_psm.policy_nums = _args.psm
            ppo_psm.search_data = all_data

            ppo_psm.feature_search()
            actions = ppo_psm.final_action
            get_reward_ins = ppo_psm.get_reward_ins

            logging.info(f'final_action: {actions}\r')
            new_df, new_columns, label = pipline.create_action_fes(actions,all_data,
                                                                   task_type=_args.task_type,target_col=_args.target_col,
                                                                   train=True,test=True)
            new_df = pd.DataFrame(new_df)
            new_df.columns = new_columns
            new_df['label'] = label
            exp_log_dir_csv = log_config(_args) / 'log.csv'
            new_df.to_csv(exp_log_dir_csv,index=False)
            # new_f1_score = calculate_f1_score(new_df,_args.target_col,_args.model, _args.task_type)
            # ori_f1_score = calculate_f1_score(all_data,_args.target_col,_args.model,_args.task_type)
            # print(new_f1_score)
            # print(ori_f1_score)

            #base
            base_score = get_reward_ins.k_fold_score(
                all_data, [], is_base=True)
            logging.info(f'base_score:{np.mean(base_score)}')
            #apply action plan
            new_score, new_columns, fe_num = get_reward_ins.k_fold_score(
                all_data, actions)
            logging.info(f'new_score:{np.mean(new_score)}')
            logging.info(f'fe_num:{fe_num}')
        else:

            ppo_ori = PPO_ori(_args, nlp_feature = nlp)
            ppo_ori.search_data = all_data
            # PPO search
            ppo_ori.feature_search()

            features = _args.continuous_col + _args.discrete_col
            all_data = all_data[features + [label]]

            get_reward_ins = ppo_ori.get_reward_ins
            base_score = get_reward_ins.k_fold_score(
                all_data, [], is_base=True)
            logging.info(f'base_score:{np.mean(base_score)}')

            # apply the actions on original dataset
            new_score, new_columns, fe_num = get_reward_ins.k_fold_score(
                all_data, ppo_ori.best_trans)
            logging.info(f'new_score:{np.mean(new_score)}')
            logging.info(f'fe_num:{fe_num}')
    current_time = time.time()
    logging.info(f'total_run_time: {current_time - old_time}')



if __name__ == '__main__':

    old_time = time.time()
    main()












