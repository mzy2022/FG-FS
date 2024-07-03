# -*- coding: utf-8 -*-
import logging
import os
import time
import warnings
from pathlib import Path
from Own.ppo_attention.emedding_data.data_utils import get_unique_categorical_counts, get_categ_cont_target_values, train_val_test_split
import pandas as pd

from Own.feature_eng.data_trans import Pipeline_data
from initial import init_param
from ppo_attention_2 import PPO_psm


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



def main():
    warnings.filterwarnings('ignore')
    _args = init_param()
    # set_cuda(_args.cuda)
    log_config(_args)
    logging.info(f'args : {_args}')
    # read data
    # csv
    ori_data = pd.read_csv(_args.dataset_path)
    # y = ori_data.iloc[:,-1]
    # print(_args.dataset_path)
    # features = _args.continuous_col + _args.discrete_col
    # label = _args.target_col
    # all_data = ori_data.iloc[:,:-1]
    # pipline = Pipeline_data(all_data, _args.continuous_col, _args.discrete_col)
    # all_data = pipline.new_pipline_main()
    # all_data = pd.DataFrame(all_data,columns=features)
    # all_data[label] = y
    all_data = ori_data
    # log settings
    ppo_psm = PPO_psm(_args)
    ppo_psm.ori_data = all_data
    ppo_psm.f_ori_data = all_data.iloc[:, :-1]
    ppo_psm.target = all_data.iloc[:, -1]
    ppo_psm.feature_search()



    current_time = time.time()
    logging.info(f'total_run_time: {current_time - old_time}')



if __name__ == '__main__':

    old_time = time.time()
    main()



