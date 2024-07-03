import argparse
import logging
import os
import sys
import warnings
from time import time

import pandas as pd

from ALL import AutoFE
from config_pool import configs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    file_names = ["yeast"]
    for file_name in file_names:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument("--epochs", type=int, default=500)
        parser.add_argument("--steps_num", type=int, default=6)  ##采样轮次
        parser.add_argument("--episodes", type=int, default=1)  ##worker数量
        parser.add_argument("--file_name", type=str, default=file_name, help='文件名称')
        parser.add_argument("--model", type=str, default='rf', help="lr or xgb or rf or lgb or cat")
        parser.add_argument("--seed", type=int, default=1, help='random seed')
        parser.add_argument("--cv", type=int, default=5)
        parser.add_argument("--n_heads", type=int, default=6)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument('--memory', type=int, default=24, help='memory capacity')
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument('--hidden-size', type=int, default=8)
        parser.add_argument("--train_size", type=float, default=0.8)
        parser.add_argument("--shuffle", type=bool, default=False)
        parser.add_argument("--split_train_test", type=bool, default=False)

        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--d_k", type=int, default=32)
        parser.add_argument("--d_v", type=int, default=32)
        parser.add_argument("--d_ff", type=int, default=64)
        args = parser.parse_args()

        data_configs = configs[args.file_name]
        c_columns = data_configs['c_columns']
        d_columns = data_configs['d_columns']
        target = data_configs['target']
        dataset_path = data_configs["dataset_path"]
        mode = data_configs['mode']
        if mode == 'classify':
            metric = 'f1'
        elif mode == 'regression':
            metric = 'rae'
        args.metric = metric
        args.mode = mode
        args.c_columns = c_columns
        args.d_columns = d_columns
        args.target = target

        df = pd.read_csv(dataset_path)

        start = time()
        autofe = AutoFE(df, args)
        autofe.fit_data(args)
        end = time()
        logging.info(f'cost time: {round((end - start), 4)} s.')
