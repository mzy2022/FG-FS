import argparse
import logging
import os
import sys
import warnings
from time import time

import pandas as pd

from AutoFE import AutoFE
from config_pool import configs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    file_name = 'PimaIndian'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default="0", help='which gpu to use')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--ppo_epochs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=5)  ##worker数量
    parser.add_argument("--file_name", type=str, default=file_name, help='task name in config_pool')
    parser.add_argument("--mode", type=str, default=None, help="classify or regression")
    parser.add_argument("--model", type=str, default='rf', help="lr or xgb or rf or lgb or cat")
    parser.add_argument("--metric", type=str, default=None, help="f1,ks,auc,r2,rae,mae,mse")
    parser.add_argument("--seed", type=int, default=1, help='random seed')
    parser.add_argument("--train_size", type=float, default=0.7)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_k", type=int, default=32)
    parser.add_argument("--d_v", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--worker", type=int, default=1)
    parser.add_argument("--steps_num", type=int, default=5)  ##每个worker的采样数量
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--memory', type=int, default=8, help='memory capacity')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=8)

    args = parser.parse_args()

    data_configs = configs[args.file_name]
    c_columns = data_configs['c_columns']
    d_columns = data_configs['d_columns']
    target = data_configs['target']
    dataset_path = data_configs["dataset_path"]

    mode = data_configs['mode']
    if args.model:
        model = args.model
    else:
        model = data_configs["model"]
    if args.metric:
        metric = args.metric
    else:
        metric = data_configs["metric"]

    if mode == 'classify':
        metric = 'f1'
    elif mode == 'regression':
        metric = 'rae'

    args.mode = mode
    args.metric = metric
    args.model = model
    args.c_columns = c_columns
    args.d_columns = d_columns
    args.target = target

    df = pd.read_csv(dataset_path)

    start = time()
    autofe = AutoFE(df, args)
    try:
        autofe.fit_attention(args)
    except Exception as e:
        import traceback

        logging.info(traceback.format_exc())
    end = time()
    logging.info(f'Total cost time: {round((end - start), 4)} s.')
