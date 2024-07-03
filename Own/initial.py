import os
import sys
import torch
import warnings
from init_dataset import dataset_config
BASE_DIR = os.path.dirname(os.path.abspath('.'))
sys.path.append(BASE_DIR)

torch.manual_seed(0)
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
os.environ['DATA_ABS_PATH'] = BASE_DIR + '/data/processed'

import argparse


def init_param():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--data', type=str, default="spectf", help='dataset name')
    parser.add_argument('--model', type=str, default="rf")
    parser.add_argument('--cuda', type=int, default=0, help='which gpu to use')  # -1 represent cpu-only
    parser.add_argument('--is_select', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--target_col', type=str, default="label", help='random seed')
    # init_dataset.py ---> args's arri.
    args = parser.parse_args()
    search_dataset_info = dataset_config[args.data]
    for key, value in search_dataset_info.items():
        setattr(args, key, value)
    return args
