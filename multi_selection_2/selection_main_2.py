import argparse
import logging
import os
import sys
import warnings
from time import time
import pandas as pd
from selection_all import AutoSelection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore")
path = os.path.dirname(os.path.realpath(__file__))

feature_nums = {'emotions':72,'Birds':260,'CHD_49':49,'flags':19,'foodtruck':21,'Image':294,'scene':294,'water-quality-nom':16,
                'yeast':103,'forest':54}

if __name__ == '__main__':
    file_names = ["flags"]
    for file_name in file_names:
        parser = argparse.ArgumentParser()
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--file_name", type=str, default=file_name, help='文件名称')
        parser.add_argument("--model", type=str, default='rf', help="lr or xgb or rf or lgb or cat")
        parser.add_argument("--seed", type=int, default=0, help='random seed')
        parser.add_argument("--n_heads", type=int, default=6)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument('--memory', type=int, default=24, help='memory capacity')
        parser.add_argument('--batch-size', type=int, default=8)
        parser.add_argument("--train_size", type=float, default=0.7)
        parser.add_argument("--val_size", type=float, default=0.1)
        parser.add_argument("--d_model", type=int, default=128)
        parser.add_argument("--alpha", type=float, default=0)
        parser.add_argument("--d_k", type=int, default=32)
        args = parser.parse_args()

        dataset_path = fr"{path}/data/{file_name}.csv"
        df = pd.read_csv(dataset_path)
        args.feature_nums = feature_nums[file_name]
        start = time()
        autofe = AutoSelection(df, args)
        autofe.fit_data(args)
        end = time()
        logging.info(f'cost time: {round((end - start), 4)} s.')
