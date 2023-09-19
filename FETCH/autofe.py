import copy
import logging
import os
import pickle
import random
import time

import multiprocessing
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold, cross_val_score

from feature_engineer import PPO, Memory
from feature_engineer import get_ops
from feature_engineer.attention_searching.training_ops import sample, multiprocess_reward, apply_actions
from feature_engineer.attention_searching.worker import Worker
from feature_engineer.fe_parsers import parse_actions
from metrics.metric_evaluate import metric_fuctions
from metrics.metric_evaluate import rae_score
from models.model_evaluate import *
from process_data import feature_type_recognition, feature_pipeline
from process_data.feature_pipeline import Pipeline
from process_data.feature_process import label_encode_to_onehot, features_process, remove_duplication,split_train_test
from utils import log_dir, get_key_from_dict, reduce_mem_usage



# 这段代码定义了一个名为get_test_score的函数，接受一些参数以计算模型在测试集上的评分，并返回得分。
def get_test_score(df_train, df_test, label_train, label_test, args, mode,model,metric):
    if args.worker == 0 or args.worker == 1:
        n_jobs = -1
    else:
        n_jobs = 1
    model = model_fuctions[f"{model}_{mode}"](n_jobs)
    model.fit(df_train, label_train)
    # pred = model.predict(df_test)
    score = metric_fuctions[metric](model, df_test, label_test, label_train)


class AutoFE:
    def __init__(self,input_data:pd.DataFrame,args):
        # Create log directory
        times = time.strftime('%Y%m%d-%H%M')
        log_path = fr"./logs/train/{args.file_name}_{times}"
        if args.enc_c_pth != '':
            log_path = fr"./logs/pre/{args.file_name}_{args.enc_c_pth.split('_')[4].split('.')[0]}_{times}"
            log_dir(log_path)
            logging.info(args)
            logging.info(f'File name: {args.file_name}')
            logging.info(f'Data shape: {input_data.shape}')

            self.seed = args.seed
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            os.environ["PYTHONHASHSEED"] = str(self.seed)
            self.shuffle = args.shuffle

            self.train_size = args.train_size
            self.split = args.split_train_test
            self.combine = args.combine
            self.info_ = {}
            self.info_['target'] = args.target
            self.info_['file_name'] = args.file_name
            self.info_['mode'] = args.mode
            self.info_['metric'] = args.metric
            self.info_['model'] = args.model
            if args.c_columns is None or args.d_columns is None:
                # Detect if a feature column is continuous or discrete
                feature_type_recognition = Feature_type_recognition()
                feature_type = feature_type_recognition.fit(input_data.drop(columns=self.info_['target']))
                args.d_columns = get_key_from_dict(feature_type,'cat')
                args.c_columns = get_key_from_dict(feature_type, 'num')
            self.info_['c_columns'] = args.c_columns
            self.info_['d_columns'] = args.d_columns

            for col in input_data.columns:
                col_type = input_data[col].dtype
                if col_type != 'object':
                    input_data[col].fillna(0,inplace=True)
                else:
                    input_data[col].fillna('unknown',inplace=True)
            self.dfs_ = {}
            self.dfs_[self.info_['file_name']] = input_data

            # Split or shuffle training and test data if needed
            self.dfs_['FE_train'] = self.dfs_[self.info_['file_name']]
            self.dfs_['FE_test'] = pd.DataFrame()
            if self.split:
                self.dfs_['FE_train'], self.dfs_['FE_test'] = split_train_test(self.dfs_[self.info_['file_name']],self.info_['d_columns'],self.info_['target'],self.info_['mode'],self.train_size,self.seed,self.shuffle)
                self.dfs_['FE_train'].reset_index(inplace=True, drop=True)
                self.dfs_['FE_test'].reset_index(inplace=True, drop=True)
            elif self.shuffle:
                self.dfs_['FE_train'] = self.dfs_['FE_train'].sample(frac=1, random_state=self.seed).reset_index(drop=True)
            feature_pipeline.Candidate_features = self.dfs_['FE_train'].copy()
            self.is_cuda, self.device = None, None
            self.set_cuda(args.cuda)
            logging.info(f'Done AutoFE initialization.')

    def set_cuda(self,cuda):
        if cuda == 'False':
            self.device = 'cpu'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda
            self.is_cuda = torch.cuda.is_available()
            self.device = torch.device('cuda:0') if self.is_cuda else torch.device('cpu')
            if self.is_cuda:
                logging.info(f"Use device: {cuda}, {self.device}, {torch.cuda.get_device_name(self.device)}")
                return
        logging.info(f"Use device: {self.device}")

    def fit_attention(self,args):
        """
        获取输入数据以及相关设置。
        计算基线分数（baseline score）并记录。
        如果进行数据集拆分，计算在测试集上的基线分数。
        根据参数决定是否对数据进行预处理。
        准备工作并初始化一些变量和数据结构。
        进行多个迭代周期。
        对于每个周期的每个特征工程实例，进行采样或并行采样。
        应用特征工程操作，并获取应用后的数据。
        计算奖励，并更新每个特征工程实例的状态。
        验证搜索得到的特征工程实例的性能。
        更新ppo模型的网络参数。
        重复上述步骤，直至完成所有的迭代周期
        :param args:
        :return:
        """
        df = self.dfs_['FE_train']
        c_columns, d_columns = self.info_['c_columns'], self.info_['d_columns']
        if len(self.info_['d_columns']) == 0:
            args.combine = False
            target, mode, model, metric = self.info_['target'], self.info_['mode'], self.info_['model'],self.info_['metric']

            pool = multiprocessing.Pool(processes=args.worker)

            n_features_c, n_features_d = len(self.info_['c_columns']), len(self.info_['d_columns'])
            c_ops, d_ops = get_ops(n_features_c, n_features_d)













