import logging
import random
import time
import lightgbm as lgb
import lightgbm
import numpy as np
import urllib.request as urllib2
import pandas as pd
import pynvml
import torch
import xgboost
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.adapt import MLkNN
from torch import optim
from xgboost import XGBClassifier
from feature_engineer.training import sample, get_reward, sample
from feature_engineer.worker import Worker
from utils import log_dir
from tqdm import tqdm
from PPO_FS_test.feature_engineer.PPO import PPO

class AutoSelection:
    def __init__(self, input_data: pd.DataFrame, args):
        times = time.strftime('%Y%m%d-%H%M')
        log_path = fr"./logs/{args.file_name}_{times}"
        log_dir(log_path)
        logging.info(args)
        logging.info(f'File name: {args.file_name}')
        logging.info(f'Data shape: {input_data.shape}')
        # Fixed random seed
        self.seed = args.seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.info_ = {}
        self.best_score = 0
        self.info_['file_name'] = args.file_name
        self.train_size = args.train_size
        self.val_size = args.val_size
        self.ori_df = input_data
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.cosine = args.cosine
        self.feature_nums = args.feature_nums
        self.alpha = args.alpha
        self.d_model = args.d_model
        self.select_feature_nums = args.select_feature_nums
    def fit_data(self, args):
        self.ori_df = self.ori_df.apply(np.nan_to_num)
        X = self.ori_df.iloc[:, :args.feature_nums]
        y = self.ori_df.iloc[:, args.feature_nums:]

        train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y = split_train_test(X, y,self.train_size,self.val_size,self.seed)
        adj_maxtrix = get_adj_matrix(train_data_x,self.cosine,args.select_feature_nums)
        data_nums = train_data_x.shape[0]
        ## 强化学习轮次

        self.ppo = PPO(args, data_nums,self.feature_nums, self.select_feature_nums, self.d_model,self.alpha,self.device)
        best = 0
        rewards = []
        best_actions = None
        for epoch in tqdm(range(args.epochs)):


            print(torch.cuda.memory_allocated()/1024 ** 2)
            print(torch.cuda.max_memory_allocated() / 1024 ** 2)

            worker = Worker(args)
            worker = sample(args, train_data_x, train_data_y,self.select_feature_nums, adj_maxtrix,self.ppo, worker, self.device)
            worker,actions,reward = get_reward(args, worker,train_data_x, train_data_y, val_data_x, val_data_y)
            rewards.append(reward)
            if reward > best:
                best = reward
                # adj_maxtrix = worker.adj_matrix
                print(f"{best}%%%%%%%%%%%")
                logging.info(best)
                best_actions = actions
            self.ppo.update(worker)

        f_reward = get_finnal_reard(train_data_x, train_data_y,val_data_x, val_data_y,test_data_x, test_data_y,best_actions)
        logging.info(f"finnal_reard{f_reward}")
        logging.info(rewards)
        logging.info(best_actions)



def split_train_test(X, y, train_size, val_size, seed):
    rng = np.random.default_rng(seed)
    inds = np.arange(len(X))
    rng.shuffle(inds)
    n_train = int(train_size * len(X))
    n_val = int(val_size * len(X))
    train_inds = inds[:n_train]
    val_inds = inds[n_train:(n_train + n_val)]
    test_inds = inds[(n_train + n_val):]
    train_data_x = X.iloc[train_inds, :]
    train_data_y = y.iloc[train_inds, :]
    val_data_x = X.iloc[val_inds, :]
    val_data_y = y.iloc[val_inds, :]
    test_data_x = X.iloc[test_inds, :]
    test_data_y = y.iloc[test_inds, :]
    return train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y


# def split_train_test(X, y, train_size, val_size, seed):
#
#     train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data'
#     train_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels'
#     val_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_valid.data'
#     val_resp_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/madelon_valid.labels'
#
#     train_X = np.loadtxt(urllib2.urlopen(train_data_url)).astype('float32')
#     train_y = np.loadtxt(urllib2.urlopen(train_resp_url))
#     test_X =  np.loadtxt(urllib2.urlopen(val_data_url)).astype('float32')
#     test_y =  np.loadtxt(urllib2.urlopen(val_resp_url))
#
#     rng = np.random.default_rng(seed)
#     inds = np.arange(len(train_X))
#     rng.shuffle(inds)
#     n_train = int(0.8 * len(train_X))
#     n_val = int(val_size * len(X))
#     train_inds = inds[:n_train]
#     val_inds = inds[n_train:(n_train + n_val)]
#
#
#     train_data_x = train_X[train_inds, :]
#     train_data_y = train_y[train_inds]
#     val_data_x = train_X[val_inds, :]
#     val_data_y = train_y[val_inds]
#     train_data_x = pd.DataFrame(train_data_x)
#     train_data_y = pd.DataFrame(train_data_y)
#     val_data_x = pd.DataFrame(val_data_x)
#     val_data_y = pd.DataFrame(val_data_y)
#     test_X = pd.DataFrame(test_X)
#     test_y = pd.DataFrame(test_y)
#     return train_data_x, train_data_y,val_data_x, val_data_y,test_X, test_y


def get_adj_matrix(feature_matrix,cosine,select_nums):
    feature_matrix = np.array(feature_matrix)
    feature_matrix = torch.tensor(feature_matrix).float()
    norms = torch.norm(feature_matrix, dim=0, keepdim=True)
    norms[norms == 0] = 0.1

    # 归一化特征向量
    normalized_features = feature_matrix / norms

    adj_list = []
    # 计算归一化后的特征向量的点积，即余弦相似度矩阵
    cosine_similarity_matrix = np.array(torch.mm(normalized_features.t(), normalized_features))
    for i in range(0, len(cosine_similarity_matrix)):
        for j in range(i + 1, len(cosine_similarity_matrix)):
            adj_list.append(cosine_similarity_matrix[i][j])
    adj_list = sorted(adj_list, reverse=True)
    x = int(cosine * len(adj_list))
    threshold = adj_list[x]
    cosine_similarity_matrix[cosine_similarity_matrix <= threshold] = 0


    return cosine_similarity_matrix
    # feature_nums = feature_matrix.shape[1]
    # numbers = list(range(0, feature_nums))
    # selected_numbers = random.sample(numbers, select_nums)
    # identity_matrix = np.eye(feature_nums)
    # for num in selected_numbers:
    #     identity_matrix[:,num] = 1
    # return identity_matrix








def get_finnal_reard(X_train, Y_train,val_data_x, val_data_y, X_test, Y_test,actions):
    X_train = pd.concat([X_train,val_data_x],axis=0)
    Y_train = pd.concat([Y_train, val_data_y], axis=0)
    actions = np.array(actions)
    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X_train.iloc[:, actions], Y_train)
    y_pred = model.predict(X_test.iloc[:, actions])
    Y_test = Y_test.values
    accuracy = accuracy_score(y_pred, Y_test)
    return accuracy



import os
import psutil


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used




