import logging
import math
import os
import random
import time
import lightgbm as lgb
import lightgbm
import numpy as np
import urllib.request as urllib2
import pandas as pd
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
from feature_engineer.training import get_reward, sample
from feature_engineer.meta_training import inner_sample, inner_update,inner_get_reward,outer_get_reward
from feature_engineer.worker import Worker
from utils import log_dir
from tqdm import tqdm
from PPO_FS_meta.feature_engineer.PPO import PPO

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
        adj_maxtrix = get_adj_matrix(train_data_x,self.cosine)
        data_nums = train_data_x.shape[0]
        each_feature_nums = [math.ceil(0.1 * m * train_data_x.shape[1]) for m in range(1,3)]
        ## 强化学习轮次

        self.ppo = PPO(args, data_nums,self.feature_nums, self.select_feature_nums, self.d_model,self.alpha,self.device)


        if args.need_meta:
            for epoch in tqdm(range(args.meta_epochs)):
                workers = []
                for episode in range(len(each_feature_nums)):
                    select_feature_nums = each_feature_nums[episode]
                    worker = Worker(args)
                    worker.select_feature_nums = select_feature_nums
                    worker = inner_sample(train_data_x, train_data_y,select_feature_nums, adj_maxtrix,self.ppo, worker, self.device)
                    worker,actions,reward = inner_get_reward(args, worker,train_data_x, train_data_y, val_data_x, val_data_y)
                    worker = inner_update(args,train_data_x, train_data_y, select_feature_nums, adj_maxtrix, self.ppo, worker,
                                    self.device)
                    worker, actions, reward = outer_get_reward(args, worker, train_data_x, train_data_y, val_data_x, val_data_y)
                    workers.append(worker)
                self.ppo.meta_update(workers)
            save_model(args,self.ppo.policy,self.ppo.policy_opt)

        else:

            checkpoint = torch.load(f"./params/{args.file_name}_policy.pth")
            self.ppo.policy.load_state_dict(checkpoint['net'])
            self.ppo.policy_opt.load_state_dict(checkpoint['opt'])
            best = 0
            rewards = []
            best_actions = None
            for epoch in tqdm(range(args.epochs)):
                worker = Worker(args)
                worker = sample(train_data_x, train_data_y, self.select_feature_nums, adj_maxtrix, self.ppo, worker,
                                self.device)
                worker, actions, reward = get_reward(args, worker, train_data_x, train_data_y, val_data_x, val_data_y)
                rewards.append(reward)
                if reward > best:
                    best = reward
                    print(f"{best}%%%%%%%%%%%")
                    best_actions = actions
                    logging.info(best)
                self.ppo.update(worker)
            f_reward = get_finnal_reard(train_data_x, train_data_y,val_data_x, val_data_y,test_data_x, test_data_y,best_actions)
            print(f_reward)
            print(rewards)
            best_actions = sorted(best_actions)
            print(best_actions)
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


def get_adj_matrix(feature_matrix,cosine):
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


def save_model(args,actor_c,actor_c_opt):
    dir = f"./params"
    name = args.file_name
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save({"net": actor_c.state_dict(), "opt": actor_c_opt.state_dict()},f"{dir}/{name}_policy.pth")