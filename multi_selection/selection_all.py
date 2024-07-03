import logging
import random
import time
import lightgbm as lgb
import lightgbm
import numpy as np
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
from xgboost import XGBClassifier

from feature_engineer import DQN_ops
from feature_engineer.replay import Replay_ops
from feature_engineer.training_ops import sample, multiprocess_reward, sample_update
from feature_engineer.worker import Worker
from utils import log_dir
from tqdm import tqdm




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

        self.is_cuda, self.device = None, None
        self.set_cuda(args.cuda)

    def set_cuda(self, cuda):
        if cuda == 'False':
            self.device = 'cpu'
        else:
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        logging.info(f"Use device:{self.device}")
        return

    def fit_data(self, args):
        self.ori_df = self.ori_df.apply(np.nan_to_num)
        X = self.ori_df.iloc[:, :49]
        y = self.ori_df.iloc[:, 49:]
        # new_y = pd.DataFrame()
        # for i in range(4):
        #     y = self.ori_df.iloc[:, 80+i:81+i]
        #     le = LabelEncoder()
        #     y = pd.DataFrame(le.fit_transform(y))
        #     new_y = pd.concat([new_y,y],axis=1)
        # y = new_y
        train_data_x, train_data_y, val_data_x,val_data_y,test_data_x, test_data_y = split_train_test(X, y, self.train_size,self.val_size, self.seed)



        d_model = args.d_model
        batch_size = args.batch_size
        memory_size = args.memory
        n_heads = args.n_heads
        alpha = args.alpha
        data_nums = train_data_x.shape[0]
        feature_nums = train_data_x.shape[1]
        d_k = args.d_k
        self.memory_ops = Replay_ops(batch_size, memory_size)
        self.dqn_ops = DQN_ops(args, data_nums, feature_nums, d_model, d_k, n_heads, self.memory_ops, alpha,
                               self.device)

        self.steps_done = 0
        ## 强化学习轮次
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
        best = 0
        for epoch in tqdm(range(args.epochs)):
            logging.info(f"now_epoch{epoch}")
            worker = Worker(args)

            worker.action_list = np.zeros(train_data_x.shape[1])
            for i in tqdm(range(feature_nums)):
                # 得到经过两个agent之后的actions，states
                choose_list = worker.action_list
                # if sum(choose_list) > 4:
                #     break
                choose_list[i] = 1
                list_num = []
                for num, j in enumerate(choose_list):
                    if j == 1:
                        list_num.append(num)
                x = train_data_x.iloc[:, list_num]
                y = train_data_y
                worker_dqn = sample(args, i, x, y, self.dqn_ops, worker, self.device, self.steps_done, train_data_x)
                # 得到reward
                worker_reward = multiprocess_reward(args, worker_dqn, train_data_x, train_data_y, val_data_x,
                                                    val_data_y)

                x = worker_dqn.new_x
                worker_dqn_ = sample_update(args, x, y, self.dqn_ops, worker, self.device, self.steps_done)

                self.dqn_ops.store_transition(args, worker_dqn_)
                self.dqn_ops.learn(args, worker_reward, self.device)

            choose_list = worker.action_list
            list_x = []
            for num, j in enumerate(choose_list):
                if j == 1:
                    list_x.append(num)

            rf_classifier.fit(train_data_x.iloc[:, list_x], train_data_y)
            y_pred = rf_classifier.predict(val_data_x.iloc[:, list_x])
            accuracy = accuracy_score(val_data_y, y_pred)
            print("分类准确性：", accuracy)
            if accuracy >= best:
                best = accuracy
                f_list = choose_list

        print(f_list)
        print(best)
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"最终的RF{accuracy}")


        base_model = SVC(kernel='linear')
        model1 = MultiOutputClassifier(base_model, n_jobs=-1)
        model1.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
        y_pred = model1.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"最终的SVC{accuracy}")

        base_model = XGBClassifier(eval_metric='logloss')
        model = MultiOutputClassifier(base_model)
        model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"XGB{accuracy}")

        base_model = DecisionTreeClassifier(random_state=42)
        model = MultiOutputClassifier(base_model)
        model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"DT{accuracy}")

        base_model = lgb.LGBMClassifier()
        model = MultiOutputClassifier(base_model)
        model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"LGB{accuracy}")
def split_train_test(X, y, train_size,val_size, seed):
    rng = np.random.default_rng(seed)
    inds = np.arange(len(X))
    rng.shuffle(inds)
    n_train = int(train_size * len(X))
    n_val = int(val_size * len(X))
    train_inds = inds[:n_train]
    val_inds = inds[n_train:(n_train + n_val)]
    test_inds = inds[(n_train + n_val):]
    train_data_x = X.iloc[train_inds,:]
    train_data_y = y.iloc[train_inds,:]
    val_data_x = X.iloc[val_inds,:]
    val_data_y = y.iloc[val_inds,:]
    test_data_x = X.iloc[test_inds,:]
    test_data_y = y.iloc[test_inds,:]
    return train_data_x, train_data_y, val_data_x,val_data_y,test_data_x, test_data_y
