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
from torch import optim
from xgboost import XGBClassifier

from feature_engineer import DQN_ops
from feature_engineer.replay import Replay_ops
from feature_engineer.training_ops import sample, multiprocess_reward, sample_update
from feature_engineer.worker import Worker
from utils import log_dir
from tqdm import tqdm
from multi_selection_2.feature_engineer.embedding_policy_network import Feature_Build


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
        global state
        self.ori_df = self.ori_df.apply(np.nan_to_num)
        X = self.ori_df.iloc[:, :args.feature_nums]
        y = self.ori_df.iloc[:, args.feature_nums:]

        train_data_x, train_data_y, val_data_x, val_data_y, test_data_x, test_data_y = split_train_test(X, y,
                                                                                                        self.train_size,
                                                                                                        self.val_size,
                                                                                                        self.seed)

        d_model = args.d_model
        batch_size = args.batch_size
        memory_size = args.memory
        n_heads = args.n_heads
        alpha = args.alpha
        data_nums = train_data_x.shape[0]
        feature_nums = train_data_x.shape[1]
        d_k = args.d_k

        self.bulid_state = Feature_Build(args, data_nums, feature_nums, d_model, d_k, n_heads, alpha, self.device).to(
            self.device)
        self.bulid_state_opt = optim.Adam(params=self.bulid_state.parameters(), lr=args.lr)
        self.steps_done = 0
        ## 强化学习轮次

        best = 0
        dqn_list = []
        for agent in range(feature_nums):
            self.memory_ops = Replay_ops(batch_size, memory_size)
            self.dqn_ops = DQN_ops(args, data_nums, feature_nums, d_model, d_k, n_heads, self.memory_ops, alpha,
                                   self.device)
            dqn_list.append(self.dqn_ops)

        for epoch in tqdm(range(args.epochs)):
            logging.info(f"now_epoch{epoch}")
            worker = Worker(args)

            if epoch == 0:
                action_list = np.random.randint(2, size=feature_nums)
                i = 0
                while sum(action_list) < 2:
                    np.random.seed(i)
                    action_list = np.random.randint(2, size=feature_nums)
                    i += 1

                X_selected = train_data_x.iloc[:, action_list == 1]
                state = self.bulid_state(X_selected, train_data_y)
                action_list_p = action_list

            action_list = np.zeros(feature_nums)
            for agent, dqn in enumerate(dqn_list):
                action_list[agent] = dqn.choose_action_ops(state, False, self.steps_done)

            while sum(action_list) < 2:
                i = epoch
                np.random.seed(i)
                action_list = np.random.randint(2, size=feature_nums)
                i += 1

            worker.states_ops = state
            worker.action_list = action_list
            # 得到reward
            model = RandomForestClassifier(n_estimators=10, random_state=0)
            model.fit(train_data_x.iloc[:, action_list == 1], train_data_y)
            y_pred = model.predict(val_data_x.iloc[:, action_list == 1])
            Y_val = val_data_y.values
            accuracy = accuracy_score(y_pred, Y_val)
            action_list_change = np.array([x or y for (x, y) in zip(action_list_p, action_list)])
            r_list = accuracy / sum(action_list_change) * action_list_change

            X_selected = train_data_x.iloc[:, action_list == 1]
            state_ = self.bulid_state(X_selected, train_data_y)
            worker.states_ops_ = state_

            for agent, dqn in enumerate(dqn_list):
                dqn.store_transition(state, action_list[agent], r_list[agent], state_)

            for dqn in dqn_list:
                dqn.learn(args, worker, self.device, self.bulid_state_opt)

            state = state_

            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
            rf_classifier.fit(train_data_x.iloc[:, action_list == 1], train_data_y)
            y_pred = rf_classifier.predict(val_data_x.iloc[:, action_list == 1])
            accuracy = accuracy_score(val_data_y, y_pred)
            print("分类准确性：", accuracy)
            if accuracy >= best:
                best = accuracy
                f_list = action_list

        logging.info(f"best{best}")
        print(f_list)
        print(best)
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        model.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"最终的RF{accuracy}")
        logging.info(f"RF{accuracy}")

        # base_model = SVC(kernel='linear')
        # model1 = MultiOutputClassifier(base_model, n_jobs=-1)
        # model1 = base_model
        # model1.fit(train_data_x.iloc[:, f_list == 1], train_data_y)
        # y_pred = model1.predict(test_data_x.iloc[:, f_list == 1])
        # accuracy = accuracy_score(y_pred, test_data_y)
        # print(f"最终的SVC{accuracy}")
        # logging.info(f"SVC{accuracy}")

        base_model = XGBClassifier(eval_metric='logloss')
        # model = MultiOutputClassifier(base_model)
        model = base_model
        le = LabelEncoder()
        label_y = le.fit_transform(train_data_y)
        yyy = le.fit_transform(test_data_y)
        model.fit(train_data_x.iloc[:, f_list == 1], label_y)
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(yyy, y_pred)
        print(f"XGB{accuracy}")
        logging.info(f"XGB{accuracy}")

        base_model = DecisionTreeClassifier(random_state=42)
        model = base_model
        # model = MultiOutputClassifier(base_model)
        model.fit(train_data_x.iloc[:, f_list == 1], train_data_y.values.reshape(-1))
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"DT{accuracy}")
        logging.info(f"DT{accuracy}")

        base_model = lgb.LGBMClassifier()
        model = base_model
        # model = MultiOutputClassifier(base_model)
        model.fit(train_data_x.iloc[:, f_list == 1], train_data_y.values.reshape(-1))
        y_pred = model.predict(test_data_x.iloc[:, f_list == 1])
        accuracy = accuracy_score(y_pred, test_data_y)
        print(f"LGB{accuracy}")
        logging.info(f"LGB{accuracy}")

        logging.info(f"f_list{f_list}")

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
