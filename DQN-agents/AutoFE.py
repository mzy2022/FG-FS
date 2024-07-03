import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from fe_operations import get_binning_df
from fe_operations import get_ops
from feature_engineer import DQN_ops, DQN_otp
from feature_engineer.replay import Replay_ops, Replay_otp
from feature_engineer.training_ops import sample, multiprocess_reward, sample_update
from feature_engineer.worker import Worker
from metric_evaluate import metric_fuctions
from metric_evaluate import rae_score
from model_evaluate import *
from process_data import Feature_type_recognition
from utils import log_dir, get_key_from_dict


def get_test_score(df_train, df_test, label_train, label_test, args, mode, model, metric):
    model = model_fuctions[f"{model}_{mode}"]
    model.fit(df_train,df_test)
    score = metric_fuctions[metric](model, df_test, label_test, label_train)
    return score

class AutoFE:
    def __init__(self,input_data:pd.DataFrame, args):
        times = time.strftime('%Y%m%d-%H%M')
        log_path = fr"./logs/train/{args.file_name}_{times}"
        log_dir(log_path)
        logging.info(args)
        logging.info(f'File name: {args.file_name}')
        logging.info(f'Data shape: {input_data.shape}')
        # Fixed random seed
        self.seed = args.seed
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        # Deal with input parameters
        self.train_size = args.train_size
        self.info_ = {}
        self.best_score = 0
        self.info_['target'] = args.target
        self.info_['file_name'] = args.file_name
        self.info_['mode'] = args.mode
        self.info_['metric'] = args.metric
        self.info_['model'] = args.model
        if args.c_columns is None or args.d_columns is None:
            # Detect if a feature column is continuous or discrete
            feature_type_recognition = Feature_type_recognition()
            feature_type = feature_type_recognition.fit(input_data.drop(columns=self.info_['target']))
            args.d_columns = get_key_from_dict(feature_type, 'cat')
            args.c_columns = get_key_from_dict(feature_type, 'num')
        self.info_['c_columns'] = args.c_columns
        self.info_['d_columns'] = args.d_columns

        for col in input_data.columns:
            col_type = input_data[col].dtype
            if col_type != 'object':
                input_data[col].fillna(0, inplace=True)
            else:
                input_data[col].fillna('unknown', inplace=True)
        self.ori_df = input_data

        self.is_cuda, self.device = None, None
        self.set_cuda(args.cuda)


    def set_cuda(self, cuda):
        if cuda == 'False':
            self.device = 'cpu'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda
            self.is_cuda = torch.cuda.is_available()
            self.device = torch.device('cuda:0') if self.is_cuda else torch.device('cpu')
            if self.is_cuda:
                logging.info(f"Use device: {cuda}, {self.device}, {torch.cuda.get_device_name(self.device)}")
                return

    def fit_attention(self, args):
        df = self.ori_df
        c_columns, d_columns = self.info_['c_columns'], self.info_['d_columns']
        target, mode, model, metric = self.info_['target'], self.info_['mode'], self.info_['model'], self.info_['metric']
        new_df,new_c_columns,new_d_columns = get_binning_df(df,c_columns,d_columns,mode)
        n_features_c, n_features_d = len(new_c_columns), len(new_d_columns)
        c_ops, d_ops, sps = get_ops(n_features_c, n_features_d)
        # Get baseline score of 5-fold cross validation
        score_b, scores_b = self._get_cv_baseline(new_df, args, mode, model, metric)
        logging.info(f'score_b={score_b}, scores_b={scores_b}')
        # processing data
        df_c_encode, df_d_encode = new_df.loc[:, new_c_columns + [target]], new_df.loc[:, new_d_columns + [target]]
        # x_d_onehot, df_d_labelencode = new_df.loc[:, new_d_columns], new_df.loc[:, new_d_columns]
        df_t, df_t_norm = new_df.loc[:, target], new_df.loc[:, target]
        feature_nums = n_features_c + n_features_d
        data_nums = self.ori_df.shape[0]
        operations_c = len(c_ops)
        operations_d = len(d_ops)
        d_model = args.d_model
        d_k = args.d_k
        d_v = args.d_v
        d_ff = args.d_ff
        n_heads = args.n_heads
        batch_size = args.batch_size
        memory_size = args.memory
        hidden_size = args.hidden_size
        self.memory_ops = Replay_ops(batch_size, memory_size)
        self.memory_otp = Replay_otp(batch_size, memory_size)
        self.dqn_ops = DQN_ops(args, data_nums,feature_nums,operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads, self.memory_ops,self.device)
        self.dqn_otp = DQN_otp(args, operations_c,hidden_size,self.memory_otp, self.device)
        self.steps_done = 0
        pipline_data = {'dataframe': new_df,
                        'continuous_columns': new_c_columns,
                        'discrete_columns': new_d_columns,
                        'label_name':target,
                        'mode': mode
                        }

        self.workers_top5 = []
        worker = Worker(args)
        init_state = torch.from_numpy(new_df.values).float().transpose(0, 1)
        worker.states = [init_state]
        worker.actions,  worker.steps = [], []
        worker.features, worker.ff = [], []

        ## 强化学习轮次
        for epoch in range(args.epochs):
            epoch_best_score = 0
            workers = []
            logging.debug(f'Start Sampling......')
            ###worker 数量
            for i in range(args.episodes):
                # 得到经过两个agent之后的actions，states
                worker_dqn = sample(args, self.dqn_ops, self.dqn_otp,pipline_data, df_c_encode, df_d_encode, df_t_norm, c_ops,d_ops,epoch, self.device,self.steps_done)
                workers.append(worker_dqn)
                # 得到reward
            for num, worker in enumerate(workers):
                w = multiprocess_reward(args, worker, scores_b, mode,model, metric,df_t.values)
                workers[num] = w
            # 得到经过两个agent之后的actions_，states_
            for num, worker in enumerate(workers):
                worker_dqn_ = sample_update(args, self.dqn_ops,self.dqn_otp,worker)
                workers.append(worker_dqn_)

            for num, worker in enumerate(workers):
                self.dqn_ops.store_transition(worker)
                self.dqn_otp.store_transition(worker)

            for i, worker in enumerate(workers):
                worker_x = workers[i]
                logging.info(f"worker{i + 1} ,results:{worker.accs},cv:{worker.cvs[-1]},")
                for step in range(args.steps_num):
                    worker = Worker(args)
                    worker.accs = worker_x.accs[step]
                    worker.scores = worker_x.scores[step]

                    if worker.scores.mean() > epoch_best_score:
                        epoch_best_score = worker.scores.mean()
                    if sum(worker.scores)/len(worker.scores) > self.best_score:
                        self.best_score = sum(worker.scores)/len(worker.scores)
                        print(f"%%%%%%%%%%%%%%{self.best_score}************")
                        xxx = worker_x.states[0].permute(1,0).cpu().numpy()
                        df = pd.DataFrame(xxx)
                        df.to_csv("test.csv")
                    self.workers_top5.append(worker)

            baseline = np.mean([worker.accs for worker in workers], axis=0)
            logging.info(f"epoch:{epoch},baseline:{baseline},score_b:{score_b},scores_b:{scores_b}")

            self.workers_top5.sort(key=lambda worker1: worker1.scores.mean(), reverse=True)
            self.workers_top5 = self.workers_top5[0:5]

            for i in range(2):
                logging.info(f"top_{i + 1}:score:{self.workers_top5[i].scores.mean()}")


            self.dqn_ops.learn(args,workers)
            self.dqn_otp.learn(args,workers)


    def _get_cv_baseline(self, df: pd.DataFrame, args, mode, model, metric):
        target = self.info_["target"]
        model = model_fuctions[f"{model}_{mode}"]()
        args.seed = None
        if mode == "classify":
            my_cv = StratifiedKFold(n_splits=args.cv, shuffle=args.shuffle, random_state=args.seed)
        else:
            my_cv = KFold(n_splits=args.cv, shuffle=args.shuffle, random_state=args.seed)

        X = df.drop(columns=[target])
        y = df[target]
        scores = []

        if mode == "classify":
            if metric == 'f1':
                scores = cross_val_score(model, X, y, scoring='f1_micro', cv=my_cv, error_score="raise")
            elif metric == 'auc':
                auc_scorer = make_scorer(roc_auc_score, needs_proba=True, average="macro", multi_class="ovo")
                scores = cross_val_score(model, X, y, scoring=auc_scorer, cv=my_cv, error_score="raise")
        else:
            if metric == 'mae':
                scores = cross_val_score(model, X, y, cv=my_cv, scoring='neg_mean_absolute_error')
            elif metric == 'mse':
                scores = cross_val_score(model, X, y, cv=my_cv, scoring='neg_mean_squared_error')
            elif metric == 'r2':
                scores = cross_val_score(model, X, y, cv=my_cv, scoring='r2')
            elif metric == 'rae':
                scores = cross_val_score(model, X, y, cv=my_cv, scoring=rae_score)
        return np.array(scores).mean(), scores




