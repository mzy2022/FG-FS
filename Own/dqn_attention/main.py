import argparse
import copy
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from feature_ops import get_ops
from worker import Worker
from feature_type_recognition import Feature_type_recognition
from feature_pipeline import Pipeline
from utils import get_key_from_dict
from DQN import ClusterDQNNetwork
from config_pool import configs
from metric import downstream_task_new
from replay import RandomClusterReplay


class AutoFE:
    """Main entry for class that implements automated feature engineering (AutoFE)"""

    def __init__(self, input_data: pd.DataFrame, args):

        # Fixed random seed
        self.seed = args.seed
        self.device = torch.device('cuda',args.cuda) if torch.cuda.is_available() and args.cuda != -1 else torch.device('cpu')
        self.get_reward = downstream_task_new()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        self.shuffle = args.shuffle
        # Deal with input parameters
        self.train_size = args.train_size
        self.combine = args.combine
        self.info_ = {}
        self.num_step = 3
        self.STATE_DIM = 128
        self.best_score = 0
        self.MEMORY_CAPACITY = 10
        self.BATCH_SIZE = 128
        self.hidden_dim = 128
        self.ENT_WEIGHT = 1e-3
        self.info_['target'] = args.target
        self.info_['file_name'] = args.file_name
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
        self.dfs_ = {}
        self.dfs_[self.info_['file_name']] = input_data

        # Split or shuffle training and test data if needed
        self.dfs_['FE_train'] = self.dfs_[self.info_['file_name']]
        self.dfs_['FE_test'] = pd.DataFrame()

    def fit_attention(self, args):
        """Fit for searching the best autofe strategy of attention method"""
        LR = 0.01
        df = self.dfs_['FE_train']
        c_columns, d_columns = self.info_['c_columns'], self.info_['d_columns']
        if len(self.info_['d_columns']) == 0:
            args.combine = False
        target, mode, model, metric = self.info_['target'], self.info_['mode'], self.info_['model'], self.info_[
            'metric']

        n_features_c, n_features_d = len(self.info_['c_columns']), len(self.info_['d_columns'])
        c_ops, d_ops = get_ops(n_features_c, n_features_d)

        df_c_encode, df_d_encode = df.loc[:, c_columns + [target]], df.loc[:, d_columns + [target]]
        df_t, df_t_norm = df.loc[:, target], df.loc[:, target]
        # Searching autofe strategy
        data_nums = self.dfs_['FE_train'].shape[0]
        operations_c = len(c_ops)
        operations_d = len(d_ops)

        pipline_args_train = {'dataframe': self.dfs_['FE_train'],
                              'continuous_columns': self.info_['c_columns'],
                              'discrete_columns': self.info_['d_columns'],
                              'label_name': self.info_['target'],
                              'mode': self.info_['mode'],
                              'isvalid': False,
                              'memory': None}

        cluster1_mem = RandomClusterReplay(self.MEMORY_CAPACITY, self.BATCH_SIZE, self.device)
        self.dqn = ClusterDQNNetwork(state_dim=self.STATE_DIM, hidden_dim=self.hidden_dim, out_dim=operations_c,
                                     memory=cluster1_mem, ent_weight=self.ENT_WEIGHT, gamma=0.99, device=self.device)
        if self.device == 'gpu':
            self.dqn = self.dqn.cuda()
        optimizer_c1 = torch.optim.Adam(self.dqn.parameters(), lr=LR)

        old_per = self.get_reward(df)
        best_per = old_per
        df_ori = df.copy()

        worker_c = Worker(args)
        worker_d = Worker(args)
        init_state_c = torch.from_numpy(df_c_encode.values).float().transpose(0, 1)
        init_state_d = torch.from_numpy(df_d_encode.values).float().transpose(0, 1)
        worker_c.states = [init_state_c]
        worker_d.states = [init_state_d]
        worker_c.actions, worker_d.actions, worker_c.steps, worker_d.steps = [], [], [], []
        worker_c.features, worker_d.features, worker_c.ff, worker_d.ff = [], [], [], []
        dones = [False for i in range(args.steps_num)]
        dones[-1] = True
        worker_c.dones, worker_d.dones = dones, dones

        for epoch in range(args.epochs):
            workers_c = []
            workers_d = []
            w_r = []
            for i in range(args.episodes):
                w_c, w_d = self.dqn.sample(args, pipline_args_train, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops,self.device)
                ## 记录w_c 和 w_d 的reward
                for i in range(self.num_step):
                    new_df = w_c.states[i]
                    new_per = self.get_reward(new_df)
                    reward = new_per - old_per
                    w_r[i] = args.param_a * reward
                    old_per = new_per
                    worker = Worker(args)
                    worker.accs = worker_c.accs[step]
                    worker.fe_nums = worker_c.fe_nums[step]
                    worker.scores = worker_c.scores[step]


                w_c_, w_d_ = self.dqn.sample_(args, w_c, w_d, df_t_norm, c_ops, d_ops,self.device)

                self.dqn.store_transition(w_c, w_d, w_r, w_c_, w_d_)
                if self.dqn.memory.memory_counter >= self.dqn.memory.MEMORY_CAPACITY:
                    self.dqn.learn(optimizer_c1)


if __name__ == '__main__':
    file_name = "SPECTF"
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default="0", help='which gpu to use')
    # parser.add_argument('--cuda', type=str, default="False", help='which gpu to use')
    parser.add_argument("--train_size", type=float, default=0.7)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--ppo_epochs", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=10)  ##worker数量
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy_weight", type=float, default=1e-4)
    parser.add_argument("--baseline_weight", type=float, default=0.95)
    parser.add_argument("--gama", type=float, default=0.9)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_k", type=int, default=32)
    parser.add_argument("--d_v", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--worker", type=int, default=1)
    parser.add_argument("--steps_num", type=int, default=3)  ##每个worker的采样数量

    parser.add_argument("--combine", type=bool, default=True, help='whether combine discrete features')
    parser.add_argument("--preprocess", type=bool, default=False, help='whether preprocess data')
    parser.add_argument("--seed", type=int, default=1, help='random seed')
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--cv_train_size", type=float, default=0.7)
    parser.add_argument("--cv_seed", type=int, default=1)
    parser.add_argument("--split_train_test", type=bool, default=False)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--enc_c_pth", type=str, default='', help="pre-trained model path of encoder_continuous")
    parser.add_argument("--enc_d_pth", type=str, default='', help="pre-trained model path of encoder_discrete")
    parser.add_argument("--mode", type=str, default=None, help="classify or regression")
    parser.add_argument("--model", type=str, default='xgb', help="lr or xgb or rf or lgb or cat")
    parser.add_argument("--metric", type=str, default=None, help="f1,ks,auc,r2,rae,mae,mse")
    parser.add_argument("--file_name", type=str, default=file_name, help='task name in config_pool')
    parser.add_argument("--param_a", type=str, default=0.1, help='reward * x')
    parser.add_argument("--cuda", type=int, default=1, help='cuda')
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
    args.model = model
    args.metric = metric
    args.c_columns = c_columns
    args.d_columns = d_columns
    args.target = target
    print(args)
    df = pd.read_csv(dataset_path)

    start = time.time()
    autofe = AutoFE(df, args)
    try:
        autofe.fit_attention(args)
    except Exception as e:
        import traceback
    end = time.time()
    print(f"{start - end}")
