from __future__ import absolute_import
from collections import namedtuple, defaultdict
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import numpy as np
import math
from feature_pipeline import Pipeline
from worker import Worker
from replay import Replay
from generate_features import parse_actions

# 初始化神经网络的参数，采用的是均匀分布的方式，并根据输入特征数量进行了缩放
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ClusterNet(nn.Module):
    def __init__(self, STATE_DIM, OUT_DIM,HIDDEN_DIM=100, init_w=0.1):
        super(ClusterNet, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, HIDDEN_DIM)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(HIDDEN_DIM, OUT_DIM)
        self.out.weight.data.normal_(-init_w, init_w)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, out_dim,gamma, device, memory: Replay, ent_weight,c_ops,d_ops,
                 EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, init_w=1e-6):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.GAMMA = gamma
        self.ENT_WEIGHT = ent_weight
        self.cuda_info = device is not None
        self.memory = memory
        self.learn_step_counter = 0
        self.init_w = init_w
        self.TARGET_REPLACE_ITER = 100
        self.BATCH_SIZE = 8
        self.loss_func = nn.MSELoss()
        self.c_ops = c_ops
        self.d_ops = d_ops

    def learn(self, optimizer):
        raise NotImplementedError()


class ClusterDQNNetwork(DQNNetwork):
    def __init__(self, state_dim, hidden_dim, out_dim,memory, ent_weight,
                 gamma=0.99, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, device=None, init_w=1e-6):
        super(ClusterDQNNetwork, self).__init__(state_dim,hidden_dim, out_dim,gamma, device, memory,
                                                ent_weight, EPS_START=EPS_START,
                                                EPS_END=EPS_END, EPS_DECAY=EPS_DECAY, init_w=init_w)

        self.state_dim = state_dim
        self.out_dim = out_dim
        self.eval_net = ClusterNet(self.state_dim, self.out_dim, HIDDEN_DIM=self.hidden_dim,init_w=self.init_w)
        self.target_net = ClusterNet(self.state_dim, self.out_dim, HIDDEN_DIM=self.hidden_dim, init_w=self.init_w)
        self.generate_state =
    def get_q_value(self, state_emb):
        return self.eval_net(state_emb)

    def get_q_value_next(self, state_emb):
        return self.target_net(state_emb)

    def forward(self, X=None, for_next=False):
        state_emb = self.generate_state.get_emb(pd.DataFrame(X))
        state_emb = torch.FloatTensor(state_emb)
        if self.cuda_info:
            state_emb = state_emb.cuda()
        if for_next:
            q_vals = self.get_q_value_next(state_emb)
        else:
            q_vals = self.get_q_value(state_emb)
        q_vals = q_vals.detach()
        return q_vals, state_emb

    def store_transition(self, s1, a1, r, s2, a2):
        self.memory.store_transition((s1, a1, r, s2, a2))

    # 𝐿 =∑𝑙𝑜𝑔𝜋𝜃(𝑠𝑡, 𝑎𝑡)(𝑟 + 𝛾𝑉(𝑠𝑡 + 1)−𝑉(𝑠𝑡))
    def learn(self, optimizer):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_, b_a_ = self.memory.sample()
        net_input = b_s
        q_eval = self.eval_net(net_input)
        net_input_ = b_s_
        q_next = self.target_net(net_input_)
        q_target = b_r + self.GAMMA * q_next.view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def select_action(self, X, for_next=False, steps_done=0):
        q_vals, state_emb = self.forward(X,for_next=for_next)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1.0 * steps_done / self.EPS_DECAY)
        if for_next:
            q_val = torch.cat(q_vals)
            acts = torch.argmax(q_val).item()
        else:
            if np.random.uniform() > eps_threshold:
                q_val = torch.cat(q_vals)
                acts = torch.argmax(q_val).item()
            else:
                acts = np.random.randint(0, len(self.c_ops))

        return acts, state_emb

    def sample(self,args, pipline_args_train, df_c_encode, df_d_encode, df_t_norm, c_ops, d_ops,device):
        for epoch in range(args.epochs):    ###轮次
            workers_c = []
            workers_d = []
            w_r = []
            pipline_ff_c = Pipeline(pipline_args_train)
            worker_c = Worker(args)
            states_c = []
            actions_c = []
            features_c = []
            ff_c = []
            steps = []

            worker_d = Worker(args)
            states_d = []
            actions_d = []
            features_d = []
            ff_d = []

            n_features_c = df_c_encode.shape[1] - 1
            n_features_d = df_d_encode.shape[1] - 1
            init_state_c = torch.from_numpy(df_c_encode.values).float().transpose(0, 1).to(device)
            init_state_d = torch.from_numpy(df_d_encode.values).float().transpose(0, 1).to(device)

            steps_num = args.steps_num
            for i in range(args.episodes):   ###每个worker采样多少state




                for step in range(args.steps_num):  ####一共有多少worker
                    steps.append(step)
                    if df_c_encode.shape[0] > 1:
                        state_c = init_state_c
                        actions, emb_state = self.select_action(state_c, for_next=False)
                        fe_c = parse_actions(actions, c_ops, n_features_c, continuous=True)
                        ff_c.append(fe_c)
                        x_c_encode, x_c_combine = pipline_ff_c.process_continuous(fe_c)
                        # Process np.nan and np.inf in np.float32
                        x_c_encode = x_c_encode.astype(np.float32).apply(np.nan_to_num)
                        x_c_combine = x_c_combine.astype(np.float32).apply(np.nan_to_num)
                        features_c.append(x_c_combine)

                        if x_c_encode.shape[0]:
                            x_encode_c = np.hstack((x_c_encode, df_t_norm.values.reshape(-1, 1)))
                            x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
                            init_state_c = x_encode_c
                            states_c.append(state_c.cpu())
                            actions_c.append(actions)
                        else:
                            states_c.append(state_c.cpu())
                            actions_c.append(actions)

                    if args.combine:
                        state_d = init_state_d
                        actions, emb_state = self.select_action(state_d,for_next=False)
                        fe_d = parse_actions(actions, d_ops, n_features_d, continuous=False)
                        ff_d.append(fe_d)
                        x_d_norm, x_d = pipline_ff_c.process_discrete(fe_d)
                        # Process np.nan and np.inf in np.float32
                        x_d_norm = x_d_norm.astype(np.float32).apply(np.nan_to_num)
                        x_d = x_d.astype(np.float32).apply(np.nan_to_num)
                        features_d.append(x_d)
                        try:
                            x_encode_d = np.hstack((x_d_norm, df_t_norm.values.reshape(-1, 1)))
                        except:
                            breakpoint()
                        x_encode_d = torch.from_numpy(x_encode_d).float().transpose(0, 1).to(device)
                        init_state_d = x_encode_d
                        states_d.append(state_d.cpu())
                        actions_d.append(actions)


                    #######计算state的奖励



                    #######由state产生新的state


                    #####store(s,a,r,n_s,done)

                if self.dqn.memory.memory_counter >= self.dqn.memory.MEMORY_CAPACITY:
                    self.dqn.learn(optimizer_c1)

                dones = [False for i in range(steps_num)]
                dones[-1] = True

                worker_c.steps = steps
                worker_c.states = states_c
                worker_c.actions = actions_c
                worker_c.dones = dones
                worker_c.features = features_c
                worker_c.ff = ff_c

                worker_d.steps = steps
                worker_d.states = states_d
                worker_d.actions = actions_d
                worker_d.dones = dones
                worker_d.features = features_d
                worker_d.ff = ff_d






    def sample_(self,args, w_c, w_d, df_t_norm, c_ops, d_ops, device):
        pipline_ff_c = Pipeline(pipline_args_train)
        worker_c = Worker(args)
        states_c = []
        actions_c = []
        features_c = []
        ff_c = []
        steps = []

        worker_d = Worker(args)
        states_d = []
        actions_d = []
        features_d = []
        ff_d = []

        n_features_c = df_c_encode.shape[1] - 1
        n_features_d = df_d_encode.shape[1] - 1
        init_state_c = torch.from_numpy(df_c_encode.values).float().transpose(0, 1).to(device)
        init_state_d = torch.from_numpy(df_d_encode.values).float().transpose(0, 1).to(device)

        steps_num = args.steps_num

        for step in range(steps_num):
            steps.append(step)
            if df_c_encode.shape[0] > 1:
                state_c = init_state_c
                actions, emb_state = self.select_action(state_c, step, c_ops)
                fe_c = parse_actions(actions, c_ops, n_features_c, continuous=True)
                ff_c.append(fe_c)
                x_c_encode, x_c_combine = pipline_ff_c.process_continuous(fe_c)
                # Process np.nan and np.inf in np.float32
                x_c_encode = x_c_encode.astype(np.float32).apply(np.nan_to_num)
                x_c_combine = x_c_combine.astype(np.float32).apply(np.nan_to_num)
                features_c.append(x_c_combine)

                if x_c_encode.shape[0]:
                    x_encode_c = np.hstack((x_c_encode, df_t_norm.values.reshape(-1, 1)))
                    x_encode_c = torch.from_numpy(x_encode_c).float().transpose(0, 1).to(device)
                    init_state_c = x_encode_c
                    states_c.append(state_c.cpu())
                    actions_c.append(actions)
                else:
                    states_c.append(state_c.cpu())
                    actions_c.append(actions)

            if args.combine:
                state_d = init_state_d
                actions, emb_state = self.select_action(state_d, step, d_ops)
                fe_d = parse_actions(actions, d_ops, n_features_d, continuous=False)
                ff_d.append(fe_d)
                x_d_norm, x_d = pipline_ff_c.process_discrete(fe_d)
                # Process np.nan and np.inf in np.float32
                x_d_norm = x_d_norm.astype(np.float32).apply(np.nan_to_num)
                x_d = x_d.astype(np.float32).apply(np.nan_to_num)
                features_d.append(x_d)
                try:
                    x_encode_d = np.hstack((x_d_norm, df_t_norm.values.reshape(-1, 1)))
                except:
                    breakpoint()
                x_encode_d = torch.from_numpy(x_encode_d).float().transpose(0, 1).to(device)
                init_state_d = x_encode_d
                states_d.append(state_d.cpu())
                actions_d.append(actions)
        dones = [False for i in range(steps_num)]
        dones[-1] = True

        worker_c.steps = steps
        worker_c.states = states_c
        worker_c.actions = actions_c
        worker_c.dones = dones
        worker_c.features = features_c
        worker_c.ff = ff_c

        worker_d.steps = steps
        worker_d.states = states_d
        worker_d.actions = actions_d
        worker_d.dones = dones
        worker_d.features = features_d
        worker_d.ff = ff_d
        return worker_c, worker_d
