import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .embedding_policy_network import Actor1, Actor2, Actor3


class DQN_ops(object):
    def __init__(self, args, data_nums, feature_nums, operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads,
                 memory, device, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, dropout=None):
        self.args = args
        self.epochs = args.epochs
        self.episodes = args.episodes
        self.operations_c = operations_c
        self.operations_d = operations_d
        self.device = device
        self.memory = memory
        self.ops_c_num = operations_c
        self.ops_d_num = operations_d
        self.eps_end = EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.TARGET_REPLACE_ITER = 50
        self.batch_size = 8
        self.gamma = 0.99
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.agent1 = Actor1(args, data_nums, feature_nums, operations_c, operations_d,d_model, d_k, d_v, d_ff, n_heads, self.device,
                             dropout=dropout).to(self.device)
        self.agent1_opt = optim.Adam(params=self.agent1.parameters(), lr=args.lr)
        self.nums = max(self.operations_c,self.operations_d)
    def choose_action_ops(self, input, for_next, steps_done, con_or_dis):
        actions_ops = []

        self.agent1.train()
        ops_vals, state = self.agent1(input.to(self.device), for_next, con_or_dis)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        ops_vals = ops_vals.detach()
        for index, out in enumerate(ops_vals):
            if for_next:
                act = torch.argmax(out).item()
            else:
                if np.random.uniform() > eps_threshold:
                    act = torch.argmax(out)
                else:
                    act = np.random.randint(0, self.nums)
            actions_ops.append(int(act))

        return actions_ops, state

    def store_transition(self, args, workers):
        store_ops_list = []
        for num, worker in enumerate(workers):
            states_ops = worker.states_ops
            ops = worker.actions_ops
            reward = worker.reward
            states_ops_ = worker.states_ops_
            ops_ = worker.actions_ops_
            store_ops_list.append([states_ops, ops, reward, states_ops_, ops_])
        self.memory.store_transition(store_ops_list)

    def learn(self, args, workers, device):
        if self.memory.memory_counter >= self.memory.memory_capacity:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent1.target_net.load_state_dict(self.agent1.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            worker_lists = self.memory.sample(args, device)

            for memory_list in worker_lists:
                # with torch.no_grad():
                q_next = self.agent1.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * torch.max(q_next, dim=-1)[0]
                q_eval = self.agent1.eval_net(memory_list[0])
                yyy = torch.tensor(memory_list[1], dtype=torch.long).unsqueeze(0)
                q_eval_selected = q_eval.gather(1, yyy).squeeze(0)
                # q_target = q_target.detach()
                loss = loss + self.loss_func(q_eval_selected, q_target)

            loss /= len(worker_lists)
            self.agent1_opt.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            self.agent1_opt.step()


class DQN_otp(object):
    def __init__(self, args, features_c,features_d,op_nums, hidden_size, d_model, memory, device, EPS_START=0.9, EPS_END=0.05,
                 EPS_DECAY=200):
        self.args = args
        self.epochs = args.epochs
        self.episodes = args.episodes
        self.device = device
        self.memory = memory
        self.otp_nums = max(features_c,features_d)

        self.op_nums = op_nums
        self.eps_end = EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 50
        self.gamma = 0.99
        self.batch_size = 8
        self.agent2 = Actor2(args, features_c,features_d, hidden_size, d_model,features_c,features_d).to(self.device)
        self.agent2_opt = optim.Adam(params=self.agent2.parameters(), lr=args.lr)

    def choose_action_otp(self, input, states_ops, for_next, steps_done):
        input = torch.tensor(input)
        actions_otp = []
        self.agent2.train()
        otp_vals, state = self.agent2(input.to(self.device), states_ops, for_next)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        otp_vals = otp_vals.detach()
        for index, out in enumerate(otp_vals):
            if for_next:
                act = torch.argmax(out).item()
            else:
                if np.random.uniform() > eps_threshold:
                    act = torch.argmax(out)
                else:
                    act = np.random.randint(0, self.otp_nums)
            actions_otp.append(int(act))

        return actions_otp, state

    def store_transition(self, args, workers):
        store_otp_list = []
        for num, worker in enumerate(workers):
            states_otp = worker.states_otp
            otp = worker.actions_otp
            reward = worker.reward
            states_otp_ = worker.states_otp_
            otp_ = worker.actions_otp_
            store_otp_list.append([states_otp, otp, reward, states_otp_, otp_])
        self.memory.store_transition(store_otp_list)

    def learn(self, args, workers, device):
        if self.memory.memory_counter >= self.memory.memory_capacity:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent2.target_net.load_state_dict(self.agent2.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            worker_lists = self.memory.sample(args, device)

            for memory_list in worker_lists:
                # with torch.no_grad():
                q_next = self.agent2.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * torch.max(q_next, dim=-1)[0]
                q_eval = self.agent2.eval_net(memory_list[0])
                xxx = torch.tensor(memory_list[1])
                yyy = xxx.unsqueeze(0)
                q_eval = q_eval.gather(1, yyy).squeeze(0)
                # q_target = q_target.detach()
                loss = loss + self.loss_func(q_eval, q_target)

            loss /= len(worker_lists)
            self.agent2_opt.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            self.agent2_opt.step()


class DQN_features(object):
    def __init__(self, args, c_features, d_features,op_nums, single_nums,hidden_size, d_model, memory, device, EPS_START=0.9, EPS_END=0.05,
                 EPS_DECAY=200):
        self.args = args
        self.epochs = args.epochs
        self.episodes = args.episodes
        self.device = device
        self.memory = memory
        self.c_features = c_features
        self.d_features = d_features
        self.feature_nums = max(len(c_features),len(d_features))
        self.eps_end = EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.loss_func = nn.MSELoss()
        self.op_nums = op_nums
        self.single_nums = single_nums
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 50
        self.gamma = 0.99
        self.batch_size = 8
        self.agent3 = Actor3(args, self.op_nums,self.single_nums, hidden_size, d_model,self.feature_nums).to(self.device)
        self.agent3_features = optim.Adam(params=self.agent3.parameters(), lr=args.lr)

    def choose_action_features(self, input, states_ops, for_next, steps_done):
        input = torch.tensor(input)
        actions_otp = []
        self.agent3.train()
        otp_vals, state = self.agent3(input.to(self.device), states_ops, for_next)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        otp_vals = otp_vals.detach()
        for index, out in enumerate(otp_vals):
            if for_next:
                act = torch.argmax(out).item()
            else:
                if np.random.uniform() > eps_threshold:
                    act = torch.argmax(out)
                else:
                    act = np.random.randint(0, self.feature_nums)
            actions_otp.append(int(act))

        return actions_otp, state

    def store_transition(self, args, workers):
        store_otp_list = []
        for num, worker in enumerate(workers):
            states_otp = worker.states_features
            otp = worker.actions_features
            reward = worker.reward
            states_otp_ = worker.states_features_
            otp_ = worker.actions_features_
            store_otp_list.append([states_otp, otp, reward, states_otp_, otp_])
        self.memory.store_transition(store_otp_list)

    def learn(self, args, workers, device):
        if self.memory.memory_counter >= self.memory.memory_capacity:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent3.target_net.load_state_dict(self.agent3.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            worker_lists = self.memory.sample(args, device)

            for memory_list in worker_lists:
                # with torch.no_grad():
                q_next = self.agent3.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * torch.max(q_next, dim=-1)[0]
                q_eval = self.agent3.eval_net(memory_list[0])
                xxx = torch.tensor(memory_list[1])
                yyy = xxx.unsqueeze(0)
                q_eval = q_eval.gather(1, yyy).squeeze(0)
                # q_target = q_target.detach()
                loss = loss + self.loss_func(q_eval, q_target)

            loss /= len(worker_lists)
            self.agent3_features.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            self.agent3_features.step()
