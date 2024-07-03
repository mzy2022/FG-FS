import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .embedding_policy_network import Actor1, Actor2


class DQN_ops(object):
    def __init__(self, args, data_nums, feature_nums,operations_c, operations_d, d_model, d_k, d_v, d_ff, n_heads,
                 memory,device,EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, dropout=None):
        self.args = args
        self.epochs = args.epochs
        self.episodes = args.episodes
        self.ppo_epochs = args.ppo_epochs
        self.operations_c = operations_c
        self.operations_d = operations_d
        self.device = device
        self.memory = memory
        self.ops_c_num = operations_c
        self.ops_d_num = operations_d
        self.eps_end = EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.TARGET_REPLACE_ITER = 100
        self.batch_size = 8
        self.gamma = 0.99
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.agent1 = Actor1(args, data_nums, feature_nums,operations_c, d_model, d_k, d_v, d_ff, n_heads, dropout=dropout).to(self.device)
        self.agent1_opt = optim.Adam(params=self.agent1.parameters(), lr=args.lr)



    def choose_action_ops(self, input, for_next,steps_done):
        actions_ops = []

        self.agent1.train()
        ops_vals = self.agent1(input.to(self.device), for_next)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        ops_vals = ops_vals.detach()
        for index, out in enumerate(ops_vals):
            if for_next:
                act = torch.argmax(out).item()
            else:
                if np.random.uniform() > eps_threshold:
                    act = torch.argmax(out)
                else:
                    act = np.random.randint(0, self.ops_c_num)
            actions_ops.append(int(act))

        return actions_ops


    def store_transition(self, workers):
        store_ops_dict = {}
        for i,worker in enumerate(workers):
            state = worker.state
            ops = worker.ops
            reward = worker.ops_reward
            state_ = worker.state_
            ops_ = worker.ops_
            store_ops_dict[i] = [state,ops,reward,state_,ops_]
        self.memory.store_transition(store_ops_dict)



    def learn(self, args, workers):
        if self.memory.memory_counter >= self.memory.MEMORY_CAPACITY:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent1.target_net.load_state_dict(self.agent1.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            memory_dict = self.memory.sample()
            for i, memory_list in memory_dict.items():
                q_eval = self.agent1.eval_net(memory_list[0])
                q_next = self.agent1.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * q_next.view(self.batch_size, 1)
                loss += self.loss_func(q_eval, q_target)
            self.agent1_opt.zero_grad()
            loss.backward()
            self.agent1_opt.step()


class DQN_otp(object):
    def __init__(self, args,operations_c,hidden_size,memory,device,EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        self.args = args
        self.epochs = args.epochs
        self.episodes = args.episodes
        self.ppo_epochs = args.ppo_epochs
        self.device = device
        self.memory = memory
        self.otp_nums = operations_c
        self.eps_end =EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.TARGET_REPLACE_ITER = 100
        self.gamma = 0.99
        self.batch_size = 8
        self.agent2 = Actor2(args, operations_c,hidden_size).to(self.device)
        self.agent2_opt = optim.Adam(params=self.agent2.parameters(), lr=args.lr)


    def choose_action_otp(self, input, for_next,steps_done):
        input = torch.tensor(input)
        actions_otp = []
        self.agent2.train()
        otp_vals = self.agent2(input.to(self.device), for_next)
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

        return actions_otp

    def store_transition(self,workers):
        store_otp_dict = {}
        for i, worker in enumerate(workers):
            emb_state = worker.emb_state
            otp = worker.otp
            reward = worker.otp_reward
            emb_state_ = worker.emb_state_
            otp_ = worker.otp_
            store_otp_dict[i] = [emb_state, otp, reward, emb_state_,otp_]
        self.memory.store_transition(store_otp_dict)


    def learn(self, args, workers):
        if self.memory.memory_counter >= self.memory.MEMORY_CAPACITY:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent2.target_net.load_state_dict(self.agent2.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            memory_dict = self.memory.sample()
            for i,memory_list in memory_dict.items():
                state = memory_list[0]
                state_ = memory_list[3]
                q_eval = self.agent2.eval_net(state)
                q_next = self.agent2.target_net(state_)
                q_target = memory_list[2] + self.gamma * q_next.view(self.batch_size, 1)
                loss += self.loss_func(q_eval, q_target)
            self.agent2_opt.zero_grad()
            loss.backward()
            self.agent2_opt.step()


