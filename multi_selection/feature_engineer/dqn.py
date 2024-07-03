import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .embedding_policy_network import Actor1


class DQN_ops(object):
    def __init__(self, args, data_nums, feature_nums,d_model, d_k, n_heads,
                 memory, alpha,device, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, dropout=None):
        self.args = args
        self.epochs = args.epochs
        self.device = device
        self.memory = memory
        self.eps_end = EPS_END
        self.eps_start = EPS_START
        self.eps_decay = EPS_DECAY
        self.TARGET_REPLACE_ITER = 50
        self.batch_size = 8
        self.gamma = 0.99
        self.loss_func = nn.MSELoss()
        self.learn_step_counter = 0
        self.agent1 = Actor1(args, data_nums, feature_nums, d_model, d_k,n_heads,alpha, self.device).to(self.device)
        self.agent1_opt = optim.Adam(params=self.agent1.parameters(), lr=args.lr)

    def choose_action_ops(self, input_x,input_y, for_next, steps_done):
        self.agent1.train()
        ops_vals, state = self.agent1(input_x.to(self.device),input_y.to(self.device), for_next)
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        ops_vals = ops_vals.detach()
        if for_next:
            act = torch.argmax(ops_vals).item()
        else:
            if np.random.uniform() > eps_threshold:
                act = torch.argmax(ops_vals)
            else:
                act = np.random.randint(0, 2)

        return act, state

    def store_transition(self, args, worker):
        store_ops_list = []

        states_ops = worker.states_ops
        ops = worker.actions_ops
        reward = worker.reward_1
        states_ops_ = worker.states_ops_
        store_ops_list.append([states_ops, ops,reward, states_ops_])
        self.memory.store_transition(store_ops_list)

    def learn(self, args, workers, device):
        if self.memory.memory_counter >= self.memory.memory_capacity:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.agent1.target_net.load_state_dict(self.agent1.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            worker_lists = self.memory.sample(args, device)

            for memory_list in worker_lists:
                q_next = self.agent1.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * torch.max(q_next, dim=-1)[0]
                q_eval = self.agent1.eval_net(memory_list[0])
                yyy = torch.tensor(memory_list[1], dtype=torch.long).unsqueeze(0).to(device)
                q_eval_selected = q_eval.gather(0, yyy).squeeze(0)
                loss = loss + self.loss_func(q_eval_selected, q_target)

            loss /= len(worker_lists)
            self.agent1_opt.zero_grad()
            loss.backward()
            self.agent1_opt.step()