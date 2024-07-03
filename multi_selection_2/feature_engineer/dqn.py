import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .embedding_policy_network import Feature_Build


class DQN_ops(object):
    def __init__(self, args, data_nums, feature_nums, d_model, d_k, n_heads,
                 memory, alpha, device, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, dropout=None):
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

        self.eval_net = QNet_ops(d_model, 2).to(self.device)
        self.target_net = QNet_ops(d_model, 2).to(self.device)
        self.eval_net_opt = optim.Adam(self.eval_net.parameters(), lr=args.lr)

    def choose_action_ops(self, state, for_next, steps_done):

        if for_next:
            ops_logits = self.target_net(state)
        else:
            ops_logits = self.eval_net(state)
        ops_logits = torch.where(torch.isnan(ops_logits), torch.full_like(ops_logits, 0), ops_logits)

        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * steps_done / self.eps_decay)
        ops_vals = ops_logits.detach()
        if for_next:
            act = torch.argmax(ops_vals).item()
        else:
            if np.random.uniform() > eps_threshold:
                act = torch.argmax(ops_vals)
            else:
                act = np.random.randint(0, 2)

        return act

    def store_transition(self, state, action, r, state_):
        store_ops_list = []
        states_ops = state
        ops = action
        reward = r
        states_ops_ = state_
        store_ops_list.append([states_ops, ops, reward, states_ops_])
        self.memory.store_transition(store_ops_list)

    def learn(self, args, workers, device, bulid_state_opt):
        if self.memory.memory_counter >= self.memory.memory_capacity:
            if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1
            loss = 0
            worker_lists = self.memory.sample(args, device)

            for memory_list in worker_lists:
                q_next = self.target_net(memory_list[3])
                q_target = memory_list[2] + self.gamma * torch.max(q_next, dim=-1)[0]
                q_eval = self.eval_net(memory_list[0])
                yyy = torch.tensor(memory_list[1], dtype=torch.long).unsqueeze(0).to(device)
                q_eval_selected = q_eval.gather(0, yyy).squeeze(0)
                loss = loss + self.loss_func(q_eval_selected, q_target)

            loss /= len(worker_lists)
            self.eval_net_opt.zero_grad()
            bulid_state_opt.zero_grad()
            loss.backward()
            self.eval_net_opt.step()
            bulid_state_opt.step()


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class QNet_ops(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, init_w=0.1, device=None):
        super(QNet_ops, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(hidden_dim, action_dim)
        self.out.weight.data.normal_(-init_w, init_w)
        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value
