import numpy as np
import torch


class Replay_ops():
    def __init__(self, size, batch_size):
        self.memory_capacity = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.memory = {}

    def _sample(self):
        sample_index = np.random.choice(self.memory_capacity, self.BATCH_SIZE)
        return sample_index

    def store_transition(self, ops_dict):
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = ops_dict
        self.memory_counter += 1

    def sample(self, args, device):
        worker_list = []
        sample_index = self._sample()
        for i in range(args.episodes):
            states = []
            actions = []
            rewards = []
            states_ = []
            actions_ = []
            for j in sample_index:
                state = self.memory[j][i][0]
                action = self.memory[j][i][1]
                reward = self.memory[j][i][2]
                state_ = self.memory[j][i][3]
                action_ = self.memory[j][i][4]
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                states_.append(state_)
                actions_.append(action_)
            states = torch.cat(states, dim=0).detach()
            actions = [torch.tensor(action) for action in actions]
            actions = torch.cat(actions, dim=0).to(device).detach()
            states_ = torch.cat(states_, dim=0).detach()
            actions_ = [torch.tensor(action) for action in actions_]
            actions_ = torch.cat(actions_, dim=0).detach()
            rewards = torch.tensor(rewards, dtype=torch.float)
            reward = rewards.mean().detach()
            worker_list.append((states, actions, reward, states_, actions_))
        return worker_list


class Replay_otp():
    def __init__(self, size, batch_size):
        self.memory_capacity = size
        self.memory_counter = 0
        self.batch_size = batch_size
        self.memory = {}

    def store_transition(self, otp_dict):
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = otp_dict
        self.memory_counter += 1

    def _sample(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        return sample_index

    def sample(self, args, device):
        worker_list = []
        sample_index = self._sample()
        for i in range(args.episodes):
            states = []
            actions = []
            rewards = []
            states_ = []
            actions_ = []
            for j in sample_index:
                state = self.memory[j][i][0]
                action = self.memory[j][i][1]
                reward = self.memory[j][i][2]
                state_ = self.memory[j][i][3]
                action_ = self.memory[j][i][4]
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                states_.append(state_)
                actions_.append(action_)
            states = torch.cat(states, dim=0).detach()
            actions = [torch.tensor(action) for action in actions]
            actions = torch.cat(actions, dim=0).to(device).detach()
            states_ = torch.cat(states_, dim=0).detach()
            actions_ = [torch.tensor(action) for action in actions_]
            actions_ = torch.cat(actions_, dim=0).detach()
            rewards = torch.tensor(rewards, dtype=torch.float)
            reward = rewards.mean().detach()
            worker_list.append((states, actions, reward, states_, actions_))
        return worker_list


class Replay_features():
    def __init__(self, size, batch_size):
        self.memory_capacity = size
        self.memory_counter = 0
        self.batch_size = batch_size
        self.memory = {}

    def store_transition(self, features_dict):
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = features_dict
        self.memory_counter += 1

    def _sample(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        return sample_index

    def sample(self, args, device):
        worker_list = []
        sample_index = self._sample()
        for i in range(args.episodes):
            states = []
            actions = []
            rewards = []
            states_ = []
            actions_ = []
            for j in sample_index:
                state = self.memory[j][i][0]
                action = self.memory[j][i][1]
                reward = self.memory[j][i][2]
                state_ = self.memory[j][i][3]
                action_ = self.memory[j][i][4]
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                states_.append(state_)
                actions_.append(action_)
            states = torch.cat(states, dim=0).detach()
            actions = [torch.tensor(action) for action in actions]
            actions = torch.cat(actions, dim=0).to(device).detach()
            states_ = torch.cat(states_, dim=0).detach()
            actions_ = [torch.tensor(action) for action in actions_]
            actions_ = torch.cat(actions_, dim=0).detach()
            rewards = torch.tensor(rewards, dtype=torch.float)
            reward = rewards.mean().detach()
            worker_list.append((states, actions, reward, states_, actions_))
        return worker_list
