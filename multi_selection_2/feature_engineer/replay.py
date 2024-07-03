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

    def sample(self,args,device):
        worker_list = []
        sample_index = self._sample()
        for j in sample_index:
            state = self.memory[j][0][0].detach()
            action = self.memory[j][0][1]
            reward = self.memory[j][0][2]
            state_ = self.memory[j][0][3].detach()
            worker_list.append((state,action,reward,state_))
        return worker_list
