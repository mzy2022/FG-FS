from collections import defaultdict
import numpy as np
import torch


class Replay:
    def __init__(self, size, batch_size, device):
        self.MEMORY_CAPACITY = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.cuda_info = device is not None

    def _sample(self):
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        return sample_index

    def sample(self):
        raise NotImplementedError()

    def store_transition(self, resource):
        raise NotImplementedError()


class RandomClusterReplay(Replay):
    def __init__(self, size, batch_size,device):
        super(RandomClusterReplay, self).__init__(size, batch_size, device)
        self.memory = []


    def store_transition(self, mems):
        s, a, r, s_, a_ = mems
        transition = [s, a, r, s_, a_]
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index] = transition
        self.memory_counter += 1

    def sample(self):
        sample_index = self._sample()
        b_memory = self.memory[sample_index]
        w_c = [sublist[0] for sublist in b_memory]
        w_d = [sublist[1] for sublist in b_memory]
        w_r = [sublist[2] for sublist in b_memory]
        w_c_ = [sublist[3] for sublist in b_memory]
        w_d_ = [sublist[4] for sublist in b_memory]
        return w_c, w_d, w_r, w_c_, w_d_
