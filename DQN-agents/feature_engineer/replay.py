import numpy as np


class Replay_ops():
    def __init__(self, size, batch_size):
        self.memory_capacity = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.memory = np.zeros(self.memory_capacity)

    def _sample(self):
        sample_index = np.random.choice(self.memory_capacity, self.BATCH_SIZE)
        return sample_index

    def store_transition(self, ops_dict):
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = ops_dict
        self.memory_counter += 1

    def sample(self):
        sample_index = self._sample()
        memory_dict = self.memory[sample_index]
        return memory_dict


class Replay_otp():
    def __init__(self, size, batch_size):
        self.memory_capacity = size
        self.memory_counter = 0
        self.batch_size = batch_size
        self.memory = np.zeros(self.memory_capacity)

    def store_transition(self, otp_dict):
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = otp_dict
        self.memory_counter += 1

    def _sample(self):
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        return sample_index
    def sample(self):
        sample_index = self._sample()
        memory_dict = self.memory[sample_index]
        return memory_dict


