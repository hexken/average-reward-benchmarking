from collections import namedtuple
import numpy as np

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])


class ERBuffer:
    def __init__(self, capacity, rand_generator):
        self.capacity = capacity
        self.buffer = []
        self.next_pos = 0
        self.rand_generator = rand_generator

    def add(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.next_pos] = Experience(*args)
        self.next_pos = (self.next_pos + 1) % self.capacity

    def sample_batch(self, batch_size):
        return self.rand_generator.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
