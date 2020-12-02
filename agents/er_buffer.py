from collections import namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])


class ERBuffer:
    def __init__(self, config, random):
        self.capacity = config['er_buffer_capacity']
        self.buffer = []
        self.next_pos = 0
        self.random = random

    def add(self, *args):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.next_pos] = Experience(*args)
        self.next_pos = (self.next_pos + 1) % self.capacity

    def sample_batch(self, batch_size):
        return self.random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
