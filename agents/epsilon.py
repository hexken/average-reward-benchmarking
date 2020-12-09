from numpy import clip


class Epsilon(object):

    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, warmup_steps=0, decay_period=1000):
        self.epsilon_end = epsilon_end
        self.epsilon_interval = epsilon_start - epsilon_end
        self.warmup_steps = warmup_steps
        self.decay_period = decay_period

    def __call__(self, time_step):
        steps_left = self.decay_period + self.warmup_steps - time_step
        bonus = self.epsilon_interval * steps_left / self.decay_period
        bonus = clip(bonus, 0., self.epsilon_interval)
        return self.epsilon_end + bonus
