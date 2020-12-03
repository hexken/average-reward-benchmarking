from numpy import clip


class Epsilon:
    """
    adapted from https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py.

    Returns epsilon_start if time_step <= warmup_steps
        Otherwise it decays linearly from epsilon_start to epsilon_end over decay_period

    """

    def __init__(self, epsilon_start=1.0, epsilon_end=0.1, warmup_steps=0, decay_period=1000):
        self.epsilon_interval = epsilon_start - epsilon_end
        self.epsilon_end = epsilon_end
        self.warmup_steps = warmup_steps
        self.decay_period = decay_period

    def get_epsilon(self, time_step):
        steps_left = self.decay_period + self.warmup_steps - time_step
        bonus = self.epsilon_interval * steps_left / self.decay_period
        bonus = clip(bonus, 0., self.epsilon_interval)
        return self.epsilon_end + bonus
