from abc import ABCMeta, abstractmethod
from math import inf

import gym
import gym_pygame

from environments.base_environment import BaseEnvironment


class GymEnvBase(BaseEnvironment):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.env = None

    def env_init(self, game, env_info):
        super().env_init()  # does nothing atm, but perhaps in the future..
        self.env = gym.make(game, init_lives=inf)
        self.env.seed(env_info['seed'])

    def env_start(self):
        return self.env.reset()

    def env_step(self, action):
        obs, reward, term, _ = self.env.step(action)  # we don't use the auxiliary info
        self.reward_obs_term = reward, obs, term
        # self.env.render()
        return self.reward_obs_term


class Catcher(GymEnvBase):
    # TODO normalize is on by default (arg to CatcherEnv())
    def __init__(self):
        super(Catcher, self).__init__()
        self.action_space = 2
        self.state_space = 4

    def env_init(self, env_info):
        super().env_init('Catcher-PLE-v0', env_info)


class PuckWorld(GymEnvBase):
    # TODO normalize is on by default (arg to CatcherEnv())
    def __init__(self):
        super(PuckWorld, self).__init__()
        self.action_space = 4
        self.state_space = 8

    def env_init(self, env_info):
        super().env_init('PuckWorld-PLE-v0', env_info)
