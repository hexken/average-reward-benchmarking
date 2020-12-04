from abc import ABCMeta, abstractmethod

import gym

from environments.base_environment import BaseEnvironment


class GymEnvBase(BaseEnvironment):
    __metaclass__ = ABCMeta

    def __init__(self):
        super().__init__()
        self.env = None

    def env_init(self, game, env_info):
        self.env = gym.make(game)
        self.env.seed(env_info['seed'])

    def env_start(self):
        return self.env.reset()

    def env_step(self, action):
        obs, reward, term = self.env.step(action)
        self.reward_obs_term = reward, obs, term
        self.env.render()
        return self.reward_obs_term


class Catcher:
    # TODO normalize is on by default (arg to CatcherEnv())
    def env_init(selfself, env_info):
        super().__init__('Catcher-PLE-v0', env_info)
