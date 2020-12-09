from abc import ABCMeta, abstractmethod
from math import inf

import gym
import gym_pygame

from environments.base_environment import BaseEnvironment


class GymEnvBase(BaseEnvironment):
    __metaclass__ = ABCMeta

    def __init__(self, state_space, action_space):
        super(GymEnvBase, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.env = None

    @abstractmethod
    def env_init(self, env_info):
        pass

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
        super(Catcher, self).__init__(4, 2)

    def env_init(self, env_info):
        self.env = gym.make('Catcher-PLE-v0', init_lives=inf)
        self.env.seed(env_info['seed'])


class PuckWorld(GymEnvBase):
    # TODO normalize is on by default (arg to CatcherEnv())
    def __init__(self):
        super(PuckWorld, self).__init__(8, 4)

    def env_init(self, env_info):
        self.env = gym.make('PuckWorld-PLE-v0')
        self.env.seed(env_info['seed'])
