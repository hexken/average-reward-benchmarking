from environments.base_environment import BaseEnvironment
from environments.catcher_game import Catch
import numpy as np


class Catcher(BaseEnvironment):

    def __init__(self):
        super().__init__()
        self.catcher = None
        self.action = None
        self.rand_generator = None

    def env_init(self, env_info):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """
        assert 'grid_size' in env_info

        self.rand_generator = np.random.RandomState(env_info.get('random_seed', 42))
        self.catcher = Catch(env_info['grid_size'])
        self.reward_obs_term = (0.0, self.catcher.observe(), False)

    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        return self.reward_obs_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if terminal.
        """
        self.reward_obs_term = self.catcher.act(action)

        return self.reward_obs_term
