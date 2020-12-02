# base imports
from environments.base_environment import BaseEnvironment, OneHotEnv
import numpy as np

# PLE imports
from ple import PLE
from ple.games.catcher import Catcher
from ple.games.puckworld import PuckWorld

class PLEEnv(BaseEnvironment):
    def __init__(self):
        super().__init__()
        reward = None
        observation = None
        termination = None
        self.reward_obs_term = (reward, observation, termination)

    def env_init(self, env_info={}):
        self.p = PLE(self.game, fps=30, display_screen=True, force_fps=False)
        self.start_state = np.array(list(self.p.getGameState().values()))
        self.reward_obs_term = [0.0, None, False]
        self.actions = self.p.getActionSet()
        self.num_actions = len(self.actions)
        self.num_states = self.start_state.shape

    def env_start(self):
        self.current_state = self.start_state
        self.reward_obs_term[1] = self.current_state

        return self.reward_obs_term[1]

    def env_step(self, action):
        '''
        if self.p.game_over():
            self.p.reset_game()
        '''
        true_action = self.actions[action]
        reward = self.p.act(true_action)
        self.current_state = np.array(list(self.p.getGameState().values()))

        self.reward_obs_term = [reward, self.current_state, False]

        return self.reward_obs_term


class CatcherEnv(PLEEnv):
    def __init__(self):
        super().__init__()
        self.game = Catcher(init_lives=np.inf)


class PuckWorldEnv(PLEEnv):
    def __init__(self):
        super().__init__()
        self.game = PuckWorld()