import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

from utils.helpers import, argmax
from agents.base_agent import BaseAgent
from agents.fa_agents import MLPControlAgent

class DifferentialQlearningAgent(MLPControlAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    def __init__(self, config):
        super().__init__(config)

        self.weights_f = None
        self.average_value = None
        self.alpha_w_f = None
        self.alpha_r_f = None

    def get_qs(self, observation):
        """returns action value vector q:S->R^{|A|}
        Args:
            observation: ndarray
        Returns:
        """
        return self.model.predict(observation)

    def max_action_value_f(self, observation):
        """
        returns the higher-order action value corresponding to the
        maximum lower-order action value for the given observation.
        Note: this is not max_a q_f(s,a)
        """
        q_f_sa = self.get_value_f(self.get_representation(observation, self.max_action))
        return q_f_sa

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        self.avg_value = 0.0
        self.alpha_w_f = agent_info.get("alpha_w_f", 0.1)
        self.eta_f = agent_info.get("eta_f", 1)
        self.alpha_r_f = self.eta_f * self.alpha_w_f

        self.model = Sequential(
    [
        Dense(4, activation="relu", name="input"),
        Dense(16, activation="relu", name="hidden1"),
        Dense(2, name="output"),
    ]
        )
        self.model.compile(optimizer='sgd', loss='mse')

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the environment's step based on where
                the agent ended up after the last step
        Returns:
            (integer) The action the agent takes given this observation.

        Note: the step size parameters are separate for the value function and the reward rate in the code,
                but will be assigned the same value in the agent parameters agent_info
        """



        delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
        self.weights += self.alpha_w * delta * self.past_state
        # self.avg_reward += self.beta * (reward - self.avg_reward)
        self.avg_reward += self.alpha_r * delta
        delta_f = self.get_value(self.past_state) - self.avg_value + \
                  self.max_action_value_f(observation) - self.get_value_f(self.past_state)
        self.weights_f += self.alpha_w_f * delta_f * self.past_state
        self.avg_value += self.alpha_r_f * delta_f

        action = self.choose_action(observation)
        state = self.get_representation(observation, action)
        self.past_state = state
        self.past_action = action

        return self.past_action


# Tests


def test_DiffQ():
    agent = DifferentialQlearningAgent({'num_states': 3, 'num_actions': 3})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5})
    observation = np.array([1, 0, 1])
    action = agent.agent_start(observation)
    print(agent.past_state, action)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.past_state, agent.past_action)


if __name__ == '__main__':
    test_DiffQ()
