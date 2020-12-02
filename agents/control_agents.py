import numpy as np

from utils.helpers import argmax
from agents.fa_base_agents import MLPBaseAgent
from agents.er_buffer import ERBuffer


class DifferentialQlearningAgent(MLPBaseAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    def __init__(self, config):
        super().__init__(config)

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

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
