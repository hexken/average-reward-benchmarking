import numpy as np
from utils.helpers import get_weights_from_npy, argmax
import agents.fa_control_agent


class DifferentialQlearningAgent(agents.FAControlAgent):
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
        self.max_action = None

    def get_qs(self, representation):
        """returns the higher-order action value linear in the representation and the weights
        Args:
            representation : ndarray
        Returns:
        """
        return np.dot(representation, self.weights_f)

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

        self.weights_f = np.zeros((self.num_states * self.num_actions))
        self.avg_value = 0.0
        self.alpha_w_f = agent_info.get("alpha_w_f", 0.1)
        self.eta_f = agent_info.get("eta_f", 1)
        self.alpha_r_f = self.eta_f * self.alpha_w_f

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


# class RVIQlearningAgent(LFAControlAgent):
#     """
#     Implements a version of the RVI Q-learning algorithm (Abounadi et al., 2001)
#     with f as the value of a fixed state-action pair.
#     self.avg_reward is used as a non-learned parameter in place of f in the algorithm.
#     """
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.f_type = None
#         self.reference_s = None  # raw state index
#         self.reference_a = None  # raw action index
#         self.reference_sa = None  # representation corresponding to this s-a pair
#
#     def agent_init(self, agent_info):
#         super().agent_init(agent_info)
#
#         self.f_type = agent_info.get('f_type', 'reference_sa')
#         if self.f_type == 'reference_sa':
#             self.reference_s = agent_info.get('reference_state', 0)
#             self.reference_a = agent_info.get('reference_action', 0)
#
#     def agent_start(self, observation):
#         action = super().agent_start(observation)
#         ref_state = np.zeros(observation.shape)
#         ref_state[self.reference_s] = 1
#         self.reference_sa = self.get_representation(ref_state, self.reference_a)
#         return action
#
#     def agent_step(self, reward, observation):
#         """A step taken by the agent.
#         Performs the Direct RL step, chooses the next action.
#         Args:
#             reward (float): the reward received for taking the last action taken
#             observation : ndarray
#                 the state observation from the environment's step based on where
#                 the agent ended up after the last step
#         Returns:
#             (integer) The action the agent takes given this observation.
#         """
#         self.avg_reward = self.get_value(self.reference_sa)
#         delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
#         self.weights += self.alpha_w * delta * self.past_state
#
#         action = self.choose_action(observation)
#         state = self.get_representation(observation, action)
#         self.past_state = state
#         self.past_action = action
#
#         return self.past_action


# class RlearningAgent(LFAControlAgent):
#     """
#     Implements the R-learning algorithm by Schwartz (1993).
#     """
#
#     def __init__(self, config):
#         super().__init__(config)
#         # self.past_max_actions holds all the greedy actions for the previous observation
#         self.past_max_actions = None
#
#     def agent_start(self, observation):
#         action = super().agent_start(observation)
#         self.past_max_actions = self.get_max_actions(observation)
#         return action
#
#     def get_max_actions(self, observation):
#         q_s = np.array([self.get_value(self.get_representation(observation, a)) for a in self.actions])
#         max_actions = np.argwhere(q_s == np.amax(q_s))
#         return max_actions
#
#     def agent_step(self, reward, observation):
#         """A step taken by the agent.
#         Performs the Direct RL step, chooses the next action.
#         Args:
#             reward (float): the reward received for taking the last action taken
#             observation : ndarray
#                 the state observation from the environment's step based on where
#                 the agent ended up after the last step
#         Returns:
#             (integer) The action the agent takes given this observation.
#         """
#         delta = reward - self.avg_reward + self.max_action_value(observation) - self.get_value(self.past_state)
#         self.weights += self.alpha_w * delta * self.past_state
#         if self.past_action in self.past_max_actions:
#             self.avg_reward += self.alpha_r * delta
#
#         action = self.choose_action(observation)
#         state = self.get_representation(observation, action)
#         self.past_max_actions = self.get_max_actions(observation)
#         self.past_state = state
#         self.past_action = action
#
#         return self.past_action


# Tests


def test_DiffQ():
    agent = DifferentialQlearningAgent_v1({'num_states': 3, 'num_actions': 3})
    agent.agent_init({'random_seed': 32, 'epsilon': 0.5})
    agent.weights = np.array([1, 2, 1, -2, 0, -1, 0, 1, 1]) * 1.0
    observation = np.array([1, 0, 1])
    action = agent.agent_start(observation)
    print(agent.past_state, action)

    for i in range(3):
        agent.agent_step(1, observation)
        print(agent.past_state, agent.past_action)


# def test_RVIQ():
#     agent = RVIQlearningAgent({'num_states': 3, 'num_actions': 3})
#     agent.agent_init({'random_seed': 32, 'epsilon': 0.5})
#     agent.weights = np.array([1, 2, 1, -2, 0, -1, 0, 1, 1]) * 1.0
#     observation = np.array([1, 0, 1])
#     action = agent.agent_start(observation)
#     print(agent.reference_sa)
#     print(agent.past_state, action)
#
#     for i in range(3):
#         agent.agent_step(1, observation)
#         print(agent.past_state, agent.past_action)


# def test_R():
#     agent = RlearningAgent({'num_states': 3, 'num_actions': 3})
#     agent.agent_init({'random_seed': 32, 'epsilon': 0.5})
#     agent.weights = np.array([1, 2, 1, -2, 0, -1, 0, 1, 1]) * 1.0
#     observation = np.array([1, 0, 1])
#     action = agent.agent_start(observation)
#     print(agent.past_state, action)
#
#     for i in range(3):
#         agent.agent_step(1, observation)
#         print(agent.past_state, agent.past_action)
#         print(agent.weights)


if __name__ == '__main__':
    test_DiffQ()
    # test_RVIQ()
    # test_R()
