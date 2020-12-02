import numpy as np
from utils.helpers import argmax
from abc import ABCMeta, abstractmethod

from agents.base_agent import BaseAgent

class FAControlAgent(BaseAgent):
    """
    A generic class that is re-used in the implementation of all sorts of control algorithms with FA.
    Only agent_step and get_q_s need to be implemented in the child classes.
    Assumes action set {0,1,2,3,..,self.num_actions}
    """

    __metaclass__ = ABCMeta

    def __init__(self, config):
        super().__init__()
        self.num_actions = config['num_actions']
        self.num_states = config['num_states']  # this could also be the size of the observation vector

        self.rand_generator = None
        self.policy = None  # the policy (e-greedy/greedy/random)

        self.epsilon = None
        self.avg_reward = None
        self.avg_value = None
        self.q_s = None # probably convenient to store this so we don't have to keep evaluating our NN

        self.past_action = None
        self.past_state = None
        self.timestep = None

    def egreedy_policy(self):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.num_actions)
        else:
            action = argmax(self.rand_generator)

        return action

    def greeedy_policy(self):
        """returns an action using a greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned greedy policy
        """
        return argmax(self.rand_generator, self.q_s)

    def random_policy(self):
        """returns a random action indifferent to the current action-value function.
        Args:
        Returns:
            (Integer) The action taken
        """
        return self.rand_generator.choice(self.num_actions)

    def set_policy(self, agent_info):
        """returns the method that'll pick num_actions based on the argument"""
        policy_type = agent_info.get('policy_type', 'egreedy')
        if policy_type == 'random':
            return self.random_policy
        elif policy_type == 'greedy':
            return self.greeedy_policy
        elif policy_type == 'egreedy':
            self.epsilon = agent_info.get('epsilon', 0.1)
            return self.egreedy_policy
        else:
            raise ValueError(f"'{policy_type}' is not a valid policy.")

    def Q(self, observation):
        """returns an array of action values at the state representation
        Args:
            observation : ndarray
        Returns:
            q(s) : ndarray q_s
        """
        raise NotImplementedError


    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""
        self.policy = self.set_policy(agent_info)

        self.avg_reward = 0.0
        self.avg_value = 0.0

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 47))

        self.timestep = 0  # for debugging

    def agent_start(self, observation):
        """The first method called when the experiment starts,
        called after the environment starts.
            observation (Numpy array): the state observation from the
                environment's env_start function.
        Returns:
            (integer) the first action the agent takes.
        """

        self.Q_current = self.Q(observation)
        self.past_action = self.policy(observation)
        self.past_state = observation
        self.timestep += 1

        return self.past_action

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (integer) The action the agent takes given this observation.
        """
        raise NotImplementedError


    def agent_end(self, reward):
        """Run when the agent terminates.
        A direct-RL update with the final transition. Not applicable for continuing tasks
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass


class MLPControlAgent(FAControlAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    __metaclass__ = ABCMeta

    def __init__(self, config):
        super().__init__(config)

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

        action = self.policy(observation)
        state = self.get_representation(observation, action)
        self.past_state = state
        self.past_action = action

        return self.past_action
