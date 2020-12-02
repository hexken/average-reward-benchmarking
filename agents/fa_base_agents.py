from abc import ABCMeta, abstractmethod

import numpy as np
import torch.nn as nn

from utils.helpers import argmax
from agents.base_agent import BaseAgent
from agents.function_approximators import MLP
from agents.er_buffer import ERBuffer


class FABaseAgent(BaseAgent):
    """
    A generic class that is re-used in the implementation of all sorts of control algorithms with FA.
    Only agent_step and get_q_s need to be implemented in the child classes.
    Assumes action set {0,1,2,3,..,self.num_actions}
    """

    __metaclass__ = ABCMeta

    def __init__(self, agent_info):
        super().__init__()
        self.num_actions = agent_info['num_actions']
        self.num_states = agent_info['num_states']  # this could also be the size of the observation vector

        self.rand_generator = None
        self.policy = None  # the policy (e-greedy/greedy/random)

        self.epsilon = None
        self.avg_reward = None
        self.avg_value = None
        self.time_step = None

        self.Q_current = None  # probably convenient to store this so we don't have to keep evaluating our NN
        self.past_action = None
        self.past_state = None

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
            action = argmax(self.rand_generator, self.Q_current)

        return action

    def greedy_policy(self):
        """returns an action using a greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned greedy policy
        """
        return argmax(self.rand_generator, self.Q_current)

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
            return self.greedy_policy
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
        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 47))
        self.avg_reward = 0.0
        self.time_step = 0  # for debugging

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
        self.time_step += 1

        return self.past_action

    def finalize_step(self, observation):
        """ Finalizes a control agents step. Call after parameter updates."""
        action = self.policy(observation)
        self.past_state = observation
        self.past_action = action
        self.time_step += 1

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


class MLPBaseAgent(FABaseAgent):
    """
    Implements an MLP agent that takes state vector as input and outputs an action-value vector.
    Uses an er buffer and target network.
    """

    __metaclass__ = ABCMeta

    def __init__(self, config):
        super().__init__(config)
        self.policy_network = None
        self.target_network = None
        self.er_buffer = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        assert "er_buffer_capacity" in agent_info
        assert "steps_per_target_network_update" in agent_info
        assert "hidden_layer_sizes" in agent_info

        self.policy_network = MLP(agent_info)
        self.target_network = MLP(agent_info)
        self.er_buffer = ERBuffer(self.rand_generator, agent_info['er_buffer_capacity'])
