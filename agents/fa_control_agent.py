from agents.base_agent import BaseAgent
import numpy as np
from utils.helpers import get_weights_from_npy, argmax
from abc import ABCMeta, abstractmethod

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

        self.learning_params = None
        self.q_params = None

        self.rand_generator = None
        self.choose_action = None  # the policy (e-greedy/greedy/random)

        self.epsilon = None
        self.avg_reward = None
        self.avg_value = None
        self.q_s = None

        self.past_action = None
        self.past_state = None
        self.timestep = None

    def choose_action_egreedy(self):
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

    def choose_action_greedy(self):
        """returns an action using a greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned greedy policy
        """
        return argmax(self.rand_generator, self.q_s)

    def choose_action_random(self):
        """returns a random action indifferent to the current action-value function.
        Args:
        Returns:
            (Integer) The action taken
        """
        return self.rand_generator.choice(self.num_actions)

    def pick_policy(self, agent_info):
        """returns the method that'll pick num_actions based on the argument"""
        policy_type = agent_info.get('policy_type', 'egreedy')
        if policy_type == 'random':
            return self.choose_action_random
        elif policy_type == 'greedy':
            return self.choose_action_greedy
        elif policy_type == 'egreedy':
            self.epsilon = agent_info.get('epsilon', 0.1)
            return self.choose_action_egreedy

    def get_q_s(self, representation):
        """returns an array of action values at the state representation
        Args:
            representation : ndarray
        Returns:
            q(s) : ndarray q_s
        """
        raise NotImplementedError

    def max_action_value(self):
        """returns the action corresponding to the maximum action value for the given observation"""
        return self.q_s[argmax(self.rand_generator, self.q_s)]

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""
        self.choose_action = self.pick_policy(agent_info)

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

        self.q_s = self.get_q_s(observation)
        self.past_action = self.choose_action(observation)
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

