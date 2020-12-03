import numpy as np
import torch
from utils.helpers import argmax, decay_epsilon
from agents.base_agent import BaseAgent
from agents.function_approximators import MLP
from agents.er_buffer import ERBuffer

from abc import ABCMeta, abstractmethod


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
        self.policy_type = None
        self.choose_action = None  # the policy (e-greedy/greedy/random)

        self.alpha = None

        self.epsilon_start = None
        self.warmup_steps = None
        self.epsilon_decay = False
        self.epsilon_end = None

        self.time_step = None

        self.past_action = None
        self.past_state = None

    def egreedy_policy(self):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.
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
        assert "policy_type" in agent_info

        policy_type = agent_info.get('policy_type')
        if policy_type == 'random':
            self.policy_type = policy_type
            return self.random_policy
        elif policy_type == 'greedy':
            self.policy_type = policy_type
            return self.greedy_policy
        elif policy_type == 'egreedy':
            assert "epsilon_start" in agent_info
            assert "epsilon_decay" in agent_info
            self.policy_type = policy_type
            self.epsilon_start = agent_info.get('epsilon_start')
            self.epsilon_decay = agent_info.get('epsilon_decay')
            if self.epsilon_decay:
                assert "epsilon_end" in agent_info
                assert "warmup_steps" in agent_info
                self.epsilon_end = agent_info('epsilon_end')
                self.warmup_steps = agent_info('warmup_steps')
            return self.egreedy_policy
        else:
            raise ValueError(f"'{policy_type}' is not a valid policy.")

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""
        assert 'alpha' in agent_info
        self.choose_action = self.set_policy(agent_info)
        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 47))
        self.alpha = agent_info['alpha']
        self.time_step = 0

    def agent_start(self, observation):
        """The first method called when the experiment starts,
        called after the environment starts.
            observation (Numpy array): the state observation from the
                environment's env_start function.
        Returns:
            (integer) the first action the agent takes.
        """

        self.past_action = self.policy(observation)
        self.past_state = observation
        self.time_step += 1

        return self.past_action

    def finalize_step(self, observation):
        """ Finalizes a control agents step. Call after parameter updates."""
        if self.decay:
            decay_epsilon(self.decay_period, self.time_step, self.warmup_steps, self.epsilon_start, self.epsilon_end)

        action = self.choose_action(observation)
        self.past_state = observation
        self.past_action = action
        self.time_step += 1

    @abstractmethod
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

    def __init__(self, agent_info):
        super().__init__(agent_info)
        self.Q_network = None
        self.target_network = None
        self.steps_per_target_network_update = None

        self.er_buffer = None
        self.batch_size = None

        self.optimizer = None
        self.loss_fn = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)

        assert "er_buffer_capacity" in agent_info
        assert "steps_per_target_network_update" in agent_info
        assert "hidden_layer_sizes" in agent_info
        assert "steps_per_target_network_update" in agent_info
        assert "batch_size" in agent_info

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Using GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU.')
            device = torch.device("cpu")

        self.Q_network = MLP(agent_info).to(device)
        self.target_network = MLP(agent_info).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.steps_per_target_network_update = agent_info['steps_per_target_network_update']
        self.target_network.eval()

        self.er_buffer = ERBuffer(agent_info['er_buffer_capacity'])
        self.batch_size = agent_info['batch_size']

        self.optimizer = torch.optim.RMSprop(self.Q_network.parameters(), lr=self.alpha)
        # TODO might have to tune beta (SmoothL1Loss param) also
        self.loss_fn = torch.nn.SmoothL1Loss()
