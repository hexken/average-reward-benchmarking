import numpy as np
from abc import ABCMeta, abstractmethod

from agents.base_agent import BaseAgent

# tensorflow imports
import tensorflow as tf
from tensorflow.keras import layers


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
        self.choose_action = None  # the policy (e-greedy/greedy/random)

        self.epsilon = None
        self.avg_reward = None
        self.avg_value = None
        self.q_s = None

        self.past_action = None
        self.past_state = None
        self.timestep = None

    def choose_action_egreedy(self, observation):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned epsilon-greedy policy
        """

        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(self.num_actions)
        else:
            q_s = self.get_one_value(observation)
            action = self.rand_generator.choice(np.argwhere(q_s == np.amax(q_s)).flatten())

        return action

    def choose_action_greedy(self, observation):
        """returns an action using a greedy policy w.r.t. the current action-value function.
        Args:
            observation (List)
        Returns:
            (Integer) The action taken w.r.t. the aforementioned greedy policy
        """
        q_s = self.get_one_value(observation)
        action = self.rand_generator.choice(np.argwhere(q_s == np.amax(q_s)).flatten())
        
        return action

    def choose_action_random(self, observation):
        """returns a random action indifferent to the current action-value function.
        Args:
        Returns:
            (Integer) The action taken
        """
        return self.rand_generator.choice(self.num_actions)

    def pick_policy(self, policy_type):
        """returns the method that'll pick num_actions based on the argument"""
        if policy_type == 'random':
            return self.choose_action_random
        elif policy_type == 'greedy':
            return self.choose_action_greedy
        elif policy_type == 'egreedy':
            return self.choose_action_egreedy

    def get_one_value(self, observation):
        """returns an array of action values at the state representation
        Args:
            observation : ndarray
        Returns:
            q(s) : ndarray q_s
        """
        model_obs = tf.convert_to_tensor(observation)
        model_obs = tf.expand_dims(model_obs, 0)
        return self.model(model_obs, training=False)[0].numpy()


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


class MLPControlAgent(FAControlAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    __metaclass__ = ABCMeta

    def __init__(self, config):
        super().__init__(config)

    def get_batch_value(self, observation):
        """returns action value vector q:S->R^{|A|}
        Args:
            observation: ndarray
        Returns:
        """
        return self.model(observation)
    
    def get_batch_target_value(self, observation):
        """returns action value vector q:S->R^{|A|}
        Args:
            observation: ndarray
        Returns:
        """
        return self.model_target.predict(observation)

    def max_action_value(self, observation):
        """
        returns the higher-order action value corresponding to the
        maximum lower-order action value for the given observation.
        Note: this is not max_a q_f(s,a)
        """
        q_s = self.get_batch_target_value(observation)
        return tf.reduce_max(q_s, axis=1)

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts."""

        super().agent_init(agent_info)

        # assert "num_actions" in agent_info
        # self.num_actions = agent_info.get("num_actions", 4)
        # assert "num_states" in agent_info
        # self.num_states = agent_info["num_states"]
        self.alpha_w = agent_info.get("alpha_w", 0.1)
        self.eta = agent_info.get("eta", 1)
        # self.alpha_r = agent_info.get("alpha_r", self.alpha_w)
        self.alpha_r = self.eta * self.alpha_w
        self.value_init = agent_info.get("value_init", 0)
        self.avg_reward_init = agent_info.get("avg_reward_init", 0)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.choose_action = self.pick_policy(agent_info.get("policy_type", "egreedy"))

        self.avg_reward = 0.0 + self.avg_reward_init
        
        # new for MLP
        self.batch_size = 512 # 128
        self.hidden_layers = [16,16] # [32,32]
        
        self.model = self.create_model()
        self.model_target = self.create_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_w,
                                                  clipnorm=1.0) # could change clipnorm?
        self.loss_function = tf.keras.losses.Huber()
        self.update_after_actions = 1 # 8
        self.update_target_network = 10000 # 1600
        self.update_avg_reward = 10
        
        # frame counts
        self.frame_count = 0
        self.epsilon_random_frames = 0
        
        # epsilon decay
        self.epsilon = 1.0 # 1.0
        self.epsilon_min = 0.1 # 0.1
        self.epsilon_max = 1.0 # 1.0
        self.epsilon_interval = self.epsilon_max - self.epsilon_min
        
        # experience replay
        self.init_experience_replay()
        
        # to test q-learning vs r-learning
        self.gamma = 1.0
        self.use_avg_reward = True
        
    
    def create_model(self):
        inputs = layers.Input(shape=(self.num_states), dtype=tf.float64)
        
        x = inputs
        for layer_size in self.hidden_layers:
            x = layers.Dense(layer_size, activation='relu',
                             kernel_initializer='glorot_uniform',
                             dtype=tf.float64)(x)

        outputs = layers.Dense(self.num_actions, activation='linear',
                               kernel_initializer='glorot_uniform',
                               dtype=tf.float64)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    def init_experience_replay(self):
      
        # experience replay buffers
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.update_rho_history = []
        
        # random action frames
        self.epsilon_random_frames = 20000 # 4000
        # greedy action frames (for epsilon decay)
        self.epsilon_greedy_frames = 100000 # 40000
        # maximum replay length
        self.max_memory_length = 100000 # 80000
    
    def decay_epsilon(self):
        if self.frame_count > self.epsilon_random_frames:
            self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
            self.epsilon = max(self.epsilon, self.epsilon_min)