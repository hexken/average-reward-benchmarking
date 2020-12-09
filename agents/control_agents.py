import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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

        self.model_f = self.create_model()
        self.model_target_f = self.create_model()

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



class RlearningAgent(MLPControlAgent):
    """
    Implements the R-learning algorithm by Schwartz (1993).
    """

    def __init__(self, config):
        super().__init__(config)
        # self.past_max_actions holds all the greedy actions for the previous observation
        self.past_max_actions = None

    def agent_start(self, observation):
        action = super().agent_start(observation)
        self.past_max_actions = self.get_max_actions(observation)
        return action

    def get_max_actions(self, observation):
        q_s = self.get_one_value(observation)
        max_actions = np.argwhere(q_s == np.amax(q_s))
        return max_actions

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
        """
        self.frame_count += 1
        
        # reward clipping?
        #reward = np.sign(reward)
        
        # test red circle
        #reward = -1 * reward
        
        '''
        if reward != 0:
            print(reward)
        '''
        
        self.action_history.append(self.past_action)
        self.state_history.append(self.past_state)
        self.state_next_history.append(observation)
        self.rewards_history.append(reward)
        self.update_rho_history.append(True)
        
        action = self.choose_action(observation)
        self.past_max_actions = self.get_max_actions(observation)
        self.past_state = observation
        self.past_action = action
        
        self.decay_epsilon()
                
        if self.frame_count % self.update_after_actions == 0 and \
            len(self.rewards_history) > self.batch_size:
                # which training samples to use
                indices = np.random.choice(range(len(self.rewards_history)),
                                           size=self.batch_size)
                
                # rho rescale to favor recent rho updates
                #rho_rescale = (indices / len(self.rewards_history))**2
                #rho_rescale = (indices / self.max_memory_length)**2
                
                # sample from replay buffer
                state_sample = np.array([self.state_history[i] for i in indices])
                state_next_sample = np.array([self.state_next_history[i] for i in indices])
                rewards_sample = np.array([self.rewards_history[i] for i in indices])
                action_sample = [self.action_history[i] for i in indices]
                #update_rho_samples = np.array([self.update_rho_history[i] for i in indices])
                
                target = rewards_sample - self.avg_reward + \
                    self.gamma * self.max_action_value(state_next_sample)
                
                masks = tf.one_hot(action_sample, self.num_actions, dtype=tf.float64)
                
                with tf.GradientTape() as tape:
                    q_values = self.get_batch_value(state_sample)
                    q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                    loss = self.loss_function(target, q_action)
                
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
                # previously had 'update average reward' here
                #rho_update_array = update_rho_samples * rho_rescale * (target - q_action)
                #self.avg_reward += self.alpha_r * tf.reduce_sum(rho_update_array)
        
        #'''
        if self.frame_count % self.update_avg_reward == 0 and self.use_avg_reward == True:
            ### average reward batch update
            # sample previous 'update_target_network' samples from replay buffer
            state_sample = np.array(self.state_history[-self.update_avg_reward:])
            state_next_sample = np.array(self.state_next_history[-self.update_avg_reward:])
            rewards_sample = np.array(self.rewards_history[-self.update_avg_reward:])
            action_sample = self.action_history[-self.update_avg_reward:]
            update_rho_samples = np.array(self.update_rho_history[-self.update_avg_reward:])
            
            target = rewards_sample - self.avg_reward + \
                self.max_action_value(state_next_sample)
            
            masks = tf.one_hot(action_sample, self.num_actions, dtype=tf.float64)
            
            q_values = self.get_batch_value(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            
            # update average reward
            rho_update_array = update_rho_samples * (target - q_action)
            self.avg_reward += self.alpha_r * tf.reduce_sum(rho_update_array).numpy()
        #'''
        
        if self.frame_count % self.update_target_network == 0:
            print(f'UPDATE TARGET. frame: {self.frame_count}')
            print(loss)
            print(self.epsilon)
            print(f'AR: {self.avg_reward}')
            self.model_target.set_weights(self.model.get_weights())
        
        if len(self.rewards_history) > self.max_memory_length:
            del self.action_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.rewards_history[:1]
            del self.update_rho_history[:1]

        return self.past_action

