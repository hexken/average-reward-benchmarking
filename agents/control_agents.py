import torch
import numpy as np

from agents.er_buffer import Experience
from agents.fa_base_agents import MLPBaseAgent


class DifferentialQlearningAgent(MLPBaseAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    def __init__(self, num_actions):
        super(DifferentialQlearningAgent, self).__init__(num_actions)
        self.avg_reward_estimate = None
        self.eta = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.avg_reward_estimate = 0.0
        self.eta = agent_info['eta']

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the environment's step based on where
                the agent ended up after the last step
        Returns:
            (torch.Tensor) The action the agent takes given this observation.

        Note: the step size parameters are separate for the value function and the reward rate in the code,
                but will be assigned the same value in the agent parameters agent_info
        """

        # for now we'll keep ERbuffer and model both on device
        observation = torch.tensor(observation, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        last_action = torch.tensor(self.last_action, device=self.device, dtype=torch.int64)
        last_state = torch.tensor(self.last_state, device=self.device, dtype=torch.float32)

        self.er_buffer.add(last_state, last_action, reward, observation)

        if len(self.er_buffer) >= self.batch_size:
            # The Diff Q-Learning updates, adapted to work with an ER buffer and target network

            # [(exp1),...,(expn)]
            experience_list = self.er_buffer.sample_batch(self.batch_size)

            # Experience(s=(exp1.s,...expn.s), a=(exp1.a,...,expn.a),...)
            experience_batch = Experience(*zip(*experience_list))
            state_batch = torch.stack(experience_batch.state)
            next_state_batch = torch.stack(experience_batch.next_state)
            action_batch = torch.tensor(experience_batch.action).view(-1, 1)

            state_action_values = torch.gather(self.Q_network(state_batch), 1, action_batch).view(-1)
            max_next_state_action_values = self.target_network(next_state_batch).max(dim=1)[0].detach()
            rewards = torch.tensor(experience_batch.reward)

            y = rewards - self.avg_reward_estimate + max_next_state_action_values - state_action_values
            loss = self.loss_fn(y, state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for param in self.Q_network.parameters():
                param.grad.data.clamp_(-1, 1)

            with torch.no_grad():
                self.avg_reward_estimate += torch.mean(self.eta * self.alpha * y)

        if self.time_step % self.steps_per_target_network_update == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())

        self.finalize_step(observation)

        return self.last_action


# TODO not implemented yet, its a duplicate of DQL
class RLearningAgent(MLPBaseAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    def __init__(self, num_actions):
        super(RLearningAgent, self).__init__(num_actions)
        self.avg_reward_estimate = None
        self.eta = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.avg_reward_estimate = 0.0
        self.eta = agent_info['eta']

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the environment's step based on where
                the agent ended up after the last step
        Returns:
            (torch.Tensor) The action the agent takes given this observation.

        Note: the step size parameters are separate for the value function and the reward rate in the code,
                but will be assigned the same value in the agent parameters agent_info
        """

        # for now we'll keep ERbuffer and model both on device
        observation = torch.tensor(observation, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        last_action = torch.tensor(self.last_action, device=self.device, dtype=torch.int64)

        self.er_buffer.add(self.last_state, last_action, reward, observation)

        if len(self.er_buffer) >= self.batch_size:
            # The Diff Q-Learning updates, adapted to work with an ER buffer and target network

            # [(exp1),...,(expn)]
            experience_list = self.er_buffer.sample_batch(self.batch_size)

            # Experience(s=(exp1.s,...expn.s), a=(exp1.a,...,expn.a),...)
            experience_batch = Experience(*zip(*experience_list))
            state_batch = torch.stack(experience_batch.state)
            next_state_batch = torch.stack(experience_batch.next_state)
            action_batch = torch.tensor(experience_batch.action).view(-1, 1)

            state_action_values = torch.gather(self.Q_network(state_batch), 1, action_batch).view(-1)
            max_next_state_action_values = self.target_network(next_state_batch).max(dim=1)[0].detach()
            rewards = torch.tensor(experience_batch.reward)

            y = rewards - self.avg_reward_estimate + max_next_state_action_values - state_action_values
            loss = self.loss_fn(y, state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for param in self.Q_network.parameters():
                param.grad.data.clamp_(-1, 1)

            with torch.no_grad():
                self.avg_reward_estimate += torch.mean(self.eta * self.alpha * y)

        if self.time_step % self.steps_per_target_network_update == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())

        self.finalize_step(observation)

        return self.last_action


# TODO not implemented yet, its a duplicate of DQL
class RVIQLearningAgent(MLPBaseAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    def __init__(self, num_actions):
        super(RVIQLearningAgent, self).__init__(num_actions)
        self.avg_reward_estimate = None
        self.eta = None

    def agent_init(self, agent_info):
        super().agent_init(agent_info)
        self.avg_reward_estimate = 0.0
        self.eta = agent_info['eta']

    def agent_step(self, reward, observation):
        """A step taken by the agent.
        Performs the Direct RL step, chooses the next action.
        Args:
            reward (float): the reward received for taking the last action taken
            observation : ndarray
                the state observation from the environment's step based on where
                the agent ended up after the last step
        Returns:
            (torch.Tensor) The action the agent takes given this observation.

        Note: the step size parameters are separate for the value function and the reward rate in the code,
                but will be assigned the same value in the agent parameters agent_info
        """

        # for now we'll keep ERbuffer and model both on device
        observation = torch.tensor(observation, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        last_action = torch.tensor(self.last_action, device=self.device, dtype=torch.int64)

        self.er_buffer.add(self.last_state, last_action, reward, observation)

        if len(self.er_buffer) >= self.batch_size:
            # The Diff Q-Learning updates, adapted to work with an ER buffer and target network

            # [(exp1),...,(expn)]
            experience_list = self.er_buffer.sample_batch(self.batch_size)

            # Experience(s=(exp1.s,...expn.s), a=(exp1.a,...,expn.a),...)
            experience_batch = Experience(*zip(*experience_list))
            state_batch = torch.stack(experience_batch.state)
            next_state_batch = torch.stack(experience_batch.next_state)
            action_batch = torch.tensor(experience_batch.action).view(-1, 1)

            state_action_values = torch.gather(self.Q_network(state_batch), 1, action_batch).view(-1)
            max_next_state_action_values = self.target_network(next_state_batch).max(dim=1)[0].detach()
            rewards = torch.tensor(experience_batch.reward)

            y = rewards - self.avg_reward_estimate + max_next_state_action_values - state_action_values
            loss = self.loss_fn(y, state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            for param in self.Q_network.parameters():
                param.grad.data.clamp_(-1, 1)

            with torch.no_grad():
                self.avg_reward_estimate += torch.mean(self.eta * self.alpha * y)

        if self.time_step % self.steps_per_target_network_update == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())

        self.finalize_step(observation)

        return self.last_action
