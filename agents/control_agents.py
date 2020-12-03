import torch
import numpy as np
from agents.er_buffer import Experience

from agents.fa_base_agents import MLPBaseAgent


class DifferentialQlearningAgent(MLPBaseAgent):
    """
    Implements the version of newly-proposed Differential Q-learning algorithm
    in which centering does not affect the learning process.
    """

    def __init__(self, agent_info):
        super().__init__(agent_info)

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
        self.er_buffer.add(self.past_state, self.past_action, reward, observation)

        # The Diff Q-Learning updates, adapted to work with an ER buffer and target netork
        if self.time_step % self.steps_per_target_network_update == 0 \
                and len(self.er_buffer) >= self.batch_size:

            # optimize

            # [(exp1),...,(expn)]
            experience_list = self.er_buffer.sample_batch(self.batch_size)

            # Experience(s=(exp1.s,...expn.s), a=(exp1.a,...,expn.a),...)
            experience_batch = Experience(*zip(*experience_list))

            state_batch = experience_batch.state
            action_batch = experience_batch.action
            reward_batch = experience_batch.reward
            next_state_batch = experience_batch.next_state

            # gather using action_batch to index into state value vector batch
            state_action_values = self.policy_network(state_batch).gather(1, action_batch)
            next_state_values = self.target_network(next_state_batch).max(1)[0].detach()

            target_batch = reward_batch - next_state_values  + reward_batch

            loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

            self.optimizer.zero_grad()

            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()



        # update params

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
