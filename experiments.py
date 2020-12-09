import numpy as np
from tqdm import tqdm
from utils.rl_glue import RLGlue


# Runs a control experiment in which the training steps are *not* interleaved with evaluation periods.
# Typically used when the behaviour policy is very similar to the target policy, say when epsilon=0.1.
# computes the average reward every avg_every_n_steps
def run_exp_learning_control_no_eval(env, agent, config):
    num_runs = config['exp_parameters']['num_runs']
    max_steps = config['exp_parameters']['num_max_steps']
    avg_every_n_steps = config['exp_parameters']['avg_every_n_steps']
    save_model_params = config['exp_parameters'].get('save_model_params', True)

    env_info = config['env_parameters']
    agent_info = config['agent_parameters']

    print('Env: %s, Agent: %s' % (env.__name__, agent.__name__))

    log_data = {}
    # time_steps start at 1!
    rewards = np.zeros((num_runs, max_steps + 1))
    avg_reward_estimates = np.zeros((num_runs, max_steps + 1))
    assert max_steps % avg_every_n_steps == 0  # ideally not necessary, but enforcing nonetheless
    # the avg of the avg reward estimates over the previous n steps
    n_step_avg_reward_estimates = np.zeros((num_runs, int(max_steps / avg_every_n_steps) + 1))

    for run in tqdm(range(num_runs)):

        # TODO check to make sure random seeds are actually consistantly set
        agent_info['seed'] = run
        env_info['seed'] = run

        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()

        eval_idx = 0
        for time_step in range(1, max_steps + 1):

            reward, obs, action, _ = rl_glue.rl_step()
            rewards[run][time_step] = reward

            if time_step % avg_every_n_steps == 0:
                avg_reward_estimates[run][eval_idx] = rl_glue.agent.avg_reward_estimate
                eval_idx += 1

    tqdm.write('AvgReward_total\t\t= %f' % (np.mean(rewards)))
    tqdm.write('AvgReward_lasthalf\t= %f\n' % np.mean(rewards[:, rewards.shape[1] // 2:]))
    tqdm.write('AgentEstAvgReward_total\t= %f' % np.mean(n_step_avg_reward_estimates))
    tqdm.write('AgentEstAvgReward_lasthalf\t= %f\n' % np.mean(
        n_step_avg_reward_estimates[:, n_step_avg_reward_estimates.shape[1] // 2:]))
    log_data['rewards_all'] = rewards
    log_data['avg_reward_estimates_all'] = avg_reward_estimates
    if save_model_params:
        log_data['model_params'] = rl_glue.agent.get_model_params()

    return log_data
