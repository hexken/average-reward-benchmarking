import numpy as np
from tqdm import tqdm
from utils.rl_glue import RLGlue


# Runs a control experiment in which the training steps are *not* interleaved with evaluation periods.
# Typically used when the behaviour policy is very similar to the target policy, say when epsilon=0.1.
# computes the average reward every avg_every_n_steps
def run_exp_learning_control_no_eval(env, agent, config):
    num_runs = config['exp_parameters']['num_runs']
    max_steps = config['exp_parameters']['num_max_steps']
    save_model_params = config['exp_parameters'].get('save_model_params', True)

    env_info = config['env_parameters']
    agent_info = config['agent_parameters']

    print('Env: %s, Agent: %s' % (env.__name__, agent.__name__))

    log_data = {}
    # time_steps start at 1!
    rewards = np.zeros((num_runs, max_steps + 1))
    avg_reward_estimates = np.zeros((num_runs, max_steps + 1))

    for run in range(num_runs):

        # TODO check to make sure random seeds are actually consistantly set
        agent_info['seed'] = run
        env_info['seed'] = run

        rl_glue = RLGlue(env, agent)
        rl_glue.rl_init(agent_info, env_info)
        rl_glue.rl_start()

        print(f'run {run + 1} of {num_runs}')
        eval_idx = 0
        for time_step in tqdm(range(1, max_steps + 1)):
            reward, obs, action, _ = rl_glue.rl_step()
            rewards[run][time_step] = reward
            avg_reward_estimates[run][time_step] = rl_glue.agent.avg_reward_estimate

    tqdm.write('AvgReward_total\t\t= %f' % (np.mean(rewards)))
    tqdm.write('AvgReward_lasthalf\t= %f' % np.mean(rewards[:, rewards.shape[1] // 2:]))
    tqdm.write('AgentEstAvgReward_total\t= %f' % np.mean(avg_reward_estimates))
    tqdm.write(
        'AgentEstAvgReward_lasthalf\t= %f\n' % np.mean(avg_reward_estimates[:, avg_reward_estimates.shape[1] // 2:]))
    log_data['rewards_all'] = rewards
    log_data['avg_reward_estimates'] = avg_reward_estimates
    if save_model_params:
        log_data['model_params'] = rl_glue.agent.get_model_params()

    return log_data
