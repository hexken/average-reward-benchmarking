import glob
import numpy as np
import os


def decay_epsilon(decay_period, time_step, warmup_steps, epsilon_start, epsilon_end):
    """
    FROM https://github.com/google/dopamine/blob/master/dopamine/agents/dqn/dqn_agent.py,
    with slight modifications.
    Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
      Begin at 1. until warmup_steps steps have been taken; then
      Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
      Use epsilon from there on.
    Args:
      decay_period: float, the period over which epsilon is decayed.
      step: int, the number of training steps completed so far.
      warmup_steps: int, the number of steps taken before epsilon is decayed.
      epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
      A float, the current epsilon value computed according to the schedule.
    """
    epsilon_interval = epsilon_start - epsilon_end
    steps_left = decay_period + warmup_steps - time_step
    bonus = epsilon_interval * steps_left / decay_period
    bonus = np.clip(bonus, 0., epsilon_interval)
    return epsilon_end + bonus


def validate_output_folder(path):
    """checks if folder exists. If not, creates it and returns its name"""
    if path[-1] != '/':
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def argmax(rand_gen, x):
    """ returns i uniformly at random from { i | x[i] == max(x) } """
    return rand_gen.choice(np.where(x == np.amax(x)))


def all_files_with_prefix_and_suffix(location, prefix, suffix):
    """returns a list of all files in the 'location' starting with the given prefix"""
    if location[-1] != '/':
        location += '/'
    files = glob.glob(location + prefix + '*' + suffix)

    return files


def get_weights_from_npy(filename):
    data = np.load(filename, allow_pickle=True).item()
    weights = data['weights']

    return weights


if __name__ == "__main__":
    print(all_files_with_prefix_and_suffix("../results/",
                                           "RiverSwim_DifferentialQlearningAgent_sensitivity_fine_",
                                           ".npy"))
