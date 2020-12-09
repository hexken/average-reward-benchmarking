import glob
import numpy as np
import os


# TODO this might be slow compared to a torch specific method
def argmax(rand_generator, x):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(x)):
        # if a value in q_values is greater than the highest value update top and reset ties to zero
        # if a value is equal to top value add the index to ties
        # return a random selection from ties.
        # YOUR CODE HERE
        if x[i] > top_value:
            top_value = x[i]
            ties = [i]
        elif x[i] == top_value:
            ties.append(i)
    return rand_generator.choice(ties)


def validate_output_folder(path):
    """checks if folder exists. If not, creates it and returns its name"""
    if path[-1] != '/':
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


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
