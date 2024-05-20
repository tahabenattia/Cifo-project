from random import randint
import numpy as np


def binary_mutation(individual_array):
    """Apply binary mutation to an individual's image array.

    Args:
        individual_array (np.ndarray): Individual's image array.

    Returns:
        np.ndarray: Mutated image array.
    """
    mut_index = randint(0, len(individual_array) - 1)
    individual_array[mut_index] = ~individual_array[mut_index]
    return individual_array
