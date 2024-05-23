from random import randint
import numpy as np


# def binary_mutation(individual_array):
#     """Apply binary mutation to an individual's image array.

#     Args:
#         individual_array (np.ndarray): Individual's image array.

#     Returns:
#         np.ndarray: Mutated image array.
#     """
#     mut_index = randint(0, len(individual_array) - 1)
#     individual_array[mut_index] = ~individual_array[mut_index]
#     return individual_array




# def swap_mutation(individual_array):
#     """Apply swap mutation to an individual's binary array.

#     Args:
#         individual_array (list): Individual's binary array.

#     Returns:
#         list: Mutated binary array.
#     """
#     # Select two random indices for swapping
#     index1 = randint(0, len(individual_array) - 1)
#     index2 = randint(0, len(individual_array) - 1)

#     # Swap the values at the selected indices
#     individual_array[index1], individual_array[index2] = individual_array[index2], individual_array[index1]

#     return individual_array




def binary_mutation(individual_array, mutation_rate):
    """Apply binary mutation to an individual's image array.

    Args:
        individual_array (np.ndarray): Individual's image array.
        mutation_rate (float): Probability of mutation for each gene.

    Returns:
        np.ndarray: Mutated image array.
    """
    mutated_array = np.copy(individual_array)
    for i in range(len(mutated_array)):
        if np.random.rand() < mutation_rate:
            mutated_array[i] = ~mutated_array[i]  # Flip the bit
    return mutated_array



def swap_mutation(individual_array, mutation_rate):
    """Apply swap mutation to an individual's binary array.

    Args:
        individual_array (list): Individual's binary array.
        mutation_rate (float): Probability of mutation for each gene.

    Returns:
        list: Mutated binary array.
    """
    mutated_array = individual_array.copy()
    if np.random.rand() < mutation_rate:
        index1 = randint(0, len(mutated_array) - 1)
        index2 = randint(0, len(mutated_array) - 1)
        mutated_array[index1], mutated_array[index2] = mutated_array[index2], mutated_array[index1]
    return mutated_array