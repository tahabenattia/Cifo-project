from random import randint
import numpy as np
from PIL import Image

# def single_point_xo(parent1, parent2):
#     """Perform single-point crossover.

#     Args:
#         parent1 (np.ndarray): First parent image array.
#         parent2 (np.ndarray): Second parent image array.

#     Returns:
#         tuple: Two offspring image arrays.
#     """
#     height, width = parent1.shape
#     xo_point = randint(1, width - 1)
#     offspring1 = np.concatenate((parent1[:, :xo_point], parent2[:, xo_point:]), axis=1)
#     offspring2 = np.concatenate((parent2[:, :xo_point], parent1[:, xo_point:]), axis=1)
#     return offspring1, offspring2



# def cycle_xo(p1, p2):
#     """Implementation of cycle crossover.

#     Args:
#         p1 (Individual): First parent for crossover.
#         p2 (Individual): Second parent for crossover.

#     Returns:
#         Individuals: Two offspring, resulting from the crossover.
#     """
#     # Offspring placeholders - None values make it easy to debug for errors
#     offspring1 = [None] * len(p1)
#     offspring2 = [None] * len(p1)

#     # While there are still None values in offspring, get the first index of
#     # None and start a "cycle" according to the cycle crossover method
#     while None in offspring1:
#         index = offspring1.index(None)
#         val1 = p1[index]
#         val2 = p2[index]

#         # copy the cycle elements
#         while val1 != val2:
#             offspring1[index] = p1[index]
#             offspring2[index] = p2[index]
#             val2 = p2[index]
#             index = p1.index(val2)

#         # copy the rest
#         for element in offspring1:
#             if element is None:
#                 index = offspring1.index(None)
#                 if offspring1[index] is None:
#                     offspring1[index] = p2[index]
#                     offspring2[index] = p1[index]

#     return offspring1, offspring2


def single_point_xo(parent1, parent2):
    """Perform single-point crossover.

    Args:
        parent1 (np.ndarray): First parent image array.
        parent2 (np.ndarray): Second parent image array.

    Returns:
        tuple: Two offspring image arrays.
    """
    height, width = parent1.shape
    xo_point = randint(1, width - 1)
    
    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)
    
    offspring1[:, xo_point:] = parent2[:, xo_point:]
    offspring2[:, xo_point:] = parent1[:, xo_point:]
    
    return offspring1, offspring2


def cycle_xo(p1, p2):
    """Perform cycle crossover.

    Args:
        p1 (np.ndarray): First parent array.
        p2 (np.ndarray): Second parent array.

    Returns:
        tuple: Two offspring arrays.
    """
    size = len(p1)
    offspring1 = np.full(size, None)
    offspring2 = np.full(size, None)

    while None in offspring1:
        # Start cycle
        cycle_start = np.where(offspring1 == None)[0][0]
        current_index = cycle_start
        while True:
            offspring1[current_index] = p1[current_index]
            offspring2[current_index] = p2[current_index]
            next_index = np.where(p1 == p2[current_index])[0][0]
            if next_index == cycle_start:
                break
            current_index = next_index

    # Fill the rest with other parent's genes
    for i in range(size):
        if offspring1[i] is None:
            offspring1[i] = p2[i]
            offspring2[i] = p1[i]

    return offspring1, offspring2