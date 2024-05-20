from random import randint
import numpy as np
from PIL import Image

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
    offspring1 = np.concatenate((parent1[:, :xo_point], parent2[:, xo_point:]), axis=1)
    offspring2 = np.concatenate((parent2[:, :xo_point], parent1[:, xo_point:]), axis=1)
    return offspring1, offspring2






def cycle_xo(p1, p2):
    """Implementation of cycle crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    # While there are still None values in offspring, get the first index of
    # None and start a "cycle" according to the cycle crossover method
    while None in offspring1:
        index = offspring1.index(None)
        val1 = p1[index]
        val2 = p2[index]

        # copy the cycle elements
        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.index(val2)

        # copy the rest
        for element in offspring1:
            if element is None:
                index = offspring1.index(None)
                if offspring1[index] is None:
                    offspring1[index] = p2[index]
                    offspring2[index] = p1[index]

    return offspring1, offspring2