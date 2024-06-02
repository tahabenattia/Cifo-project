from random import randint, sample, uniform, random
import numpy as np
from scipy.ndimage import gaussian_filter
from charles import Individual

def two_point_xo(parent1, parent2):
    """Implementation of two-point crossover for HSV images.

    Args:
        parent1 (Individual): First parent for crossover.
        parent2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    parent1_repr = np.array(parent1.representation).reshape((300, 300, 3))
    parent2_repr = np.array(parent2.representation).reshape((300, 300, 3))

    # Randomly select two crossover points
    points = sorted(sample(range(parent1_repr.size // 3), 2))
    points = [p * 3 for p in points]  # Convert points to correspond to 3 channels

    # Convert to lists to use list concatenation
    parent1_repr = parent1_repr.flatten().tolist()
    parent2_repr = parent2_repr.flatten().tolist()

    # Create the offspring by combining segments from both parents
    offspring1 = parent1_repr[:points[0]] + parent2_repr[points[0]:points[1]] + parent1_repr[points[1]:]
    offspring2 = parent2_repr[:points[0]] + parent1_repr[points[0]:points[1]] + parent2_repr[points[1]:]

    # Convert back to numpy arrays and reshape
    offspring1 = np.array(offspring1).reshape((300, 300, 3)).astype(np.uint8)
    offspring2 = np.array(offspring2).reshape((300, 300, 3)).astype(np.uint8)

    return Individual(representation=offspring1.flatten()), Individual(representation=offspring2.flatten())

def block_uniform_crossover(parent1, parent2, block_size=10):
    """Implementation of block-based uniform crossover for HSV image recreation.

    Args:
        parent1 (Individual): First parent for crossover.
        parent2 (Individual): Second parent for crossover.
        block_size (int): Size of the blocks to swap.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    parent1_repr = np.array(parent1.representation).reshape((300, 300, 3))
    parent2_repr = np.array(parent2.representation).reshape((300, 300, 3))

    offspring1_repr = np.copy(parent1_repr)
    offspring2_repr = np.copy(parent2_repr)

    for i in range(0, 300, block_size):
        for j in range(0, 300, block_size):
            if np.random.rand() < 0.5:
                offspring1_repr[i:i+block_size, j:j+block_size, :] = parent2_repr[i:i+block_size, j:j+block_size, :]
                offspring2_repr[i:i+block_size, j:j+block_size, :] = parent1_repr[i:i+block_size, j:j+block_size, :]

    offspring_1 = Individual(representation=offspring1_repr.flatten())
    offspring_2 = Individual(representation=offspring2_repr.flatten())

    return offspring_1, offspring_2
