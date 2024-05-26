from random import randint, sample, uniform, random
import numpy as np
from scipy.ndimage import gaussian_filter
from charles import Individual


def two_point_xo(parent1, parent2):
    """Implementation of two-point crossover.

    Args:
        parent1 (Individual): First parent for crossover.
        parent2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    # Randomly select two crossover points
    points = sorted(sample(range(len(parent1.representation)), 2))

    # Convert to list, to be able to use + operations to concatenate lists
    parent1 = list(parent1)
    parent2 = list(parent2)
    
    # Create the offspring by combining segments from both parents
    offspring1 = parent1[:points[0]] + parent2[points[0]:points[1]] + parent1[points[1]:]
    offspring2 = parent2[:points[0]] + parent1[points[0]:points[1]] + parent2[points[1]:]

    # Convert back to numpy array, to ensure data consistency
    offspring1 = np.array(offspring1).astype(np.uint8)
    offspring2 = np.array(offspring2).astype(np.uint8)

    return offspring1, offspring2


def smooth_two_point_crossover(parent1, parent2, sigma=1.0):
    """Implementation of two-point crossover with smoothing for image recreation.

    Args:
        parent1 (Individual): First parent for crossover.
        parent2 (Individual): Second parent for crossover.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    parent1_repr = np.array(parent1.representation).reshape((300, 300))
    parent2_repr = np.array(parent2.representation).reshape((300, 300))
    
    points = sorted(sample(range(300), 2))
    
    offspring1_repr = np.copy(parent1_repr)
    offspring2_repr = np.copy(parent2_repr)
    
    offspring1_repr[:, points[0]:points[1]] = parent2_repr[:, points[0]:points[1]]
    offspring2_repr[:, points[0]:points[1]] = parent1_repr[:, points[0]:points[1]]
    
    # Apply Gaussian smoothing to the boundary regions
    offspring1_repr = gaussian_filter(offspring1_repr, sigma=sigma)
    offspring2_repr = gaussian_filter(offspring2_repr, sigma=sigma)

    offspring_1 = Individual(representation=offspring1_repr.flatten())
    offspring_2 = Individual(representation=offspring2_repr.flatten())
    
    return offspring_1, offspring_2


def block_uniform_crossover(parent1, parent2, block_size=10):
    """Implementation of block-based uniform crossover for image recreation.

    Args:
        parent1 (Individual): First parent for crossover.
        parent2 (Individual): Second parent for crossover.
        block_size (int): Size of the blocks to swap.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    parent1_repr = np.array(parent1.representation).reshape((300, 300))
    parent2_repr = np.array(parent2.representation).reshape((300, 300))
    
    offspring1_repr = np.copy(parent1_repr)
    offspring2_repr = np.copy(parent2_repr)
    
    for i in range(0, 300, block_size):
        for j in range(0, 300, block_size):
            if random() < 0.5:
                offspring1_repr[i:i+block_size, j:j+block_size] = parent2_repr[i:i+block_size, j:j+block_size]
                offspring2_repr[i:i+block_size, j:j+block_size] = parent1_repr[i:i+block_size, j:j+block_size]
    
    offspring_1 = Individual(representation=offspring1_repr.flatten())
    offspring_2 = Individual(representation=offspring2_repr.flatten())
    
    return offspring_1, offspring_2

