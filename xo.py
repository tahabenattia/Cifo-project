from random import randint, sample, uniform, random
import numpy as np
from scipy.ndimage import gaussian_filter
from charles import Individual


def single_point_xo(parent1, parent2):
    """Implementation of single point crossover.

    Args:
        parent1 (Individual): First parent for crossover.
        parent2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    xo_point = randint(1, len(parent1)-1)
    offspring1 = parent1[:xo_point] + parent2[xo_point:]
    offspring2 = parent2[:xo_point] + parent1[xo_point:]
    return offspring1, offspring2

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

def cycle_xo(p1, p2):
    """Implementation of cycle crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # offspring placeholders
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

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


def pmx(p1, p2):
    """Implementation of partially matched/mapped crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    xo_points = sample(range(len(p1)), 2)
    #xo_points = [3,6]
    xo_points.sort()

    def pmx_offspring(x,y):
        o = [None] * len(x)
        # offspring2
        o[xo_points[0]:xo_points[1]]  = x[xo_points[0]:xo_points[1]]
        z = set(y[xo_points[0]:xo_points[1]]) - set(x[xo_points[0]:xo_points[1]])

        # numbers that exist in the segment
        for i in z:
            temp = i
            index = y.index(x[y.index(temp)])
            while o[index] is not None:
                temp = index
                index = y.index(x[temp])
            o[index] = i

        # numbers that doesn't exist in the segment
        while None in o:
            index = o.index(None)
            o[index] = y[index]
        return o

    o1, o2 = pmx_offspring(p1, p2), pmx_offspring(p2, p1)
    return o1, o2


def geo_xo(p1,p2):
    """Implementation of arithmetic crossover/geometric crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individual: Offspring, resulting from the crossover.
    """
    o = [None] * len(p1)
    for i in range(len(p1)):
        r = uniform(0,1)
        o[i] = p1[i] * r + (1-r) * p2[i]
    return o


if __name__ == "__main__":
    #p1, p2 = [9,8,2,1,7,4,5,10,6,3], [1,2,3,4,5,6,7,8,9,10]
    #p1, p2 = [2,7,4,3,1,5,6,9,8], [1,2,3,4,5,6,7,8,9]
    p1, p2 = [9,8,4,5,6,7,1,3,2,10], [8,7,1,2,3,10,9,5,4,6]
    o1, o2 = pmx(p1, p2)
    print(o1,o2)