from PIL import Image
import os
import numpy as np

def initialize_population(population_size, target_image):
    """Initialize the population of individuals.

    Args:
        population_size (int): Size of the population.
        target_image (np.ndarray): Target image as a NumPy array.

    Returns:
        list: List of individual arrays representing the population.
    """
    population = []
    height, width = target_image.shape
    for _ in range(population_size):
        individual = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
        population.append(individual)
    return population




