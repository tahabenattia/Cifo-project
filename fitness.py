import numpy as np

def calculate_fitness(individual_array, target_array):

        # Calculate pixel-wise difference
        difference = np.abs(individual_array - target_array)

        # Calculate fitness as the sum of absolute differences
        fitness = np.sum(difference)

        return fitness


