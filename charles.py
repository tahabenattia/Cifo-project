from PIL import Image
import os
import numpy as np

        





# #The Individual class represents an individual solution or candidate image in the population.
# #It encapsulates the data and operations associated with an individual, such as the image data, fitness value,
# #and methods for initialization and representation.
# #In the context of image generation, an individual represents a single image with randomly initialized pixel values.
# #The purpose of the Individual class is to encapsulate the properties and behavior of an individual within the population.

# class Individual:
#     def __init__(self, target_image):
#         self.width, self.height = target_image.size
#         self.image = Image.fromarray(np.random.randint(0, 256, size=(self.height, self.width), dtype=np.uint8), mode='L')
#         self.fitness = None
#         self.representation = np.array(self.image)

#     def update_representation(self):
#         self.representation = np.array(self.image)

#     def __len__(self):
#         return len(self.representation)

#     def __getitem__(self, position):
#         return self.representation[position]

#     def __setitem__(self, position, value):
#         self.representation[position] = value    

#     def __repr__(self):
#         return f"Fitness: {self.fitness}"

#     def get_width(self):
#         return self.width

#     def get_height(self):
#         return self.height



# #The Population class manages a group of individuals, representing a population of potential solutions or images.
# #It initializes the population with a specified number of individuals, each representing a randomly generated image.
# #The Population class provides methods for accessing individuals within the population and managing the population as a whole, 
# #such as evaluating fitness and evolving individuals over generations.
# #It serves as a container for organizing and manipulating the individuals within the genetic algorithm framework.


# class Population:
#     def __init__(self, size, target_image, optim):
#         self.size = size
#         self.individuals = [Individual(target_image) for _ in range(size)]
#         self.optim = optim

#     def __len__(self):
#         return len(self.individuals)

#     def __getitem__(self, position):
#         return self.individuals[position]



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




