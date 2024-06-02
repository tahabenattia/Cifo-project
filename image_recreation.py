from charles import Population, Individual
from initialization import random_pattern_initialization
from selection import tournament_sel, fps
from mutation import inversion_mutation, salt_and_pepper_mutation, edge_detection_mutation, random_shape_mutation
from xo import two_point_xo, block_uniform_crossover, smooth_two_point_crossover
from evaluation import display_and_save_image_pil

import os
from random import random, randint
from copy import copy

from skimage.metrics import structural_similarity as ssim
from operator import attrgetter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def get_fitness(self, target_array, method='mse'):
    # Ensure both arrays are numpy arrays
    individual_array = np.array(self.representation)
    target_array = np.array(target_array)

    if method == 'mse':
        # Mean Squared Error
        mse = np.mean((individual_array - target_array) ** 2)
        return mse

    elif method == 'mae':
        # Mean Absolute Error
        mae = np.mean(np.abs(individual_array - target_array))
        return mae

    elif method == 'ssim':
        # Structural Similarity Index
        individual_2d = individual_array.reshape(target_array.shape)
        score, _ = ssim(individual_2d, target_array, full=True, data_range=target_array.max() - target_array.min())
        return -score  # Return negative SSIM because higher SSIM means more similarity

    else:
        raise ValueError("Invalid method specified. Choose 'mse', 'mae' or 'ssim'.")



"""
Difficulties with random pixel initiation:
https://medium.com/@sebastian.charmot/genetic-algorithm-for-image-recreation-4ca546454aaa
"""

configurations = [
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": False,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": fps,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'ssim',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mse',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pixel_values',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 25,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": two_point_xo,
    "mutate1": random_shape_mutation,
    "mutate2": random_shape_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": inversion_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": block_uniform_crossover,
    "mutate1": edge_detection_mutation,
    "mutate2": inversion_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": two_point_xo,
    "xo2": smooth_two_point_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": salt_and_pepper_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    },
    {
    "fitness_method": 'mae',
    "population_size": 50,
    "init_method": 'random_pattern',
    "gens": 3000,
    "xo_prob1": 0.6,
    "xo_prob2": 0.4,
    "mut_prob1": 0.2,
    "mut_prob2": 0.15,
    "select": tournament_sel,
    "xo1": smooth_two_point_crossover,
    "xo2": block_uniform_crossover,
    "mutate1": random_shape_mutation,
    "mutate2": edge_detection_mutation,
    "elitism": True,
    "image_shape": (200, 200),
    "visualize_evolution": True
    }
]

for configs in configurations:
    # Reset P
    P = None
    print('__________________________________')
    print('New evolution configuration start.')
    print(f"Population size: {configs['population_size']}, Generations: {configs['gens']}.")
    # Target Image Path
    image_path = "./data/image.jpg"
    # Load Target Image
    target_image = Image.open(image_path).convert("L")
    # Resize the image to numpy array
    image_shape = configs["image_shape"]
    target_image = np.array(target_image.resize(image_shape))
    # Flatten Target Image to 1D Numpy Array
    target_image_flatten = target_image.flatten()

    # Monkey patching
    # Define a lambda function to bind the target image and method to get_fitness
    if configs["fitness_method"] == 'ssim':
        Individual.get_fitness = lambda self: get_fitness(self, target_image, method=configs["fitness_method"])
    else:
        Individual.get_fitness = lambda self: get_fitness(self, target_image_flatten, method=configs["fitness_method"])
    
    # Initiate Population
    # Choose initialization method
    if configs["init_method"] == 'random_pattern':
        init_method = lambda: random_pattern_initialization(image_shape=image_shape)
        P = Population(size=configs["population_size"], optim="min", sol_size=target_image_flatten.shape[0],
                        valid_set=[i for i in range(256)], repetition = True, init_method=init_method)
    # Else random pixel value initialization
    else:
        P = Population(size=configs["population_size"], optim="min", sol_size=target_image_flatten.shape[0],
                valid_set=[i for i in range(256)], repetition = True)

    # Evolve population and get fitness values
    best_fitness_values, avg_fitness_values, diversity_values = P.evolve(
        gens=configs["gens"], xo_prob1=configs["xo_prob1"], xo_prob2=configs["xo_prob2"], mut_prob1=configs["mut_prob1"],
        mut_prob2=configs["mut_prob2"], select=configs["select"], xo1=configs["xo1"], xo2=configs["xo2"],
        mutate1=configs["mutate1"], mutate2=configs["mutate2"], elitism=configs["elitism"],
        image_shape=image_shape, visualize_evolution=configs["visualize_evolution"], configs=configs
    )


