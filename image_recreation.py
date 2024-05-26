from charles import Population, Individual
from copy import copy
from selection import fps, tournament_sel
from mutation import swap_mutation, inversion_mutation, salt_and_pepper_mutation, edge_detection_mutation, random_shape_mutation
from xo import cycle_xo, pmx, two_point_xo, block_uniform_crossover
from skimage.metrics import structural_similarity as ssim
from operator import attrgetter
from random import random, randint

from PIL import Image
import os
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

    elif method == 'psnr':
        # Peak Signal-to-Noise Ratio
        mse = np.mean((individual_array - target_array) ** 2)
        if mse == 0:
            return float('inf')  # If the MSE is zero, return infinity (perfect match)
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return -psnr  # Return negative PSNR because higher PSNR means more similarity

    else:
        raise ValueError("Invalid method specified. Choose 'mse', 'mae', 'ssim', or 'psnr'.")


# Advanced Initialization Methods
def random_checkerboard_pattern(image_shape, square_size=5, flip_prob=0.5):
    """Create an individual with a randomized checkerboard pattern."""
    individual = np.zeros(image_shape)
    for i in range(0, image_shape[0], square_size * 2):
        for j in range(0, image_shape[1], square_size * 2):
            if random() < flip_prob:
                individual[i:i+square_size, j:j+square_size] = 255
                individual[i+square_size:i+square_size*2, j+square_size:j+square_size*2] = 255
            else:
                individual[i+square_size:i+square_size*2, j:j+square_size] = 255
                individual[i:i+square_size, j+square_size:j+square_size*2] = 255
    return individual


def random_circle_pattern(image_shape, num_circles=10, max_radius=20):
    """Create an individual with randomly placed circles."""
    individual = np.zeros(image_shape)
    for _ in range(num_circles):
        radius = randint(5, max_radius)
        x, y = randint(radius, image_shape[1] - radius), randint(radius, image_shape[0] - radius)
        Y, X = np.ogrid[:image_shape[0], :image_shape[1]]
        mask = (X - x) ** 2 + (Y - y) ** 2 <= radius ** 2
        individual[mask] = 255
    return individual


def random_stripe_pattern(image_shape, min_stripe_width=5, max_stripe_width=20, orientation='vertical', flip_prob=0.3):
    """Create an individual with a random stripe pattern.

    Args:
        image_shape (tuple): Shape of the image (height, width).
        min_stripe_width (int): Minimum width of the stripes.
        max_stripe_width (int): Maximum width of the stripes.
        orientation (str): 'vertical' or 'horizontal' orientation for the stripes.
        flip_prob (float): Probability of flipping the color of the stripe.

    Returns:
        np.array: The image array with the stripe pattern.
    """
    individual = np.zeros(image_shape)
    if orientation == 'vertical':
        j = 0
        while j < image_shape[1]:
            stripe_width = randint(min_stripe_width, max_stripe_width)
            if random() < flip_prob:
                individual[:, j:j + stripe_width] = 255
            j += stripe_width
    else:  # horizontal stripes
        i = 0
        while i < image_shape[0]:
            stripe_width = randint(min_stripe_width, max_stripe_width)
            if random() < flip_prob:
                individual[i:i + stripe_width, :] = 255
            i += stripe_width
    return individual


def random_pattern_initialization(image_shape):
    """Initialize an individual using either checkerboard or gradient pattern with equal probability."""

    pattern_type = np.random.choice(['checkerboard', 'stripe', 'circle'])

    if pattern_type == 'checkerboard':
        individual = random_checkerboard_pattern(image_shape, square_size=5, flip_prob=0.4)
    elif pattern_type == 'stripe':
        orientation = 'vertical' if random() < 0.5 else 'horizontal'
        individual = random_stripe_pattern(image_shape, orientation=orientation, flip_prob=0.3)
    elif pattern_type == 'circle':
        individual = random_circle_pattern(image_shape, num_circles=25, max_radius=20)
    
    return Individual(representation=individual.flatten())


def display_and_save_image_pil(best_individual, image_shape, save_path='./data/output_image.png'):
    # Convert the individual's representation to a 2D array
    image_array = np.array(best_individual.representation).reshape(image_shape).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(image_array, mode='L')

    # Display the image
    image.show()
    # Save the image locally
    image.save(save_path)

"""
Difficulties with random pixel initiation:
https://medium.com/@sebastian.charmot/genetic-algorithm-for-image-recreation-4ca546454aaa
"""

# Target Image Path
image_path = "./data/image.jpg"
# Load Target Image
target_image = Image.open(image_path).convert("L")
# Resize the image to 400x400 numpy array
target_image = np.array(target_image.resize((300, 300)))
# Flatten Target Image to 1D Numpy Array
target_image_flatten = target_image.flatten()


# Monkey patching
# Define a lambda function to bind the target image and method to get_fitness
Individual.get_fitness = lambda self: get_fitness(self, target_image_flatten, method='mae')
#Individual.get_fitness = get_fitness(self, target_array=target_image_flatten)
# Individual.get_neighbours = get_neighbours

# Choose initialization method
init_method = lambda: random_pattern_initialization(image_shape=(300, 300))

P = Population(size=200, optim="min", sol_size=target_image_flatten.shape[0],
                 valid_set=[i for i in range(256)], repetition = True, init_method=init_method)

P.evolve(gens=5000, xo_prob1=0.6, xo_prob2=0.4, mut_prob1=0.2, mut_prob2=0.15,
         select=tournament_sel, xo1=two_point_xo, xo2=block_uniform_crossover,
         mutate1=random_shape_mutation, mutate2=salt_and_pepper_mutation, elitism=True)

# Get the best individual
if P.optim == "min":
    best_individual = min(P, key=attrgetter('fitness'))
elif P.optim == "max":
    best_individual = max(P, key=attrgetter('fitness'))


# Display the resulting image
display_and_save_image_pil(best_individual, (target_image.shape[0], target_image.shape[1]))


