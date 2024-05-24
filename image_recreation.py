from charles import Population, Individual
from copy import copy
from selection import fps, tournament_sel
from mutation import swap_mutation, inversion_mutation
from xo import cycle_xo, pmx, two_point_xo
from skimage.metrics import structural_similarity as ssim
from operator import attrgetter

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
        score, _ = ssim(individual_2d, target_array, full=True)
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


def display_image_pil(best_individual, image_shape):
    # Convert the individual's representation to a 2D array
    image_array = np.array(best_individual.representation).reshape(image_shape).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(image_array, mode='L')

    # Display the image
    image.show()


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
Individual.get_fitness = lambda self: get_fitness(self, target_image_flatten, method='mse')
#Individual.get_fitness = get_fitness(self, target_array=target_image_flatten)
# Individual.get_neighbours = get_neighbours


P = Population(size=200, optim="min", sol_size=target_image_flatten.shape[0],
                 valid_set=[i for i in range(256)], repetition = True)

P.evolve(gens=1000, xo_prob=1, mut_prob=0.3, select=tournament_sel,
         xo=two_point_xo, mutate=inversion_mutation, elitism=True)


# Get the best individual
if P.optim == "min":
    best_individual = min(P, key=attrgetter('fitness'))
elif P.optim == "max":
    best_individual = max(P, key=attrgetter('fitness'))


# Display the resulting image
display_image_pil(best_individual, (target_image.shape[0], target_image.shape[1]))


