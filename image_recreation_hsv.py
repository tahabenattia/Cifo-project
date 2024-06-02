from charles import Population, Individual
from selection import tournament_sel
from mutation_hsv import salt_and_pepper_mutation, random_shape_mutation
from xo_hsv import two_point_xo, block_uniform_crossover
from skimage.metrics import structural_similarity as ssim
from operator import attrgetter
from random import random, randint

from PIL import Image
import cv2
import os
import numpy as np

def get_fitness(self, target_array, method='mse'):
    # Reshape the flattened images to 300x300x3
    individual_array = np.array(self.representation).reshape((300, 300, 3))
    target_array = np.array(target_array).reshape((300, 300, 3))

    if method == 'hsv':
        diff = np.linalg.norm(individual_array - target_array, axis=1)
        hsv = np.sum(diff)
        return hsv

    else:
        raise ValueError("Invalid method specified. Choose 'mse', 'mae', 'ssim', or 'psnr'.")

# different initialization methods
def random_checkerboard_pattern(image_shape, square_size=5, flip_prob=0.5):
    rows, cols = image_shape
    checkerboard = np.zeros((rows, cols, 3), dtype=np.uint8)
    num_squares_x = cols // square_size
    num_squares_y = rows // square_size
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            if random() < flip_prob:
                checkerboard[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = [255, 255, 255]
    return checkerboard

def random_stripe_pattern(image_shape, orientation='horizontal', flip_prob=0.3):
    rows, cols = image_shape
    stripe = np.zeros((rows, cols, 3), dtype=np.uint8)
    if orientation == 'horizontal':
        for i in range(rows):
            if random() < flip_prob:
                stripe[i, :] = [255, 255, 255]
    else:
        for j in range(cols):
            if random() < flip_prob:
                stripe[:, j] = [255, 255, 255]
    return stripe

def random_circle_pattern(image_shape, num_circles=25, max_radius=20):
    rows, cols = image_shape
    circle_pattern = np.zeros((rows, cols, 3), dtype=np.uint8)
    for _ in range(num_circles):
        center_x = np.random.randint(0, cols)
        center_y = np.random.randint(0, rows)
        radius = np.random.randint(1, max_radius)
        color = (255, 255, 255) if random() < 0.5 else (0, 0, 0)
        cv2.circle(circle_pattern, (center_x, center_y), radius, color, -1)
    return circle_pattern

# helper function
def convert_to_hsv_and_flatten(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv_image.reshape(-1, 3)

def random_pattern_initialization(image_shape):
    """Initialize an individual using either checkerboard, stripe, or circle pattern."""
    
    pattern_type = np.random.choice(['checkerboard', 'stripe', 'circle'])

    if pattern_type == 'checkerboard':
        individual_rgb = random_checkerboard_pattern(image_shape, square_size=5, flip_prob=0.4)
    elif pattern_type == 'stripe':
        orientation = 'vertical' if random() < 0.5 else 'horizontal'
        individual_rgb = random_stripe_pattern(image_shape, orientation=orientation, flip_prob=0.3)
    elif pattern_type == 'circle':
        individual_rgb = random_circle_pattern(image_shape, num_circles=25, max_radius=20)

    individual_hsv_flat = convert_to_hsv_and_flatten(individual_rgb)
    
    return Individual(representation=individual_hsv_flat)

def display_and_save_image_pil(best_individual, image_shape, save_path='./data/output_image.png'):
    # Convert the individual's representation back to a 3D HSV array
    image_hsv = np.array(best_individual.representation).reshape(image_shape).astype(np.uint8)

    # Convert HSV image back to RGB
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)

    # Convert RGB array to a PIL image
    image = Image.fromarray(image_rgb)

    # Display the image
    image.show()

    # Save the image locally
    image.save(save_path)

if __name__ == "__main__":
    image_path = "./data/starry_night.jpg"
    # Load Target Image
    target_image = Image.open(image_path).convert("RGB")
    # Resize the image to 400x400 numpy array
    target_image = np.array(target_image.resize((300, 300)))
    # Convert image to HSV
    target_image_hsv = cv2.cvtColor(target_image, cv2.COLOR_RGB2HSV)
    # Flatten Target Image to 1D Numpy Array
    target_image_flatten = target_image_hsv.reshape(-1, 3)


    # Monkey patching
    # Define a lambda function to bind the target image and method to get_fitness
    Individual.get_fitness = lambda self: get_fitness(self, target_image_flatten, method='hsv')
    #Individual.get_fitness = get_fitness(self, target_array=target_image_flatten)
    # Individual.get_neighbours = get_neighbours

    # Choose initialization method
    init_method = lambda: random_pattern_initialization(image_shape=(300, 300))

    P = Population(size=200, optim="min", sol_size=target_image_flatten.shape[0],
                    valid_set=[i for i in range(256)], repetition = True, init_method=init_method)

    P.evolve(gens=10000, xo_prob1=0.45, xo_prob2=0.45, mut_prob1=0.03, mut_prob2=0.03,
            select=tournament_sel, xo1=two_point_xo, xo2=block_uniform_crossover,
            mutate1=random_shape_mutation, mutate2=salt_and_pepper_mutation, elitism=True)

    # Get the best individual
    if P.optim == "min":
        best_individual = min(P, key=attrgetter('fitness'))
    elif P.optim == "max":
        best_individual = max(P, key=attrgetter('fitness'))
    
    display_and_save_image_pil(best_individual, (300, 300, 3))