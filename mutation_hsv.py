from random import randint, sample, uniform, random
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from charles import Individual

def salt_and_pepper_mutation(individual, noise_level=0.1):
    """Applies salt-and-pepper noise to the HSV image.

    Args:
        individual (Individual): The individual to mutate.
        noise_level (float): Proportion of pixels to be altered.

    Returns:
        Individual: Mutated individual.
    """
    image_hsv = np.array(individual.representation).reshape((300, 300, 3))
    num_pixels = int(noise_level * image_hsv.size / 3)

    # Apply salt noise
    coords = [np.random.randint(0, i, num_pixels) for i in image_hsv.shape[:2]]
    image_hsv[coords[0], coords[1], :] = [0, 0, 255]  # maximum brightness (white in HSV)

    # Apply pepper noise
    coords = [np.random.randint(0, i, num_pixels) for i in image_hsv.shape[:2]]
    image_hsv[coords[0], coords[1], :] = [0, 0, 0]  # minimum brightness (black in HSV)

    individual = Individual(representation=image_hsv.flatten())

    return individual

def random_shape_mutation(individual, image_shape=(300, 300, 3)):
    """Adds a random shape with random HSV color to a random area of the image.

    Args:
        individual (Individual): The individual to mutate.
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        Individual: Mutated individual.
    """
    image_hsv = np.array(individual.representation).reshape(image_shape)

    # Select a random shape
    shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
    color = np.random.randint(0, 256, size=3)  # Random HSV color

    if shape_type == 'circle':
        # Parameters for the circle
        center = (np.random.randint(0, image_shape[1]), np.random.randint(0, image_shape[0]))
        radius = np.random.randint(2, 20)
        cv2.circle(image_hsv, center, radius, color.tolist(), -1)

    elif shape_type == 'rectangle':
        # Parameters for the rectangle
        pt1 = (np.random.randint(0, image_shape[1]), np.random.randint(0, image_shape[0]))
        pt2 = (pt1[0] + np.random.randint(2, 20), pt1[1] + np.random.randint(2, 20))
        cv2.rectangle(image_hsv, pt1, pt2, color.tolist(), -1)

    elif shape_type == 'triangle':
        # Parameters for the triangle
        pt1 = (np.random.randint(0, image_shape[1]), np.random.randint(0, image_shape[0]))
        pt2 = (pt1[0] + np.random.randint(2, 20), pt1[1] + np.random.randint(2, 20))
        pt3 = (pt1[0] + np.random.randint(2, 20), pt1[1] - np.random.randint(2, 20))
        points = np.array([pt1, pt2, pt3], np.int32)
        cv2.drawContours(image_hsv, [points], 0, color.tolist(), -1)
    
    individual = Individual(representation=image_hsv.flatten())
    return individual