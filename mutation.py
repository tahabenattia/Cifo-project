from random import randint, sample
import numpy as np
from charles import Individual
from scipy.ndimage import sobel
import cv2


def salt_and_pepper_mutation(individual, image_shape, noise_level=0.1):
    """Applies salt-and-pepper noise to the image.

    Args:
        individual (Individual): The individual to mutate.
        noise_level (float): Proportion of pixels to be altered.

    Returns:
        Individual: Mutated individual.
    """
    image_array = np.array(individual).reshape(image_shape)
    num_pixels = int(noise_level * image_array.size)
    
    # Apply salt noise
    coords = [np.random.randint(0, i, num_pixels) for i in image_array.shape]
    image_array[coords[0], coords[1]] = 255
    
    # Apply pepper noise
    coords = [np.random.randint(0, i, num_pixels) for i in image_array.shape]
    image_array[coords[0], coords[1]] = 0

    individual = Individual(representation=image_array.flatten())

    return individual

def random_shape_mutation(individual, image_shape):
    """Adds a random black or white shape to a random area of the image.

    Args:
        individual (Individual): The individual to mutate.
        image_shape (tuple): Shape of the image (height, width).

    Returns:
        Individual: Mutated individual.
    """
    image_array = np.array(individual).reshape(image_shape)
    
    # Select a random shape
    shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
    color = int(np.random.choice([0, 255])) # Randomly choose black or white

    if shape_type == 'circle':
        # Parameters for the circle
        center = (randint(0, image_shape[1]), randint(0, image_shape[0]))
        radius = randint(2, 10)
        cv2.circle(image_array, center, radius, color, -1)

    elif shape_type == 'rectangle':
        # Parameters for the rectangle
        pt1 = (randint(0, image_shape[1]), randint(0, image_shape[0]))
        pt2 = (pt1[0] + randint(2, 10), pt1[1] + randint(2, 10))
        cv2.rectangle(image_array, pt1, pt2, color, -1)

    elif shape_type == 'triangle':
        # Parameters for the triangle
        pt1 = (randint(0, image_shape[1]), randint(0, image_shape[0]))
        pt2 = (pt1[0] + randint(2, 10), pt1[1] + randint(2, 10))
        pt3 = (pt1[0] + randint(2, 10), pt1[1] - randint(2, 10))
        points = np.array([pt1, pt2, pt3], np.int32)
        cv2.drawContours(image_array, [points], 0, color, -1)
    
    individual = Individual(representation=image_array.flatten())
    return individual

def edge_detection_mutation(individual, image_shape):
    """Enhances the edges in the image.

    Args:
        individual (Individual): The individual to mutate.

    Returns:
        Individual: Mutated individual.
    """

    image_array = np.array(individual).reshape(image_shape)
    
    # Apply Sobel filter for edge detection
    sx = sobel(image_array, axis=0, mode='constant')
    sy = sobel(image_array, axis=1, mode='constant')
    edges = np.hypot(sx, sy)
    
    # Add edges to the original image
    enhanced_image = image_array + edges
    enhanced_image = np.clip(enhanced_image, 0, 255)
    
    individual = Individual(representation=enhanced_image.flatten())

    return individual


def inversion_mutation(individual, image_shape):
    """Inversion mutation for a GA individual. Reverts a portion of the representation.

    Args:
        individual (Individual): A GA individual from charles.py

    Returns:
        Individual: Mutated Individual
    """
    mut_indexes = sample(range(0, len(individual)), 2)
    mut_indexes.sort()
    individual[mut_indexes[0]:mut_indexes[1]] = individual[mut_indexes[0]:mut_indexes[1]][::-1]
    return individual

