import numpy as np
from random import random, randint
from charles import Individual

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