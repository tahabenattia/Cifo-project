from PIL import Image
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    if callable(obj):
        return obj.__name__
    return str(obj)

def save_configs(configs, save_path):
    serializable_configs = convert_to_serializable(configs)
    with open(save_path, 'w') as f:
        json.dump(serializable_configs, f, indent=4)


def display_and_save_image_pil(best_individual, image_shape, save_path='./data/output_image.png', show=False):
    # Convert the individual's representation to a 2D array
    image_array = np.array(best_individual.representation).reshape(image_shape).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(image_array, mode='L')

    # Display the image if show = True
    if show == True:
        image.show()

    # Save the image locally
    image.save(save_path)


def plot_evolution_statistics(best_fitness_values, avg_fitness_values, diversity_values,
                                generations, save_path=None, fitness_title='Evolution of Fitness',
                                diversity_title='Diversity Over Generations', show=False):
    """
    Plots the evolution statistics for fitness and diversity over generations.

    Args:
        best_fitness_values (list): List of best fitness values per generation.
        avg_fitness_values (list): List of average fitness values per generation.
        diversity_values (list): List of diversity values per generation.
        generations (int): Number of generations.
        save_path (str, optional): Path to save the plot image. Defaults to None.
        fitness_title (str, optional): Title for the fitness plot. Defaults to 'Evolution of Fitness'.
        diversity_title (str, optional): Title for the diversity plot. Defaults to 'Diversity Over Generations'.
        show(boolean): True if plots should be displayed.

    Returns:
        None
    """
    plt.figure(figsize=(12, 6))

    # Plot fitness values
    plt.subplot(1, 2, 1)
    plt.plot(range(1, generations + 1), best_fitness_values, label='Best Fitness')
    plt.plot(range(1, generations + 1), avg_fitness_values, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(fitness_title)
    plt.legend()
    plt.grid(True)

    # Plot diversity values
    plt.subplot(1, 2, 2)
    plt.plot(range(1, generations + 1), diversity_values, label='Population Diversity')
    plt.xlabel('Generation')
    plt.ylabel('Diversity (Standard Deviation of Fitness)')
    plt.title(diversity_title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path)
    
    # Show the plot if show = True
    if show == True:
        plt.show()
