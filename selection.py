from random import uniform , sample
import numpy as np
from fitness import calculate_fitness


# def fps_selection(population, num_selected):
#     """Perform fitness proportionate selection.

#     Args:
#         population (list): List of individual arrays.
#         num_selected (int): Number of individuals to select.

#     Returns:
#         list: List of selected individual arrays.
#     """
#     total_fitness = sum([calculate_fitness(individual, population[0]) for individual in population])
#     selected_individuals = []
#     for _ in range(num_selected):
#         r = uniform(0, total_fitness)
#         position = 0
#         for individual in population:
#             position += calculate_fitness(individual, population[0])
#             if position > r:
#                 selected_individuals.append(individual)
#                 break
#     return selected_individuals


# def tournament_selection(population, num_selected, tournament_size):
#     """Perform tournament selection.

#     Args:
#         population (list): List of individual arrays.
#         num_selected (int): Number of individuals to select.
#         tournament_size (int): Size of the tournament.

#     Returns:
#         list: List of selected individual arrays.
#     """
#     selected_individuals = []
#     population_size = len(population)
    
#     while len(selected_individuals) < num_selected:
#         tournament = sample(population, tournament_size)
#         tournament_fitness = [calculate_fitness(individual, population[0]) for individual in tournament]
#         winner_index = tournament_fitness.index(max(tournament_fitness))
#         selected_individuals.append(tournament[winner_index])
    
#     return selected_individuals




def fps_selection(population, num_selected, target_image):
    """Perform fitness proportionate selection.

    Args:
        population (list): List of individual arrays.
        num_selected (int): Number of individuals to select.
        target_image (np.ndarray): Target image array for fitness calculation.

    Returns:
        list: List of selected individual arrays.
    """
    fitness_scores = np.array([calculate_fitness(individual, target_image) for individual in population])
    total_fitness = np.sum(fitness_scores)
    selection_probabilities = fitness_scores / total_fitness
    selected_individuals = np.random.choice(population, size=num_selected, p=selection_probabilities)
    return selected_individuals.tolist()


def tournament_selection(population, num_selected, tournament_size, target_image):
    """Perform tournament selection.

    Args:
        population (list): List of individual arrays.
        num_selected (int): Number of individuals to select.
        tournament_size (int): Size of the tournament.
        target_image (np.ndarray): Target image array for fitness calculation.

    Returns:
        list: List of selected individual arrays.
    """
    selected_individuals = []
    population_size = len(population)
    
    while len(selected_individuals) < num_selected:
        tournament = sample(population, tournament_size)
        tournament_fitness = [calculate_fitness(individual, target_image) for individual in tournament]
        winner_index = tournament_fitness.index(min(tournament_fitness))  # Select the individual with the best (lowest) fitness
        selected_individuals.append(tournament[winner_index])
    
    return selected_individuals
