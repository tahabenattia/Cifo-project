# from random import uniform


# def fps(population):
#     """Fitness proportionate selection implementation.

#     Args:
#         population (Population): The population we want to select from.

#     Returns:
#         Individual: selected individual.
#     """
#     if population.optim == "max":
#         total_fitness = sum([i.fitness for i in population])
#         r = uniform(0, total_fitness)
#         position = 0
#         for individual in population:
#             position += individual.fitness
#             if position > r:
#                 return individual
#     elif population.optim == "min":
#         raise NotImplementedError
#     else:
#         raise Exception(f"Optimization not specified (max/min)")



from random import uniform
import numpy as np
from fitness import calculate_fitness


def fps_selection(population, num_selected):
    """Perform fitness proportionate selection.

    Args:
        population (list): List of individual arrays.
        num_selected (int): Number of individuals to select.

    Returns:
        list: List of selected individual arrays.
    """
    total_fitness = sum([calculate_fitness(individual, population[0]) for individual in population])
    selected_individuals = []
    for _ in range(num_selected):
        r = uniform(0, total_fitness)
        position = 0
        for individual in population:
            position += calculate_fitness(individual, population[0])
            if position > r:
                selected_individuals.append(individual)
                break
    return selected_individuals

