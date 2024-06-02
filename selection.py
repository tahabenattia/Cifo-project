from operator import attrgetter
from random import uniform, choice


def tournament_sel(population, tour_size=5):
    tournament = [choice(population) for _ in range(tour_size)]
    if population.optim == "max":
        return max(tournament, key=attrgetter('fitness'))
    elif population.optim == "min":
        return min(tournament, key=attrgetter('fitness'))

def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """
    if population.optim == "max":
        total_fitness = sum([i.fitness for i in population])
        r = uniform(0, total_fitness)
        position = 0
        for individual in population:
            position += individual.fitness
            if position > r:
                return individual
    elif population.optim == "min":
        max_fitness = max([i.fitness for i in population])
        min_fitness = min([i.fitness for i in population])
        # Transform fitness values so higher values represent better fitness
        transformed_fitness = [max_fitness - i.fitness + min_fitness for i in population]
        total_fitness = sum(transformed_fitness)
        r = uniform(0, total_fitness)
        position = 0
        for i, individual in enumerate(population):
            position += transformed_fitness[i]
            if position > r:
                return individual
    else:
        raise Exception(f"Optimization not specified (max/min)")