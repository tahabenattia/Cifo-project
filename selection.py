from operator import attrgetter
from random import uniform, choice


def tournament_sel(population, tour_size=3):
    tournament = [choice(population) for _ in range(tour_size)]
    if population.optim == "max":
        return max(tournament, key=attrgetter('fitness'))
    elif population.optim == "min":
        return min(tournament, key=attrgetter('fitness'))
