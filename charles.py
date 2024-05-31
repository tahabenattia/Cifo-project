from operator import attrgetter
from random import shuffle, choice, sample, random
from copy import copy
import numpy as np

class Individual:
    # we always initialize
    def __init__(self, representation=None, size=None, valid_set=None, repetition=True):

        if representation is None:
            if repetition:
                # individual will be chosen from the valid_set with a specific size
                self.representation = np.array([choice(valid_set) for i in range(size)]).astype(np.uint8)
            else:
                self.representation = sample(valid_set, size)

        # if we pass an argument like Individual(my_path)
        else:
            self.representation = representation

        # fitness will be assigned to the individual
        self.fitness = self.get_fitness()

    # methods for the class
    def get_fitness(self):
        raise Exception("You need to monkey patch the fitness function.")

    def get_neighbours(self):
        raise Exception("You need to monkey patch the neighbourhood function.")

    def index(self, value):
        return self.representation.index(value)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f" Fitness: {self.fitness}"
    

class Population:
    def __init__(self, size, optim, **kwargs):

        # population size
        self.size = size

        # defining the optimization problem as a minimization or maximization problem
        self.optim = optim

        self.individuals = []

        if 'init_method' in kwargs:
            init_method = kwargs['init_method']
            for _ in range(size):
                self.individuals.append(init_method())
        else:
            for _ in range(size):
                self.individuals.append(
                    Individual(
                        size=kwargs["sol_size"],
                        valid_set=kwargs["valid_set"],
                        repetition=kwargs["repetition"]
                    )
                )
    
    def evolve(self, gens, xo_prob1, xo_prob2, mut_prob1, mut_prob2, select, xo1, xo2, mutate1, mutate2, elitism):
        self.best_fitness_values = []
        self.avg_fitness_values = []
        self.diversity_values = []

        for i in range(gens):
            new_pop = []

            if elitism:
                if self.optim == "max":
                    elite = copy(max(self.individuals, key=attrgetter('fitness')))
                elif self.optim == "min":
                    elite = copy(min(self.individuals, key=attrgetter('fitness')))

            while len(new_pop) < self.size:
                # selection
                parent1, parent2 = select(self), select(self)
                # xo1 with prob1
                if random() < xo_prob1:
                    offspring1, offspring2 = xo1(parent1, parent2)
                elif random() < (xo_prob1 + xo_prob2):
                    offspring1, offspring2 = xo2(parent1, parent2)
                # replication
                else:
                    offspring1, offspring2 = parent1, parent2
                # mutation with prob
                if random() < mut_prob1:
                    offspring1 = mutate1(offspring1)
                elif random() < (mut_prob1 + mut_prob2):
                    offspring1 = mutate2(offspring1)
                if random() < mut_prob1:
                    offspring2 = mutate1(offspring2)
                elif random() < (mut_prob1 + mut_prob2):
                    offspring2 = mutate2(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism:
                if self.optim == "max":
                    worst = min(new_pop, key=attrgetter('fitness'))
                    if elite.fitness > worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)
                if self.optim == "min":
                    worst = max(new_pop, key=attrgetter('fitness'))
                    if elite.fitness < worst.fitness:
                        new_pop.pop(new_pop.index(worst))
                        new_pop.append(elite)

            self.individuals = new_pop

            # Log fitness values
            best_fitness = min(individual.fitness for individual in self.individuals)
            avg_fitness = sum(individual.fitness for individual in self.individuals) / len(self.individuals)
            diversity = np.std([individual.fitness for individual in self.individuals])
            
            self.best_fitness_values.append(best_fitness)
            self.avg_fitness_values.append(avg_fitness)
            self.diversity_values.append(diversity)

            if self.optim == "max":
                print(f"Best individual of gen #{i + 1}: {max(self, key=attrgetter('fitness'))}")
            elif self.optim == "min":
                print(f"Best individual of gen #{i + 1}: {min(self, key=attrgetter('fitness'))}")

        return self.best_fitness_values, self.avg_fitness_values, self.diversity_values

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]




