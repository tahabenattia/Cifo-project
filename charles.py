from evaluation import display_and_save_image_pil, plot_evolution_statistics, save_configs

from operator import attrgetter
from random import shuffle, choice, sample, random
from copy import copy
import numpy as np
import datetime
import os


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
    
    def evolve(self, gens, xo_prob1, xo_prob2, mut_prob1, mut_prob2, select, xo1, xo2,
                mutate1, mutate2, elitism, image_shape, visualize_evolution, configs):
        self.best_fitness_values = []
        self.avg_fitness_values = []
        self.diversity_values = []

        # Store the start time as a datetime object
        start_time = datetime.datetime.now()
        start_time_formatted = start_time.strftime('%d.%m.%y_%H-%M-%S')
        print('START TIME:', start_time_formatted)
        os.makedirs(f'./output/{start_time_formatted}', exist_ok=True)

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
                    offspring1, offspring2 = xo1(parent1, parent2, image_shape)
                elif random() < (xo_prob1 + xo_prob2):
                    offspring1, offspring2 = xo2(parent1, parent2, image_shape)
                # replication
                else:
                    offspring1, offspring2 = parent1, parent2
                # mutation with prob
                if random() < mut_prob1:
                    offspring1 = mutate1(offspring1, image_shape)
                elif random() < (mut_prob1 + mut_prob2):
                    offspring1 = mutate2(offspring1, image_shape)
                if random() < mut_prob1:
                    offspring2 = mutate1(offspring2, image_shape)
                elif random() < (mut_prob1 + mut_prob2):
                    offspring2 = mutate2(offspring2, image_shape)

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

            if (i + 1) % 100 == 0 or i == 0:
                 # Some time later, calculate and print the elapsed time
                current_time = datetime.datetime.now()
                elapsed_time = current_time - start_time
                print('-------------------------')
                # Format elapsed time as hh:mm:ss
                elapsed_hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
                elapsed_minutes, elapsed_seconds = divmod(remainder, 60)
                formatted_elapsed_time = f'{int(elapsed_hours):02}:{int(elapsed_minutes):02}:{int(elapsed_seconds):02}'
                print('Time since start: ', formatted_elapsed_time )

                if self.optim == "max":
                    print(f"Best individual of gen #{i + 1}: {max(self, key=attrgetter('fitness'))}")
                elif self.optim == "min":
                    print(f"Best individual of gen #{i + 1}: {min(self, key=attrgetter('fitness'))}")

            # Take screenshots of evolution progress at certain checkpoints
            if (i == 0) or (i + 1 == gens // 10) or (i + 1 == gens // 5) or ( i+ 1 == gens // 2) or (i + 1 == gens // 1.5) or (i + 1 == gens):
                if visualize_evolution == True:
                    # Save progress every certain generations
                    best_individual = min(self.individuals, key=attrgetter('fitness'))
                    image_save_path = f'./output/{start_time_formatted}/output_image_gen_{i + 1}_{start_time_formatted}.png'
                    # Save visual progress
                    display_and_save_image_pil(best_individual, image_shape, save_path=image_save_path, show=False)

        # Save Metrics      
        plot_save_path = f'./output/{start_time_formatted}/evolution_statistics_gen_{gens}_{start_time_formatted}.png'
        plot_evolution_statistics(
                self.best_fitness_values, 
                self.avg_fitness_values, 
                self.diversity_values, 
                generations=gens,
                save_path=plot_save_path,
                show=False
            )
        # Save used configs
        config_save_path = f'./output/{start_time_formatted}/config_gen_{gens}_{start_time_formatted}.json'
        save_configs(configs, save_path=config_save_path)

        print('-------------------')
        print('Evolution finished.')

        return self.best_fitness_values, self.avg_fitness_values, self.diversity_values

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

