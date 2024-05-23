# from PIL import Image
# import os
# from charles import Individual , Population 
# from fitness import calculate_fitness
# from selection import fps
# from xo import single_point_xo
# from mutation import binary_mutation
# import numpy as np



# # Get the current working directory
# current_directory = "C:\\Users\\35193\\Desktop\\nova stuff\\sem 2\\Cifo\\Project"

# # Define the filename of your image
# image_filename = "image.jpg"

# # Construct the full path to the image file
# image_path = os.path.join(current_directory, image_filename)

# # Load the image using PIL
# image = Image.open(image_path)
# target_array = np.array(image)  # Convert the target image to numpy array


# # Create an instance of Population
# population_size = 50  # Adjust population size as needed
# optimization = "max"
# population = Population(population_size, image, optim=optimization)

# for individual in population:
#     try:
#         if not isinstance(individual, Individual):
#             raise TypeError(f"Expected Individual, got {type(individual)}")
        
#         individual.update_representation()  # Ensure representation is updated
#         individual_array = individual.representation  # Use the representation for fitness calculation
#         individual.fitness = calculate_fitness(individual_array, target_array)
#         print(f"Individual fitness calculated: {individual.fitness}")
#     except Exception as e:
#         print(f"Error calculating fitness for {individual}: {e}")


# # Genetic Algorithm Loop
# num_generations = 100  # Adjust number of generations as needed
# for generation in range(num_generations):
#     # Selection
#     selected_individuals = [fps(population) for _ in range(population_size)]

#     # Crossover and Mutation
#     offspring = []
#     for i in range(0, population_size, 2):
#         parent1 = selected_individuals[i]
#         parent2 = selected_individuals[i + 1]
#         offspring1, offspring2 = single_point_xo(parent1, parent2)
#         offspring1 = binary_mutation(offspring1)
#         offspring2 = binary_mutation(offspring2)
#         offspring.extend([offspring1, offspring2])

#     # Evaluate offspring fitness
#     for individual in offspring:
#         individual.fitness = calculate_fitness(individual, image)

#     # Replacement: Replace the current population with the offspring
#     population.individuals = offspring

#     # Output generation information, best fitness, etc. (optional)
#     best_individual = min(population, key=lambda x: x.fitness)
#     print(f"Generation {generation+1}: Best Fitness = {best_individual.fitness}")

# # After the loop, retrieve the best individual as the final result
# best_individual = min(population, key=lambda x: x.fitness)
# best_image = best_individual.image
# best_image.show()  # Display or save the final result




# from PIL import Image
# import os
# import numpy as np
# from charles import initialize_population
# from fitness import calculate_fitness
# from selection import fps_selection , tournament_selection
# from xo import single_point_xo
# from mutation import binary_mutation , swap_mutation

# # Load the target image
# current_directory = "C:\\Users\\35193\\Desktop\\nova stuff\\sem 2\\Cifo\\Projet\\Cifo-project"
# image_filename = "image.jpg"
# image_path = os.path.join(current_directory, image_filename)
# target_image = np.array(Image.open(image_path).convert("L"))

# # Genetic Algorithm Parameters
# population_size = 100
# num_generations = 1000
# num_individuals_selected=10
# tournament_size=3

# # Initialize the population
# population = initialize_population(population_size, target_image)
# # Genetic Algorithm Loop
# for generation in range(num_generations):
#     # Evaluate fitness of individuals in the population
#     fitness_scores = [calculate_fitness(individual, target_image) for individual in population]

#     # Select individuals for reproduction
#     selected_individuals = tournament_selection(population, num_individuals_selected,tournament_size)

#     # Create offspring through crossover and mutation
#     offspring = []
#     for i in range(0, num_individuals_selected, 2):
#         parent1 = selected_individuals[i]
#         parent2 = selected_individuals[i + 1]
#         offspring1, offspring2 = single_point_xo(parent1, parent2)
#         offspring1 = swap_mutation(offspring1)
#         offspring2 = swap_mutation(offspring2)
#         offspring.extend([offspring1, offspring2])

#     # Evaluate fitness of offspring
#     offspring_fitness = [calculate_fitness(individual, target_image) for individual in offspring]

#     # Replace the population with the offspring
#     population = offspring

#     # Output generation information
#     best_fitness = min(offspring_fitness)
#     print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

# # Retrieve the best individual as the final result
# best_individual = min(population, key=lambda x: calculate_fitness(x, target_image))
# best_image = Image.fromarray(best_individual)
# best_image.show()


from PIL import Image
import os
import numpy as np
from random import uniform, sample
from charles import initialize_population
from fitness import calculate_fitness
from selection import fps_selection, tournament_selection
from xo import single_point_xo, cycle_xo
from mutation import binary_mutation, swap_mutation

# Load the target image
current_directory = "C:\\Users\\35193\\Desktop\\nova stuff\\sem 2\\Cifo\\Projet\\Cifo-project"
image_filename = "image.jpg"
image_path = os.path.join(current_directory, image_filename)
target_image = np.array(Image.open(image_path).convert("L"))

# Genetic Algorithm Parameters
population_size = 100
num_generations = 15000
num_individuals_selected = 20
mutation_rate = 0.01
num_elites = 2
tournament_size = 5

# Initialize the population
population = initialize_population(population_size, target_image)

# Genetic Algorithm Loop
for generation in range(num_generations):
    # Evaluate fitness of individuals in the population
    fitness_scores = [calculate_fitness(individual, target_image) for individual in population]

    # Select individuals for reproduction
    selected_individuals = tournament_selection(population, num_individuals_selected, tournament_size, target_image)

    # Create offspring through crossover and mutation
    offspring = []
    for i in range(0, num_individuals_selected, 2):
        parent1 = selected_individuals[i]
        parent2 = selected_individuals[i + 1]
        offspring1, offspring2 = single_point_xo(parent1, parent2)  # or cycle_xo(parent1, parent2)
        offspring1 = binary_mutation(offspring1, mutation_rate)
        offspring2 = binary_mutation(offspring2, mutation_rate)
        offspring.extend([offspring1, offspring2])

    # Apply elitism
    elites = tournament_selection(population, num_elites, tournament_size, target_image)
    population = offspring + elites

    # Evaluate fitness of offspring
    offspring_fitness = [calculate_fitness(individual, target_image) for individual in offspring]

    # Output generation information
    best_fitness = min(offspring_fitness)
    print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

# Retrieve the best individual as the final result
best_individual = min(population, key=lambda x: calculate_fitness(x, target_image))
best_image = Image.fromarray(best_individual)
best_image.show()
