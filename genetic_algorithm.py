
import numpy  # for demo
from numpy import empty, where, max, random, array, uint8
from sys import maxint
# generic genetic algorithm

# calculate fitness of population
# population: vector with inputs
# fitness_func: funktion that can rate every input of the population
def calc_fitness(population, fitness_func):
    fitness = [fitness_func(i) for i in population]
    return fitness

# selects the number of parents for the next population
def select_mating_pool(population, fitness, num_parents):
    parents = empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_id = where(fitness == max(fitness))
        max_id = max_id[0][0]
        parents[i, :] = population[max_id, :]
        fitness[max_id] = -maxint
    return parents


def crossover(parents, offspring_size):
    offspring = empty(offspring_size)
    crossover_point = uint8(offspring_size[1] / 2)
    for i in range(offspring_size[0]):
        parent1_id = i % parents.shape[0]
        parent2_id = (i + 1) % parents.shape[0]
        offspring[i, :crossover_point] = parents[parent1_id, :crossover_point]
        offspring[i, crossover_point:] = parents[parent2_id, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    for i in range(offspring_crossover.shape[0]):
        random_value = random.uniform(-1.0, 1.0, 1)
        random_index = random.randint(0, offspring_crossover.shape[1], 1)[0]
        offspring_crossover[i, random_index] = offspring_crossover[i,
                                                                   random_index] + random_value
    return offspring_crossover


if __name__ == "__main__":
    # this is a pretty easy function
    def sphere_function(x):
        total = 0
        for xi in x:
            total = total + pow(xi, 2)
        return - total
    pop_size = (15, 4)
    min_in_population_start = 1
    max_in_population_start = 100
    num_parents_mating = 5
    generations = 2000
    new_population = numpy.random.uniform(
        low=min_in_population_start, high=max_in_population_start, size=pop_size)
    for generation in range(generations):
        fitness = calc_fitness(new_population, sphere_function)
        parents = select_mating_pool(
            new_population, fitness, num_parents_mating)
        print("Best Indiividual: ", parents[0])
        offspring_crossover = crossover(
            parents, (pop_size[0] - parents.shape[0], pop_size[1]))
        offspring_mutation = mutation(offspring_crossover)
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
    final_fitness = calc_fitness(new_population, sphere_function)
    id_of_best = numpy.where(final_fitness == max(final_fitness))
    best_normalized = new_population[id_of_best][0]
    print("x", best_normalized, "min_output", sphere_function(best_normalized))
