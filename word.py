import random
import string

# Target word to guess
TARGET_WORD = "ironman my man"

# Genetic Algorithm parameters
POPULATION_SIZE = 200
MUTATION_RATE = 0.1
NUM_GENERATIONS = 200

def generate_random_word(length):
    word = ""
    available_characters = string.ascii_lowercase + " "
    for _ in range(length):
        word += random.choice(available_characters)
    return word

def calculate_fitness(word):
    fitness = sum(1 for a, b in zip(word, TARGET_WORD) if a == b)
    return fitness / len(TARGET_WORD)

def mutate_word(word, generation):
    mutated_word = list(word)
    for i in range(len(mutated_word)):
        if random.random() < (MUTATION_RATE / (generation + 1)):
            mutated_word[i] = random.choice(string.ascii_lowercase + " ")
    return ''.join(mutated_word)

def select_parent(population):
    return random.choices(population, weights=[calculate_fitness(word) for word in population])[0]

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def generate_initial_population(size):
    population = []
    for _ in range(size):
        word = generate_random_word(len(TARGET_WORD))
        population.append(word)
    return population

def genetic_algorithm():
    population = generate_initial_population(POPULATION_SIZE)

    best_word = ""
    best_fitness = 0

    for generation in range(NUM_GENERATIONS):
        fitness_scores = [calculate_fitness(word) for word in population]
        best_word = population[fitness_scores.index(max(fitness_scores))]
        best_fitness = max(fitness_scores)

        if best_word == TARGET_WORD:
            print("Target word '{}' found in generation {}!".format(TARGET_WORD, generation))
            return

        new_population = [best_word]

        while len(new_population) < POPULATION_SIZE:
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = crossover(parent1, parent2)
            child_fitness = calculate_fitness(child)
            mutated_child = mutate_word(child, generation)
            new_population.append(mutated_child)
            # if child_fitness < best_fitness:
            #     mutated_child = mutate_word(child, generation)
            #     new_population.append(mutated_child)
            # else:
            #     new_population.append(child)

        population = new_population
        population.sort(key=lambda x: calculate_fitness(x))
        population = population[:POPULATION_SIZE]

        # Print progress after each iteration
        print("Generation: {}, Best Word: {}, Fitness: {}".format(generation, best_word, best_fitness))

    print("Target word '{}' not found after {} generations.".format(TARGET_WORD, NUM_GENERATIONS))

# Run the genetic algorithm
genetic_algorithm()
