import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Genetic Algorithm parameters
population_size = 10
num_generations = 100
mutation_rate = 0.01

# Load the target image
target_image = Image.open('Picture1.png').convert('RGB')
target_array = np.array(target_image)

# Image dimensions
image_height, image_width, num_channels = target_array.shape

# Function to calculate the fitness of an image
def calculate_fitness(image):
    # Calculate fitness based on the difference between the target image and the current image
    fitness = np.sum(np.abs(target_array - image))
    return fitness

# Function to generate a random image
def generate_random_image():
    return np.random.randint(0, 256, size=(image_height, image_width, num_channels))

# Function to perform selection based on tournament selection
def selection(population, fitness):
    tournament_size = 5
    selected_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitness = fitness[selected_indices]
    best_index = selected_indices[np.argmin(tournament_fitness)]
    return population[best_index]

# Function to perform crossover
def crossover(parent1, parent2):
    # Perform a simple average crossover
    child = (parent1 + parent2) // 2
    return child

# Function to perform mutation
def mutate(image):
    mutated_image = image.copy()
    for i in range(image_height):
        for j in range(image_width):
            if np.random.rand() < mutation_rate:
                mutated_image[i, j] = np.random.randint(0, 256, size=num_channels)
    return mutated_image

# Function to plot the best image at each generation
def plot_best_image(best_image):
    plt.imshow(best_image)
    plt.axis('off')
    plt.title('Best Image')
    plt.show()

# Main Genetic Algorithm loop
population = [generate_random_image() for _ in range(population_size)]
fitness = np.array([calculate_fitness(image) for image in population])
best_fitness = np.min(fitness)
best_image = population[np.argmin(fitness)]

for generation in range(num_generations):
    new_population = []

    for _ in range(population_size // 2):
        parent1 = selection(population, fitness)
        parent2 = selection(population, fitness)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_population.extend([child])

    population = np.array(new_population)
    fitness = np.array([calculate_fitness(image) for image in population])

    if np.min(fitness) < best_fitness:
        best_fitness = np.min(fitness)
        best_image = population[np.argmin(fitness)]

    # Display the best image at each generation
    plot_best_image(best_image)
