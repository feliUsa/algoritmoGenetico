import numpy as np
from PIL import Image, ImageDraw
from numba import jit
import random

# Cargar imagen objetivo y convertirla a escala de grises
def load_target_image(path):
    img = Image.open(path).convert('L')  # Convertir a escala de grises
    return np.array(img)

target_image = load_target_image('./images/firma.png')
height, width = target_image.shape

# Función de fitness: calcula la diferencia total de píxeles
@jit
def fitness_function(candidate):
    return np.sum(np.abs(candidate - target_image))

# Inicializar población
def initialize_population(pop_size, height, width):
    return [np.random.randint(0, 256, (height, width), dtype=np.uint8) for _ in range(pop_size)]

# Selección por torneo
def tournament_selection(population, fitness_scores, k=3):
    selected = random.sample(range(len(population)), k)
    selected_fitness = [fitness_scores[i] for i in selected]
    return population[selected[np.argmin(selected_fitness)]]

# Cruce de un punto
def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(0, height * width - 1)
    child = np.copy(parent1)
    child.ravel()[crossover_point:] = parent2.ravel()[crossover_point:]
    return child

# Mutación scramble: altera aleatoriamente una sección de la imagen
def scramble_mutation(candidate, mutation_rate=0.01):
    if random.random() < mutation_rate:
        start_x = random.randint(0, height - 10)
        start_y = random.randint(0, width - 10)
        section = candidate[start_x:start_x + 10, start_y:start_y + 10].ravel()
        np.random.shuffle(section)
        candidate[start_x:start_x + 10, start_y:start_y + 10] = section.reshape((10, 10))
    return candidate

# Crear la función principal del algoritmo genético
def genetic_algorithm(target_image, pop_size=50, generations=500, mutation_rate=0.01):
    population = initialize_population(pop_size, height, width)
    best_individual = None
    best_fitness = float('inf')

    # Crear lista para almacenar cada generación para el GIF
    frames = []

    for gen in range(generations):
        fitness_scores = [fitness_function(ind) for ind in population]
        
        # Elitismo: guardar el mejor individuo de la generación
        current_best = population[np.argmin(fitness_scores)]
        current_best_fitness = min(fitness_scores)
        
        if current_best_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness

        # Mostrar progreso
        print(f"Generación {gen+1}, Mejor fitness: {best_fitness}")

        # Guardar la mejor imagen de la generación para el GIF
        img = Image.fromarray(best_individual)
        frames.append(img)

        # Nueva generación
        new_population = [best_individual]  # Elitismo
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child = one_point_crossover(parent1, parent2)
            child = scramble_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

    # Guardar el GIF
    frames[0].save('evolution.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)
    print("GIF guardado en evolution.gif")

# Ejecutar el algoritmo genético
genetic_algorithm(target_image, pop_size=50, generations=500, mutation_rate=0.01)
