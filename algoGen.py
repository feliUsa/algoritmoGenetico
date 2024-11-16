import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Cargar y procesar la imagen objetivo
ruta_imagen = "./images/firma.png"  # Ruta de la imagen objetivo
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
_, imagen_binarizada = cv2.threshold(imagen, 127, 1, cv2.THRESH_BINARY)
vectori = imagen_binarizada.flatten()
alto, ancho = imagen.shape[:2]

@njit
def calculate_fitness(population):
    fitness_scores = (np.sum(population == vectori, axis=1)) / len(vectori)
    return fitness_scores

@njit
def tournament_selection(population, fitness_scores, num_selections, tournament_size=50):
    selected_population = np.empty((num_selections, population.shape[1]), dtype=np.uint8)
    for i in range(num_selections):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        best_index = tournament_indices[np.argmax(fitness_scores[tournament_indices])]
        selected_population[i] = population[best_index]
    return selected_population

@njit
def one_point_crossover(selected_population):
    population_size, gene_length = selected_population.shape
    new_population = np.empty_like(selected_population)
    for i in range(0, population_size, 2):
        parent1 = selected_population[i]
        parent2 = selected_population[(i + 1) % population_size]
        crossover_point = np.random.randint(1, gene_length)
        new_population[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        new_population[(i + 1) % population_size] = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return new_population

@njit
def scramble_mutation(population, mutation_rate):
    population_size, gene_length = population.shape
    mutated_population = population.copy()
    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            start, end = sorted(np.random.choice(gene_length, 2, replace=False))
            np.random.shuffle(mutated_population[i, start:end])
    return mutated_population

@njit
def elitism(population, fitness_scores, num_elites):
    elite_indices = np.argsort(fitness_scores)[-num_elites:]
    elites = population[elite_indices]
    return elites

def calculate_image_difference(individual, target):
    return np.sum(individual != target)

# Parámetros
population_size = 600
gene_length = vectori.size
num_selections = population_size
num_elites = 50
max_no_improvement = 20
mutation_rate = 0.05
tournament_size = 50

# Inicializar población
population = np.random.randint(2, size=(population_size, gene_length), dtype=np.uint8)

fitness_scores = calculate_fitness(population)
generations = 0
frames = []
previous_max_fitness = np.max(fitness_scores)

# Ejecución del algoritmo genético
while True:
    generations += 1

    # Selección, cruce y mutación
    selected_population = tournament_selection(population, fitness_scores, num_selections, tournament_size)
    new_population = one_point_crossover(selected_population)
    mutated_population = scramble_mutation(new_population, mutation_rate)

    # Elitismo
    elites = elitism(population, fitness_scores, num_elites)
    worst_indices = np.argsort(fitness_scores)[:num_elites]
    mutated_population[worst_indices] = elites

    # Actualizar población y recalcular el fitness
    population = mutated_population
    fitness_scores = calculate_fitness(population)

    # Verificar la diferencia con la imagen objetivo
    best_individual = population[np.argmax(fitness_scores)]
    difference = calculate_image_difference(best_individual, vectori)

    # Guardar frames para el GIF cada 50 generaciones o cuando haya una mejora
    if generations % 50 == 0 or max(fitness_scores) > previous_max_fitness:
        frames.append(best_individual.reshape(alto, ancho))

    # Ajuste de la tasa de mutación adaptativo
    if max(fitness_scores) <= previous_max_fitness:
        mutation_rate = min(1.0, mutation_rate * 1.2)
    else:
        mutation_rate = max(0.05, mutation_rate * 0.9)
    previous_max_fitness = max(fitness_scores)

    # Introducir diversidad si no hay mejora en varias generaciones
    if generations % max_no_improvement == 0:
        random_replace_indices = np.random.choice(population_size, size=10, replace=False)
        population[random_replace_indices] = np.random.randint(2, size=(10, gene_length), dtype=np.uint8)

    # Condición de parada
    if max(fitness_scores) >= 1.0:
        print(f"Proceso detenido en generación {generations} por alcanzar fitness 1.")
        break

    print(f"Generación {generations}, Max Fitness: {max(fitness_scores):.4f}, Diferencia: {difference}, Tasa de mutación: {mutation_rate:.2f}")

# Mostrar la mejor aproximación
plt.imshow(best_individual.reshape(alto, ancho), cmap='gray')
plt.title("Resultado Final")
plt.show()

# Guardar el GIF
frames_as_images = [(frame * 255).astype(np.uint8) for frame in frames]
frames_as_images = [Image.fromarray(frame) for frame in frames_as_images]

frames_as_images[0].save(
    'firmaGenerada.gif', 
    save_all=True, 
    append_images=frames_as_images[1:], 
    loop=0, 
    duration=100
)

print("GIF guardado como 'firmaGenerada.gif'.")
