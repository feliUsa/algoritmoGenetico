import numpy as np
from numba import njit, prange

@njit
def aplicar_fitness(cromosoma, maximize=True):
    """Calcula el valor de fitness de un cromosoma."""
    fitness_value = np.sum(cromosoma)
    return fitness_value if maximize else -fitness_value

@njit
def aplicar_elitismo(cromosomas, fitness_values, tamano_elite):
    """Selecciona los cromosomas de mejor rendimiento como élite."""
    elite_indices = np.argsort(fitness_values)[-tamano_elite:]  # índices de los mejores
    elite = cromosomas[elite_indices]
    elite_fitness = fitness_values[elite_indices]
    return elite, elite_fitness

@njit
def aplicar_seleccion(fitness_values, k=3):
    """Método de selección por torneo."""
    num_individuos = len(fitness_values)
    seleccionados_indices = np.empty(num_individuos, dtype=np.int32)
    for i in range(num_individuos):
        competidores = np.random.choice(num_individuos, k, replace=False)
        mejor = competidores[np.argmax(fitness_values[competidores])]
        seleccionados_indices[i] = mejor
    return seleccionados_indices

@njit
def aplicar_cruce(cromosoma1, cromosoma2):
    """Cruce de un punto entre dos cromosomas."""
    punto = np.random.randint(1, len(cromosoma1) - 1)
    hijo1 = np.concatenate((cromosoma1[:punto], cromosoma2[punto:]))
    hijo2 = np.concatenate((cromosoma2[:punto], cromosoma1[punto:]))
    return hijo1, hijo2

@njit(parallel=True)
def aplicar_mutacion(cromosoma, tasa_mutacion):
    """Mutación tipo scramble en una sección aleatoria del cromosoma."""
    if np.random.rand() < tasa_mutacion:
        inicio, fin = sorted(np.random.randint(0, len(cromosoma), 2))
        np.random.shuffle(cromosoma[inicio:fin])
    return cromosoma