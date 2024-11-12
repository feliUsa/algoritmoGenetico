import numpy as np
from numba import njit

class Seleccion:
    @staticmethod
    def torneo(poblacion, k=3):
        fitness_values = np.array([ind.fitness for ind in poblacion.individuos], dtype=np.float64)
        seleccionados_indices = torneo_seleccion(fitness_values, k)
        return [poblacion.individuos[i] for i in seleccionados_indices]

@njit
def torneo_seleccion(fitness_values, k):
    num_individuos = len(fitness_values)
    seleccionados_indices = []
    for _ in range(num_individuos):
        competidores = np.random.choice(num_individuos, k, replace=False)
        mejor = competidores[np.argmax(fitness_values[competidores])]
        seleccionados_indices.append(mejor)
    return seleccionados_indices