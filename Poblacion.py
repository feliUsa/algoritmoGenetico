import numpy as np
from Individuo import Individuo
from numba import njit, prange

class Poblacion:
    def __init__(self, size, cromosoma_size, fitness_func):
        self.individuos = [Individuo(cromosoma_size) for _ in range(size)]
        self.fitness_func = fitness_func

    def calcular_fitness(self):
        cromosomas = np.array([ind.cromosoma for ind in self.individuos])
        fitness_values = calcular_fitness_poblacion(cromosomas)

        for i, individuo in enumerate(self.individuos):
            individuo.fitness = fitness_values[i]

    def obtener_estadisticas_fitness(self):
        fitness_values = [ind.fitness for ind in self.individuos]
        return {
            'promedio': np.mean(fitness_values),
            'max': np.max(fitness_values),
            'min': np.min(fitness_values),
            'varianza': np.var(fitness_values)
        }

@njit(parallel=True)
def calcular_fitness_poblacion(cromosomas):
    fitness_values = np.zeros(cromosomas.shape[0])
    for i in prange(cromosomas.shape[0]):
        fitness_values[i] = np.sum(cromosomas[i])
    return fitness_values
