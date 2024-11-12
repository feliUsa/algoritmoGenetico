import numpy as np
from numba import njit

@njit
def fitness_func(cromosoma, maximize=True):
    fitness_value = np.sum(cromosoma)
    if not maximize:
        fitness_value *= -1
    return fitness_value
