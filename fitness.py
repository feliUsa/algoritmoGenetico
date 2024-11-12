import numpy as np
from numba import njit

@njit
def fitness_func(cromosoma, maximize=True):
    fitness_value = np.sum(cromosoma)
    return fitness_value if maximize else -fitness_value