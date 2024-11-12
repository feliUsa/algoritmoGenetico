import numpy as np
from numba import njit, prange

class Mutacion:
    @staticmethod
    def scramble(individuo):
        Mutacion._scramble_cromosoma(individuo.cromosoma)

    @staticmethod
    @njit(parallel=True)
    def _scramble_cromosoma(cromosoma):
        inicio, fin = sorted(np.random.randint(0, len(cromosoma), 2))
        np.random.shuffle(cromosoma[inicio:fin])