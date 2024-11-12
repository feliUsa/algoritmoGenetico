import numpy as np
from numba import njit

class Mutacion:
    @staticmethod
    def scramble(individuo):
        Mutacion._scramble_cromosoma(individuo.cromosoma)

    @staticmethod
    @njit
    def _scramble_cromosoma(cromosoma):
        inicio, fin = sorted(np.random.randint(0, len(cromosoma), 2))
        np.random.shuffle(cromosoma[inicio:fin])
