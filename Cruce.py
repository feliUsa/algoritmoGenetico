import numpy as np
from Individuo import Individuo
from numba import njit

class Cruce:
    @staticmethod
    def un_punto(padre1, padre2):
        hijo1_cromosoma, hijo2_cromosoma = Cruce._un_punto_cruce_cromosomas(padre1.cromosoma, padre2.cromosoma)
        hijo1 = Individuo(len(hijo1_cromosoma))
        hijo2 = Individuo(len(hijo2_cromosoma))
        hijo1.cromosoma, hijo2.cromosoma = hijo1_cromosoma, hijo2_cromosoma
        return hijo1, hijo2

    @staticmethod
    @njit  # Quitamos parallel=True
    def _un_punto_cruce_cromosomas(cromosoma1, cromosoma2):
        punto = np.random.randint(1, len(cromosoma1) - 1)
        hijo1_cromosoma = np.concatenate((cromosoma1[:punto], cromosoma2[punto:]))
        hijo2_cromosoma = np.concatenate((cromosoma2[:punto], cromosoma1[punto:]))
        return hijo1_cromosoma, hijo2_cromosoma