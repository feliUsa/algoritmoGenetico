import numpy as np

class Elitismo:
    @staticmethod
    def aplicar(poblacion, tamano_elite):
        mejores_indices = np.argpartition([ind.fitness for ind in poblacion.individuos], -tamano_elite)[-tamano_elite:]
        return [poblacion.individuos[i] for i in mejores_indices]