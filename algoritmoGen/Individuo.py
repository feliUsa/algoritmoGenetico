import numpy as np

class Individuo:
    def __init__(self, cromosoma_size):
        self.cromosoma = np.random.choice([0, 1], size=cromosoma_size).astype(np.bool_)
        self.fitness = 0

    def calcular_fitness(self, fitness_func):
        # Calcular y almacenar el valor de fitness
        self.fitness = fitness_func(self.cromosoma)