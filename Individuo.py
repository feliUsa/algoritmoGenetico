import numpy as np

class Individuo:
    def __init__(self, cromosoma_size):
        # Generar un cromosoma con 10% de unos y 90% de ceros
        self.cromosoma = np.random.choice([0, 1], size=cromosoma_size, p=[0.9, 0.1])
        self.fitness = 0

    def calcular_fitness(self, fitness_func):
        self.fitness = fitness_func(self.cromosoma)
