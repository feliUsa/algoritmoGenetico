import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

# Clase del Algoritmo Genético
class AlgoritmoGenetico:
    def __init__(self, tam_poblacion, prob_mutacion, prob_cruce, tasa_elitismo, tam_cromosoma, fitness_objetivo):
        self.tam_poblacion = tam_poblacion
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.tasa_elitismo = tasa_elitismo
        self.tam_cromosoma = tam_cromosoma
        self.fitness_objetivo = fitness_objetivo
        self.poblacion = self._inicializar_poblacion()
        self.valores_fitness = np.zeros(self.tam_poblacion)
        self.generacion = 0

    def _inicializar_poblacion(self):
        """Crea la población inicial con valores binarios aleatorios."""
        return np.random.randint(2, size=(self.tam_poblacion, self.tam_cromosoma))

    def _calcular_fitness(self, individuo, objetivo):
        """Calcula el fitness como la distancia inversa de Hamming."""
        return self.tam_cromosoma - np.sum(individuo != objetivo)

    def evaluar_poblacion(self, objetivo):
        """Calcula el fitness de cada individuo en la población."""
        for i in range(self.tam_poblacion):
            self.valores_fitness[i] = self._calcular_fitness(self.poblacion[i], objetivo)

    @staticmethod
    @njit
    def seleccion_torneo(valores_fitness, tamanio_torneo=3):
        """Selecciona individuos usando el método de torneo."""
        num_individuos = len(valores_fitness)
        indices_seleccionados = np.empty(num_individuos, dtype=np.int32)
        for i in range(num_individuos):
            participantes = np.random.choice(num_individuos, tamanio_torneo, replace=False)
            ganador = participantes[np.argmax(valores_fitness[participantes])]
            indices_seleccionados[i] = ganador
        return indices_seleccionados

    @staticmethod
    @njit
    def cruce_punto_unico(cromosoma1, cromosoma2):
        """Realiza un cruce de punto único entre dos cromosomas."""
        punto_corte = np.random.randint(1, len(cromosoma1))
        hijo1 = np.concatenate((cromosoma1[:punto_corte], cromosoma2[punto_corte:]))
        hijo2 = np.concatenate((cromosoma2[:punto_corte], cromosoma1[punto_corte:]))
        return hijo1, hijo2

    @staticmethod
    @njit
    def mutacion_scramble(cromosoma, prob_mutacion):
        """Aplica la mutación tipo scramble sobre un segmento del cromosoma."""
        if np.random.rand() < prob_mutacion:
            inicio, fin = sorted(np.random.randint(0, len(cromosoma), 2))
            np.random.shuffle(cromosoma[inicio:fin])
        return cromosoma

    @staticmethod
    def aplicar_elitismo(poblacion, fitness, tasa_elitismo):
        """Selecciona a los mejores individuos de la población según su fitness."""
        num_elite = int(tasa_elitismo * len(poblacion))
        indices_elite = np.argsort(fitness)[-num_elite:]
        return poblacion[indices_elite]

    def ejecutar(self, estadisticas, target_array):
        """Corre el algoritmo sin generación de GIF."""
        sin_mejora = 0
        generaciones_max_estancadas = 50
        mejor_fitness_anterior = 0

        while True:
            self.evaluar_poblacion(target_array)
            estadisticas.registrar(self.valores_fitness)

            # Selección
            padres_indices = self.seleccion_torneo(self.valores_fitness)
            nueva_poblacion = []
            for i in range(0, self.tam_poblacion, 2):
                padre1, padre2 = self.poblacion[padres_indices[i]], self.poblacion[padres_indices[i + 1]]
                if np.random.rand() < self.prob_cruce:
                    hijo1, hijo2 = self.cruce_punto_unico(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1, padre2
                nueva_poblacion.extend([hijo1, hijo2])

            # Mutación
            nueva_poblacion = np.array([self.mutacion_scramble(ind, self.prob_mutacion) for ind in nueva_poblacion])

            # Elitismo
            elite = self.aplicar_elitismo(self.poblacion, self.valores_fitness, self.tasa_elitismo)
            self.poblacion = np.vstack((elite, nueva_poblacion[:self.tam_poblacion - len(elite)]))

            # Evaluar nueva población
            mejor_fitness = np.max(self.valores_fitness)
            if mejor_fitness == self.fitness_objetivo:
                print(f"¡Objetivo alcanzado en la generación {self.generacion}!")
                break

            if mejor_fitness == mejor_fitness_anterior:
                sin_mejora += 1
            else:
                sin_mejora = 0
            mejor_fitness_anterior = mejor_fitness

            if sin_mejora >= generaciones_max_estancadas:
                print("Estancamiento detectado. Deteniendo el algoritmo.")
                break

            self.generacion += 1

    def ejecutar_con_gif(self, estadisticas, frames, objetivo, tamano_imagen):
        """Corre el algoritmo con generación de GIF."""
        while True:
            self.evaluar_poblacion(objetivo)
            estadisticas.registrar(self.valores_fitness)

            padres_indices = self.seleccion_torneo(self.valores_fitness)
            nueva_poblacion = []
            for i in range(0, self.tam_poblacion, 2):
                padre1, padre2 = self.poblacion[padres_indices[i]], self.poblacion[padres_indices[i + 1]]
                if np.random.rand() < self.prob_cruce:
                    hijo1, hijo2 = self.cruce_punto_unico(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1, padre2
                nueva_poblacion.extend([hijo1, hijo2])

            nueva_poblacion = np.array([self.mutacion_scramble(ind, self.prob_mutacion) for ind in nueva_poblacion])
            elite = self.aplicar_elitismo(self.poblacion, self.valores_fitness, self.tasa_elitismo)
            self.poblacion = np.vstack((elite, nueva_poblacion[:self.tam_poblacion - len(elite)]))

            mejor_individuo = self.poblacion[np.argmax(self.valores_fitness)]
            frames.append(mejor_individuo.reshape(tamano_imagen))

            if np.max(self.valores_fitness) == self.fitness_objetivo or self.generacion >= 7000:
                print(f"Generación {self.generacion}: Fin del proceso.")
                break

            self.generacion += 1
