# AlgoritmoGenetico.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

# Clase del Algoritmo Genético
class AlgoritmoGenetico:
    def __init__(self, tam_poblacion, prob_mutacion, prob_cruce, tasa_elitismo, tam_cromosoma, objetivo):
        self.tam_poblacion = tam_poblacion
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.tasa_elitismo = tasa_elitismo
        self.tam_cromosoma = tam_cromosoma
        self.objetivo = objetivo
        self.poblacion = self._inicializar_poblacion()
        self.valores_fitness = np.zeros(self.tam_poblacion)
        self.generacion = 0

    def _inicializar_poblacion(self):
        """Crea la población inicial con valores binarios aleatorios."""
        return np.random.randint(2, size=(self.tam_poblacion, self.tam_cromosoma))

    def _calcular_fitness(self, individuo):
        """Calcula el fitness como la proporción de bits coincidentes con la matriz objetivo."""
        coincidencias = np.sum(individuo == self.objetivo)  # Número de bits iguales
        return coincidencias / self.tam_cromosoma  # Proporción de coincidencias

    def evaluar_poblacion(self):
        """Calcula el fitness de cada individuo en la población."""
        for i in range(self.tam_poblacion):
            self.valores_fitness[i] = self._calcular_fitness(self.poblacion[i])

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


    def ejecutar_con_gif(self, estadisticas, tamano_imagen, intervalo_guardado=10, max_generaciones =1000, directorio_guardado="imagenes", nombre_base="gen"):
        """Corre el algoritmo con generación de GIF."""
        
        import os
        os.makedirs(directorio_guardado, exist_ok=True)  # Crear el directorio si no existe
        
        while self.generacion < max_generaciones:
            self.evaluar_poblacion()  # No requiere pasar la matriz objetivo
            estadisticas.registrar(self.valores_fitness)
            mejor_fitness = np.max(self.valores_fitness)
            print(f"Generación {self.generacion}: Mejor Fitness = {mejor_fitness}")

            if mejor_fitness >= 0.99:  # Umbral de parada opcional
                print("Condición de parada alcanzada.")
                break

            # Seleccionar la pipol
            padres_indices = self.seleccion_torneo(self.valores_fitness)
            nueva_poblacion = []
            for i in range(0, self.tam_poblacion, 2):
                padre1, padre2 = self.poblacion[padres_indices[i]], self.poblacion[padres_indices[i + 1]]
                if np.random.rand() < self.prob_cruce:
                    hijo1, hijo2 = self.cruce_punto_unico(padre1, padre2)
                else:
                    hijo1, hijo2 = padre1, padre2
                nueva_poblacion.extend([hijo1, hijo2])

            # Mutar la pipol
            nueva_poblacion = np.array([self.mutacion_scramble(ind, self.prob_mutacion) for ind in nueva_poblacion])

            # Elitisar la pipol
            elite = self.aplicar_elitismo(self.poblacion, self.valores_fitness, self.tasa_elitismo)
            self.poblacion = np.vstack((elite, nueva_poblacion[:self.tam_poblacion - len(elite)]))

            # Guardar la imagen si es generación divisible por el intervalo
            if self.generacion % intervalo_guardado == 0:
                mejor_individuo = self.poblacion[np.argmax(self.valores_fitness)]
                estadisticas.guardar_imagen_individuo(mejor_individuo, tamano_imagen, self.generacion, directorio_guardado, nombre_base)

            self.generacion += 1
