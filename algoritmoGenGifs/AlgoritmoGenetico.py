# AlgoritmoGenetico.py

import numpy as np
from numba import njit
import os
from PIL import Image

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
        """Inicializa la población de cromosomas de forma aleatoria."""
        return np.random.randint(2, size=(self.tam_poblacion, self.tam_cromosoma))

    @staticmethod
    @njit
    def hamming_distance_fitness(poblacion, objetivo):
        """Calcula el fitness basado en la distancia de Hamming."""
        fitness = np.array([np.sum(indiv != objetivo) for indiv in poblacion])
        return -fitness  # Negativo porque queremos minimizar la distancia

    @staticmethod
    @njit
    def seleccion_torneo(poblacion, fitness, k=3):
        """Selección por torneo: selecciona padres."""
        tam_poblacion = poblacion.shape[0]
        seleccionados = np.empty_like(poblacion)
        for i in range(tam_poblacion):
            competidores = np.random.choice(tam_poblacion, k, replace=False)
            ganador_idx = competidores[np.argmax(fitness[competidores])]
            seleccionados[i] = poblacion[ganador_idx]
        return seleccionados

    @staticmethod
    @njit
    def cruce_punto_unico(padres, prob_cruce):
        """
        Realiza cruce de un punto entre los padres.
        Si la población es impar, el último individuo se incluye sin cambios.
        """
        tam_poblacion, tam_cromosoma = padres.shape
        hijos = np.empty_like(padres)

        # Procesar en pares
        for i in range(0, tam_poblacion - 1, 2):  # Procesa de dos en dos
            padre1, padre2 = padres[i], padres[i + 1]
            if np.random.rand() < prob_cruce:
                # Generar punto de corte válido
                punto_corte = np.random.randint(1, tam_cromosoma)
                # Realizar el cruce
                hijos[i, :punto_corte] = padre1[:punto_corte]
                hijos[i, punto_corte:] = padre2[punto_corte:]
                hijos[i + 1, :punto_corte] = padre2[:punto_corte]
                hijos[i + 1, punto_corte:] = padre1[punto_corte:]
            else:
                # Si no hay cruce, los hijos son copias de los padres
                hijos[i], hijos[i + 1] = padre1, padre2

        # Si la población es impar, incluye el último individuo sin cambios
        if tam_poblacion % 2 != 0:
            hijos[-1] = padres[-1]

        return hijos


    @staticmethod
    @njit
    def mutacion_flip_bit(hijos, prob_mutacion):
        """Mutación flip-bit."""
        tam_poblacion, tam_cromosoma = hijos.shape
        for i in range(tam_poblacion):
            for j in range(tam_cromosoma):
                if np.random.rand() < prob_mutacion:
                    hijos[i, j] = 1 - hijos[i, j]  # Flip bit
        return hijos

    def aplicar_elitismo(self, hijos):
        """Aplica elitismo para preservar mejores individuos."""
        num_elite = max(1, int(self.tasa_elitismo * self.tam_poblacion))
        elite_indices = np.argsort(self.valores_fitness)[-num_elite:]
        elite = self.poblacion[elite_indices]
        hijos[:num_elite] = elite
        return hijos

    def guardar_imagen_individuo(self, individuo, tamano_imagen, generacion, directorio="imagenes"):
        """
        Guarda el mejor individuo como una imagen en escala de grises.
        
        Args:
            individuo (numpy.ndarray): Cromosoma del mejor individuo.
            tamano_imagen (tuple): Dimensiones (alto, ancho) para convertir el cromosoma en imagen.
            generacion (int): Número de generación actual.
            directorio (str): Carpeta donde se guardará la imagen.
        """
        if not os.path.exists(directorio):
            os.makedirs(directorio)

        # Convertir a escala de grises (0-255)
        matriz = (individuo.reshape(tamano_imagen) * 255).astype(np.uint8)
        imagen = Image.fromarray(matriz, mode="L")
        imagen.save(f"{directorio}/gen_{generacion}.png")
        print(f"Imagen de la generación {generacion} guardada en {directorio}/gen_{generacion}.png")

    def ejecutar(self, max_generaciones=1000, intervalo_guardado=10, tamano_imagen=(10, 10)):
        """
        Ejecuta el algoritmo genético y guarda imágenes cada cierto intervalo.

        Args:
            max_generaciones (int): Número máximo de generaciones a ejecutar.
            intervalo_guardado (int): Intervalo de generaciones para guardar imágenes.
            tamano_imagen (tuple): Dimensiones (alto, ancho) para convertir el cromosoma en imagen.
        """
        for _ in range(max_generaciones):
            self.valores_fitness = self.hamming_distance_fitness(self.poblacion, self.objetivo)
            padres = self.seleccion_torneo(self.poblacion, self.valores_fitness)
            hijos = self.cruce_punto_unico(padres, self.prob_cruce)
            hijos = self.mutacion_flip_bit(hijos, self.prob_mutacion)
            hijos = self.aplicar_elitismo(hijos)
            self.poblacion = hijos

            # Evaluar progreso
            mejor_fitness = np.max(self.valores_fitness)
            print(f"Generación {self.generacion}: Mejor Fitness: {mejor_fitness}")

            # Guardar imagen del mejor individuo en el intervalo especificado
            if self.generacion % intervalo_guardado == 0:
                mejor_individuo = self.poblacion[np.argmax(self.valores_fitness)]
                self.guardar_imagen_individuo(mejor_individuo, tamano_imagen, self.generacion)

            self.generacion += 1

            # Verificar convergencia
            if mejor_fitness == 0:
                print("¡Solución óptima encontrada!")
                mejor_individuo = self.poblacion[np.argmax(self.valores_fitness)]
                self.guardar_imagen_individuo(mejor_individuo, tamano_imagen, self.generacion)
                break

        # Guardar la imagen de la última generación
        print(f"Guardando imagen final de la generación {self.generacion}")
        mejor_individuo = self.poblacion[np.argmax(self.valores_fitness)]
        self.guardar_imagen_individuo(mejor_individuo, tamano_imagen, self.generacion)
