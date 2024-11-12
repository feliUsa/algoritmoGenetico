import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Poblacion import Poblacion
from Elitismo import Elitismo
from fitness import fitness_func
import time

class AlgoritmoGenetico:
    def __init__(self, poblacion_size, cromosoma_size, tasa_cruce, tasa_mutacion, num_generaciones,
                 operador_seleccion, operador_cruce, operador_mutacion, tamano_elite=0.1, maximize=True, tolerancia_generaciones=10):
        self.poblacion = Poblacion(poblacion_size, cromosoma_size, lambda cromosoma: fitness_func(cromosoma, maximize))
        self.tasa_cruce = tasa_cruce
        self.tasa_mutacion = tasa_mutacion
        self.num_generaciones = num_generaciones
        self.operador_seleccion = operador_seleccion
        self.operador_cruce = operador_cruce
        self.operador_mutacion = operador_mutacion
        self.tamano_elite = tamano_elite
        self.maximize = maximize
        self.tolerancia_generaciones = tolerancia_generaciones

        # Variables para almacenar datos de fitness
        self.fitness_promedio = []
        self.fitness_max = []
        self.fitness_min = []
        self.fitness_por_generacion = []
        self.dataset = []
        self.tiempo_acumulado = 0

    def ejecutar(self):
        self.poblacion.calcular_fitness()
        generaciones_estables = 0
        tiempo_inicio_total = time.time()

        for gen in range(self.num_generaciones):
            tiempo_inicio = time.time()
            print(f"Generacion {gen}")
            num_elite = max(1, int(len(self.poblacion.individuos) * self.tamano_elite))

            # Mantener los individuos élite
            elite = Elitismo.aplicar(self.poblacion, num_elite)
            nueva_poblacion = elite.copy()  # Empezamos la nueva población con los élite

            # Seleccionar individuos para cruce y mutación
            seleccionados = self.operador_seleccion(self.poblacion)

            # Cruce
            for i in range(0, len(seleccionados) - 1, 2):
                if np.random.rand() < self.tasa_cruce:
                    hijo1, hijo2 = self.operador_cruce(seleccionados[i], seleccionados[i+1])
                    hijo1.calcular_fitness(self.poblacion.fitness_func)  # Calcular fitness de los hijos directamente
                    hijo2.calcular_fitness(self.poblacion.fitness_func)
                    nueva_poblacion.extend([hijo1, hijo2])
                else:
                    nueva_poblacion.extend([seleccionados[i], seleccionados[i+1]])

            # Mutación solo en individuos nuevos, excluyendo a los élite
            for individuo in nueva_poblacion[num_elite:]:
                if np.random.rand() < self.tasa_mutacion:
                    self.operador_mutacion(individuo)
                    individuo.calcular_fitness(self.poblacion.fitness_func)  # Recalcular fitness solo si fue mutado

            # Actualizar la población con los individuos nuevos
            self.poblacion.individuos = nueva_poblacion[:len(self.poblacion.individuos)]
            
            # Almacenamiento de estadísticas en cada generación
            estadisticas = self.poblacion.obtener_estadisticas_fitness()
            self.fitness_promedio.append(estadisticas['promedio'])
            self.fitness_max.append(estadisticas['max'])
            self.fitness_min.append(estadisticas['min'])
            self.fitness_por_generacion.append([ind.fitness for ind in self.poblacion.individuos])
            
            # Guardar datos de cada individuo en el dataset para el reporte
            for individuo in self.poblacion.individuos:
                self.dataset.append({
                    'generacion': gen,
                    'fitness': individuo.fitness,
                    'fitness_promedio': estadisticas['promedio'],
                    'fitness_max': estadisticas['max'],
                    'fitness_min': estadisticas['min'],
                    'fitness_varianza': estadisticas['varianza'],
                    'tiempo (ms)': self.tiempo_acumulado
                })

            tiempo_fin = time.time()
            self.tiempo_acumulado += (tiempo_fin - tiempo_inicio) * 1000

            # Verificación de estabilidad del fitness
            if len(self.fitness_max) > 1 and self.fitness_max[-1] == self.fitness_max[-2]:
                generaciones_estables += 1
            else:
                generaciones_estables = 0

            if generaciones_estables >= self.tolerancia_generaciones:
                print(f"\nFitness estabilizado durante {self.tolerancia_generaciones} generaciones consecutivas. Deteniendo la ejecución.")
                break
        
        tiempo_total = (time.time() - tiempo_inicio_total) * 1000
        print(f"\nTiempo total de ejecución: {tiempo_total} ms")

    def generar_dataset(self, nombre_archivo="resultado_algoritmo_genetico.csv"):
        df = pd.DataFrame(self.dataset)
        df.to_csv(nombre_archivo, index=False)
        print(f"Dataset guardado como {nombre_archivo}")

    def mostrar_estadisticas_finales(self):
        print("\nEstadísticas Finales:")
        print(f"Fitness Promedio: {self.fitness_promedio[-1]}")
        print(f"Fitness Máximo: {self.fitness_max[-1]}")
        print(f"Fitness Mínimo: {self.fitness_min[-1]}")

    def plot_fitness_evolution(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.fitness_promedio, label='Promedio')
        plt.plot(self.fitness_max, label='Máximo')
        plt.plot(self.fitness_min, label='Mínimo')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.legend()
        plt.title('Evolución del Fitness')
        plt.show()

    def plot_boxplot_fitness(self):
        plt.figure(figsize=(10, 5))
        plt.boxplot(self.fitness_por_generacion, showmeans=True)
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Diagrama de Cajas del Fitness por Generación')
        plt.show()