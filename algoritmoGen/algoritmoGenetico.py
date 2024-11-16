import numpy as np
import time
import matplotlib.pyplot as plt
from operadoresGeneticos import aplicar_seleccion, aplicar_cruce, aplicar_mutacion, aplicar_elitismo, aplicar_fitness

class AlgoritmoGenetico:
    def __init__(self, poblacion_size, cromosoma_size, tasa_cruce, tasa_mutacion, num_generaciones, tamano_elite=0.1, maximize=True, tolerancia_generaciones=10):
        self.poblacion_size = poblacion_size
        self.cromosoma_size = cromosoma_size
        self.tasa_cruce = tasa_cruce
        self.tasa_mutacion = tasa_mutacion
        self.num_generaciones = num_generaciones
        self.tamano_elite = int(tamano_elite * poblacion_size)
        self.maximize = maximize
        self.tolerancia_generaciones = tolerancia_generaciones

        # Inicializar población como matriz
        self.poblacion = np.random.choice([0, 1], size=(poblacion_size, cromosoma_size), p=[0.9, 0.1]).astype(np.bool_)
        self.fitness_values = np.array([aplicar_fitness(cromosoma, maximize) for cromosoma in self.poblacion])

        # Variables para almacenar datos de fitness a lo largo de las generaciones
        self.fitness_promedio = []
        self.fitness_max = []
        self.fitness_min = []
        self.varianza_fitness = []
        self.tiempo_acumulado = 0

    def ejecutar(self):
        generaciones_estables = 0
        tiempo_inicio_total = time.time()

        for gen in range(self.num_generaciones):
            tiempo_inicio = time.time()

            # Elitismo
            elite, elite_fitness = aplicar_elitismo(self.poblacion, self.fitness_values, self.tamano_elite)

            # Selección
            seleccion_indices = aplicar_seleccion(self.fitness_values)
            seleccionados = self.poblacion[seleccion_indices]

            # Cruce y creación de nueva población
            nueva_poblacion = np.empty_like(self.poblacion)
            nueva_poblacion[:self.tamano_elite] = elite  # Mantener los élites
            nueva_fitness_values = np.empty(self.poblacion_size)
            nueva_fitness_values[:self.tamano_elite] = elite_fitness

            for i in range(self.tamano_elite, self.poblacion_size, 2):
                if np.random.rand() < self.tasa_cruce and i + 1 < self.poblacion_size:
                    hijo1, hijo2 = aplicar_cruce(seleccionados[i - self.tamano_elite], seleccionados[i + 1 - self.tamano_elite])
                    nueva_poblacion[i] = hijo1
                    nueva_poblacion[i + 1] = hijo2
                    nueva_fitness_values[i] = aplicar_fitness(hijo1, self.maximize)
                    nueva_fitness_values[i + 1] = aplicar_fitness(hijo2, self.maximize)
                else:
                    nueva_poblacion[i] = seleccionados[i - self.tamano_elite]
                    nueva_poblacion[i + 1] = seleccionados[i + 1 - self.tamano_elite]
                    nueva_fitness_values[i] = self.fitness_values[seleccion_indices[i - self.tamano_elite]]
                    nueva_fitness_values[i + 1] = self.fitness_values[seleccion_indices[i + 1 - self.tamano_elite]]

            # Mutación
            for i in range(self.tamano_elite, self.poblacion_size):
                nueva_poblacion[i] = aplicar_mutacion(nueva_poblacion[i], self.tasa_mutacion)
                nueva_fitness_values[i] = aplicar_fitness(nueva_poblacion[i], self.maximize)

            # Actualizar población y fitness
            self.poblacion = nueva_poblacion
            self.fitness_values = nueva_fitness_values

            # Estadísticas de fitness
            promedio_fitness = np.mean(self.fitness_values)
            max_fitness = np.max(self.fitness_values)
            min_fitness = np.min(self.fitness_values)
            varianza_fitness = np.var(self.fitness_values)

            self.fitness_promedio.append(promedio_fitness)
            self.fitness_max.append(max_fitness)
            self.fitness_min.append(min_fitness)
            self.varianza_fitness.append(varianza_fitness)

            # Tiempo acumulado
            tiempo_fin = time.time()
            self.tiempo_acumulado += tiempo_fin - tiempo_inicio

            # Imprimir resultados de la generación
            print(f"Generación: {gen + 1} de {self.num_generaciones}")
            print(f"Fitness promedio: {promedio_fitness}")
            print(f"Fitness máximo: {max_fitness}")
            print(f"Fitness mínimo: {min_fitness}")
            print(f"Tiempo de ejecución: {self.tiempo_acumulado:.2f} s\n")

            # Verificación de estabilidad del fitness
            if len(self.fitness_max) > 1 and self.fitness_max[-1] == self.fitness_max[-2]:
                generaciones_estables += 1
            else:
                generaciones_estables = 0

            if generaciones_estables >= self.tolerancia_generaciones:
                print(f"\nFitness estabilizado durante {self.tolerancia_generaciones} generaciones consecutivas. Deteniendo la ejecución.")
                break

        # Tiempo total
        tiempo_total = time.time() - tiempo_inicio_total
        print(f"Tiempo total de ejecución: {tiempo_total:.2f} s")

    def mostrar_estadisticas_finales(self):
        """Mostrar las estadísticas de fitness y graficar su evolución a lo largo de las generaciones."""
        
        # Imprimir estadísticas finales
        print("\nEstadísticas finales de la ejecución:")
        print(f"Fitness máximo en todas las generaciones: {max(self.fitness_max)}")
        print(f"Fitness mínimo en todas las generaciones: {min(self.fitness_min)}")
        print(f"Fitness promedio en todas las generaciones: {np.mean(self.fitness_promedio)}")
        print(f"Varianza de fitness en todas las generaciones: {np.mean(self.varianza_fitness)}")

        # Graficar la evolución del fitness
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_promedio, label="Fitness Promedio", color='blue')
        plt.plot(self.fitness_max, label="Fitness Máximo", color='green')
        plt.plot(self.fitness_min, label="Fitness Mínimo", color='red')
        plt.fill_between(range(len(self.fitness_promedio)), self.fitness_min, self.fitness_max, color='gray', alpha=0.2)
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Evolución del Fitness a lo largo de las generaciones")
        plt.legend()
        plt.show()