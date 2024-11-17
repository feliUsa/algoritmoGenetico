import numpy as np
from AlgoritmoGenetico import AlgoritmoGenetico
from Graficos import graficos


inicial = [
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
]
fotogramas = []
dimensiones_matriz = (4, 3)
vector_objetivo = np.array(inicial).flatten()

stats = graficos()

genetic = AlgoritmoGenetico(
    tam_poblacion=50, 
    prob_mutacion=0.02,
    prob_cruce=0.8,
    tasa_elitismo=0.1,
    tam_cromosoma=12,
    fitness_objetivo=12
)

genetic.ejecutar_con_gif(stats, fotogramas, vector_objetivo, dimensiones_matriz)

stats.crear_gif(fotogramas, nombre_archivo='letraInicial.gif', duracion=200)

# Graficar estadísticas
stats.graficar_evolucion_fitness()
stats.graficar_boxplot()
stats.graficar_varianza()