from algoritmoGenetico import AlgoritmoGenetico
from Seleccion import Seleccion
from Cruce import Cruce
from Mutacion import Mutacion
import concurrent.futures

experimentos = [
    {"poblacion_size": 1000, "cromosoma_size": 100000, "tasa_cruce": 0.7, "tasa_mutacion": 0.5},
    {"poblacion_size": 1000, "cromosoma_size": 100000, "tasa_cruce": 0.8, "tasa_mutacion": 0.3},
    {"poblacion_size": 1000, "cromosoma_size": 100000, "tasa_cruce": 0.3, "tasa_mutacion": 0.7},
]

def ejecutar_experimento(idx, config):
    print(f"\nEjecutando Experimento {idx+1}")
    genetico = AlgoritmoGenetico(
        poblacion_size=config["poblacion_size"],
        cromosoma_size=config["cromosoma_size"],
        tasa_cruce=config["tasa_cruce"],
        tasa_mutacion=config["tasa_mutacion"],
        num_generaciones=500,
        operador_seleccion=Seleccion.torneo,
        operador_cruce=Cruce.un_punto,
        operador_mutacion=Mutacion.scramble,
        tamano_elite=0.1,
        maximize=True,
        tolerancia_generaciones=10
    )
    genetico.ejecutar()
    nombre_archivo = f"resultado_experimento_{idx+1}.csv"
    genetico.generar_dataset(nombre_archivo)
    genetico.mostrar_estadisticas_finales()
    genetico.plot_fitness_evolution()
    genetico.plot_boxplot_fitness()
    print(f"Experimento {idx+1} finalizado y guardado en {nombre_archivo}")

# Ejecutar los experimentos en paralelo
if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(ejecutar_experimento, idx, config) for idx, config in enumerate(experimentos)]
        concurrent.futures.wait(futures)  # Espera hasta que todos los experimentos hayan terminado
