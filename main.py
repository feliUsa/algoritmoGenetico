from algoritmoGenetico import AlgoritmoGenetico
from Seleccion import Seleccion
from Cruce import Cruce
from Mutacion import Mutacion

# Configuraciones de prueba
genetico = AlgoritmoGenetico(
    poblacion_size=1000,
    cromosoma_size=1000000,
    tasa_cruce=0.7,
    tasa_mutacion=0.5,
    num_generaciones=3000000,
    operador_seleccion=Seleccion.torneo,
    operador_cruce=Cruce.un_punto,
    operador_mutacion=Mutacion.scramble,
    tamano_elite=0.1,
    maximize=True,
    tolerancia_generaciones=10
)

genetico.ejecutar()

genetico.mostrar_estadisticas_finales()
genetico.generar_dataset("resultado_algoritmo_genetico.csv")
genetico.plot_fitness_evolution()
genetico.plot_boxplot_fitness()
