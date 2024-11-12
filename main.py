from algoritmoGenetico import AlgoritmoGenetico

# Configuración de parámetros
poblacion_size = 500
cromosoma_size = 1200000
tasa_cruce = 0.7
tasa_mutacion = 0.1
num_generaciones = 20
tamano_elite = 0.1
maximize = True

# Crear y ejecutar el algoritmo genético
genetico = AlgoritmoGenetico(
    poblacion_size=poblacion_size,
    cromosoma_size=cromosoma_size,
    tasa_cruce=tasa_cruce,
    tasa_mutacion=tasa_mutacion,
    num_generaciones=num_generaciones,
    tamano_elite=tamano_elite,
    maximize=maximize,
    tolerancia_generaciones=10
)

genetico.ejecutar()
genetico.mostrar_estadisticas_finales()
