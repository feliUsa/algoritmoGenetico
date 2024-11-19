# dibujarImg.py

import numpy as np
from Graficos import graficos
from AlgoritmoGenetico import AlgoritmoGenetico
import matplotlib.pyplot as plt


#img = np.array('./images/firma.png')
stats = graficos()

# Leer y transformar la imagen a un vector unidimensional y obtener sus dimensiones
ruta_imagen = 'C:\\Users\\juanp\\OneDrive\\Documentos\\Universidad\\Vida\\algoritmoGenetico\\images\\firma.png'
vector_objetivo, dimensiones_imagen = stats.leer_y_transformar_imagen(ruta_imagen)

# Visualizar la imagen de la matriz objetivo
plt.imshow(vector_objetivo.reshape(dimensiones_imagen), cmap='gray')
plt.title("Matriz Objetivo")
plt.axis('off')
plt.show()

genetic = AlgoritmoGenetico(
    tam_poblacion=30, 
    prob_mutacion=0.6,
    prob_cruce=0.3,
    tasa_elitismo=0.1,
    tam_cromosoma=len(vector_objetivo),
    objetivo=vector_objetivo
)

# Calcular fitness entre la matriz objetivo y sí misma
fitness = genetic._calcular_fitness(vector_objetivo)
print(f"Fitness entre la matriz objetivo y sí misma: {fitness}")

genetic.ejecutar_con_gif(stats, dimensiones_imagen, intervalo_guardado=50)

stats.crear_gif_imagen(
    interval=200, 
    imageName="firma.gif", 
    n_frames_to_save=500,
    directorio_guardado="resultados",  
    nombre_base="evolucion"
)