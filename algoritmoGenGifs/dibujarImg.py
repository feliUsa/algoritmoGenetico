# dibujarImg.py

import numpy as np
import matplotlib.pyplot as plt
from AlgoritmoGenetico import AlgoritmoGenetico
from Graficos import graficos

# Inicialización de herramientas gráficas
stats = graficos()

# Ruta de la imagen objetivo
ruta_imagen = 'C:\\Users\\juanp\\OneDrive\\Documentos\\Universidad\\Vida\\algoritmoGenetico\\images\\FirmaNormal.jpg'

# Transformar la imagen en un vector unidimensional y obtener sus dimensiones
vector_objetivo, dimensiones_imagen = stats.leer_y_transformar_imagen(ruta_imagen)

# Mostrar la imagen objetivo
plt.imshow(vector_objetivo.reshape(dimensiones_imagen), cmap='gray')
plt.title("Matriz Objetivo")
plt.axis('off')
plt.show()

# Configuración del algoritmo genético
genetic = AlgoritmoGenetico(
    tam_poblacion=10,
    prob_mutacion=0.0001,
    prob_cruce=0.3,
    tasa_elitismo=0.3,
    tam_cromosoma=len(vector_objetivo),
    objetivo=vector_objetivo
)

# Verificar fitness inicial
fitness_inicial = AlgoritmoGenetico.hamming_distance_fitness(np.array([vector_objetivo]), vector_objetivo)[0]
print(f"Fitness entre la matriz objetivo y sí misma: {fitness_inicial:.4f}")

# Ejecutar el algoritmo genético con generación de imágenes periódicas
genetic.ejecutar(
    max_generaciones=50000,               # Máximo número de generaciones
    intervalo_guardado=50,               # Guardar cada 50 generaciones
    tamano_imagen=dimensiones_imagen     # Tamaño de la imagen (para reconstruir correctamente)
)

# Crear un GIF con las imágenes generadas
frames = []  # Lista para almacenar los fotogramas del GIF

# Leer imágenes generadas y agregarlas a la lista de frames
for gen in range(0, genetic.generacion + 1, 50):  # Cada 50 generaciones
    img_path = f"imagenes/gen_{gen}.png"
    frame = plt.imread(img_path)  # Cargar la imagen
    frames.append(frame)

stats.crear_gif_imagen(
    frames=frames,
    interval=200,                        # Duración entre frames del GIF en milisegundos
    imageName="firma.gif"                # Nombre del GIF resultante
)
