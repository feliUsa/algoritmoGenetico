# Graficos.py

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.animation as animation

class graficos:
    def __init__(self):
        self.registro_fitness = []

    def registrar(self, valores_fitness):
        """Registra los valores de fitness de cada generación."""
        self.registro_fitness.append(valores_fitness)

    def graficar_evolucion_fitness(self):
        """Grafica la evolución del mejor, promedio y peor fitness."""
        mejores = [np.max(g) for g in self.registro_fitness]
        promedios = [np.mean(g) for g in self.registro_fitness]
        peores = [np.min(g) for g in self.registro_fitness]

        plt.plot(mejores, label="Mejor Fitness")
        plt.plot(promedios, label="Fitness Promedio")
        plt.plot(peores, label="Peor Fitness")
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.legend()
        plt.title("Evolución del Fitness")
        plt.show()

    def graficar_boxplot(self):
        """Muestra un boxplot de los valores de fitness por generación."""
        plt.boxplot(self.registro_fitness, showmeans=True)
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.title("Distribución del Fitness por Generación")
        plt.show()

    def graficar_varianza(self):
        """Grafica la varianza del fitness en cada generación."""
        varianzas = [np.var(g) for g in self.registro_fitness]
        plt.plot(varianzas, label="Varianza del Fitness")
        plt.xlabel("Generación")
        plt.ylabel("Varianza")
        plt.title("Varianza del Fitness por Generación")
        plt.legend()
        plt.show()
        
    def crear_gif(self, frames, nombre_archivo="evolucion.gif", duracion=200, nuevo_tamano=(300, 400)):
        """Crea un GIF a partir de los fotogramas almacenados y ajusta su tamaño."""
        imagenes = [
            Image.fromarray((frame * 255).astype(np.uint8)).resize(nuevo_tamano, Image.NEAREST)
            for frame in frames
        ]
        imagenes[0].save(
            nombre_archivo,
            save_all=True,
            append_images=imagenes[1:],
            duration=duracion,
            loop=2
        )
        print(f"GIF guardado como {nombre_archivo}")
        
        
    def crear_gif_imagen(self, frames, interval=200, imageName="imagen.gif",n_frames_to_save=15):
        fig, ax = plt.subplots()
        ims = []
        
        # Selecciona solo algunos frames
        selected_frames = frames[::n_frames_to_save]
        
        for frame in selected_frames:
            im = ax.imshow(frame, cmap='gray', animated=True)
            ims.append([im])
            
        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
        ani.save(imageName, writer="imagemagick")

        
    
    def leer_y_transformar_imagen(self, ruta_img, threshold=128):
        """
        Lee una imagen desde una ruta, la convierte a blanco y negro, y la retorna como un vector unidimensional.

        Args:
            ruta_img (str): Ruta de la imagen a procesar.
            threshold (int): Umbral para binarizar la imagen (0-255).

        Returns:
            np.ndarray: Imagen binarizada en una sola dimensión.
            tuple: Dimensiones originales de la imagen (alto, ancho).
        """
        # Leer la imagen desde la ruta
        image = Image.open(ruta_img)
        print(f"Tamaño original de la imagen: {image.size}")

        # Convertir a escala de grises
        image = image.convert('L')

        # Binarizar la imagen
        image = image.point(lambda p: p > threshold and 1 or 0)

        # Convertir a un arreglo de NumPy
        image_array = np.array(image)
        print(f"Dimensiones de la imagen binarizada: {image_array.shape}")

        # Retornar la imagen como un vector unidimensional y sus dimensiones originales
        return image_array.flatten(), image_array.shape
    
    def guardar_imagen_individuo(self, individuo, tamano_imagen, generacion, directorio=".", nombre_base="gen"):
        """
        Guarda una representación de la imagen generada a partir del mejor individuo.
        
        Args:
            individuo (numpy.ndarray): Cromosoma del mejor individuo.
            tamano_imagen (tuple): Dimensiones originales de la imagen.
            generacion (int): Número de la generación actual.
            directorio (str): Directorio donde se guardarán las imágenes.
            nombre_base (str): Nombre base para los archivos.
        """
        # Convertir el cromosoma a su forma de imagen
        imagen = individuo.reshape(tamano_imagen)
        
        # Normalizar valores si es necesario (0 o 1 -> 0 a 255)
        imagen = (imagen * 255).astype(np.uint8)
        
        # Crear y guardar la imagen
        imagen_pil = Image.fromarray(imagen)
        nombre_archivo = f"{directorio}/{nombre_base}_gen{generacion}.png"
        imagen_pil.save(nombre_archivo)
        print(f"Imagen guardada: {nombre_archivo}")