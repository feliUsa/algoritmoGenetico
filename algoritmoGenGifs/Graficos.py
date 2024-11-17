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
        
        
    def crear_gif_imagen(frames, interval=200, imageName="imagen.png",n_frames_to_save=15):
        fig, ax = plt.subplots()
        ims = []
        
        # Selecciona solo algunos frames
        selected_frames = frames[::n_frames_to_save]
        
        for frame in selected_frames:
            im = ax.imshow(frame, cmap='gray', animated=True)
            ims.append([im])
            
        ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True)
        ani.save(imageName, writer="imagemagick")

        
    
    def escalaGrisesConvert(self, image):
        # Convert image to grayscale
        image = image.convert('L')
        # Binarize the image (convert to black and white)
        threshold = 128  # Adjust this value as needed
        image = image.point(lambda p: p > threshold and 1 or 0)

        # Display the binarized image
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

        # Convert the image to a NumPy array
        image_array = np.array(image)
        print(image_array.shape)
        return image_array
    
    
    def leerImagen(self, rutaImg):
        image = Image.open(rutaImg)
        print(image.size)
        image = image.resize((150, 150))  # Ajusta el tamaño según lo necesites
        # Display the image (optional)
        plt.figure()
        plt.imshow(image)
        plt.axis('off') # Hide axes
        plt.show()

        # Convert the image to a NumPy array (if needed)
        image_array = np.array(image)
        return image_array, image



