import numpy as np
from Graficos import graficos
from AlgoritmoGenetico import AlgoritmoGenetico


#img = np.array('./images/firma.png')
stats = graficos()

imgArr, image = stats.leerImagen('/home/daniel/Universidad/octavoSemestre/vida/algoritmosGeneticos/images/firma.png')
frames = []

imagenArreglada = stats.escalaGrisesConvert(image)
vector_objetivo = np.array(imagenArreglada).flatten()

genetic = AlgoritmoGenetico(
    tam_poblacion=10, 
    prob_mutacion=0.0001,
    prob_cruce=0.9,
    tasa_elitismo=0.1,
    tam_cromosoma=vector_objetivo.size,
    fitness_objetivo=1
)


genetic.ejecutar_con_gif(stats, frames, vector_objetivo, imagenArreglada.shape)
stats.crear_gif_imagen(frames, interval=200, imageName="firma.png", n_frames_to_save=500)