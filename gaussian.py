from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib.pyplot as plt


# Parámetros para la generación de partículas
N = 1000
N_particles = 100
N_samples_per_particle = N // N_particles  #

# Inicializar listas para almacenar las partículas
particle_X = []
particle_Y = []

for _ in range(N_particles):
    # Generar partícula con make_gaussian_quantiles
    particle_data = make_gaussian_quantiles(mean=None,
                                            cov=0.1,
                                            n_samples=N_samples_per_particle,
                                            n_features=2,
                                            n_classes=2,
                                            shuffle=True,
                                            random_state=None)

    particle_X.append(particle_data[0])
    particle_Y.append(particle_data[1][:, np.newaxis])

# Convertir las listas en arrays
particle_X = np.concatenate(particle_X, axis=0)
particle_Y = np.concatenate(particle_Y, axis=0)

plt.scatter(particle_X[:, 0], particle_X[:, 1], c=particle_Y.flatten(), cmap='viridis', edgecolors='k', marker='o')
plt.title('Partículas Aleatorias')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()
