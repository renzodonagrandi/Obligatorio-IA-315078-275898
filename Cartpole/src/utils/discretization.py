import numpy as np

class Discretizer:
    def __init__(self, bins, low, high):
        """
        bins = lista con número de bins por variable
        low  = límites inferiores del espacio de observación
        high = límites superiores del espacio de observación
        """
        self.bins = bins
        self.low = low
        self.high = high

        # Crear "grid" con los puntos de corte
        self.split_points = [
            np.linspace(low[i], high[i], bins[i] - 1)
            for i in range(len(bins))
        ]

    def discretize(self, obs):
        """
        Convierte una observación continua en un estado discreto (tupla)
        """
        return tuple(
            int(np.digitize(obs[i], self.split_points[i]))
            for i in range(len(obs))
        )


def create_discretizers(env):
    """
    Crea y retorna un diccionario con discretizaciones disponibles.
    """

    low = env.observation_space.low
    high = env.observation_space.high

    low[1], high[1] = -3.0, 3.0       # velocidad del carro
    low[3], high[3] = -4.0, 4.0       # velocidad angular

    return {
        # Discretización de 162 estados
        "A_coarse": Discretizer(
            bins=[3, 3, 6, 3],
            low=low,
            high=high
        ),

        # Discretización de ≈5.145 estados
        "A_5000": Discretizer(
            bins=[7, 7, 15, 7],
            low=low,
            high=high
        )
    }
