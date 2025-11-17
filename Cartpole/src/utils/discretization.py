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

    low = env.observation_space.low.copy()
    high = env.observation_space.high.copy()

    # Ajustes razonables para las variables con rangos infinitos
    # y limitación de velocidades para estabilidad.
    low[1], high[1] = -3.0, 3.0       # velocidad del carro
    low[3], high[3] = -4.0, 4.0       # velocidad angular

    # Preparar una versión con discretización fina (10x10x10x10)
    low_fine = env.observation_space.low.copy()
    high_fine = env.observation_space.high.copy()

    # Reemplazar infinitos por valores razonables para la versión fina
    low_fine = np.where(np.isinf(low_fine), -5.0, low_fine)
    high_fine = np.where(np.isinf(high_fine), 5.0, high_fine)

    # Mantener límites realistas para velocidades en la versión fina
    low_fine[1], high_fine[1] = -3.0, 3.0
    low_fine[3], high_fine[3] = -4.0, 4.0

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
        ),

        # Discretización fina de 10x10x10x10 = 10_000 estados
        "A_fine": Discretizer(
            bins=[10, 10, 10, 10],
            low=low_fine,
            high=high_fine
        )
    }


def get_discretizer(env, name="A_coarse"):
    """
    Devuelve la instancia `Discretizer` solicitada por su clave.

    Uso:
        d = get_discretizer(env, "A_fine")
        estado = d.discretize(obs)
    """
    discrs = create_discretizers(env)
    try:
        return discrs[name]
    except KeyError:
        raise ValueError(f"Discretización desconocida: {name}. Opciones: {list(discrs.keys())}")


def get_bins(env, name="A_coarse"):
    """
    Devuelve la lista de umbrales (`split_points`) usada internamente
    por el `Discretizer`. Esto es compatible con código que espera
    una lista de arrays para pasar a `np.digitize`.
    """
    d = get_discretizer(env, name)
    return d.split_points
