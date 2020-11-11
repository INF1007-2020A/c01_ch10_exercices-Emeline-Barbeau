#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
from matplotlib import pyplot as plt
from cmath import polar
from scipy.integrate import quad



# Array présentant 64 valeurs uniformément réparties entre -1.3 et 2.5.
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)
    #return np.linspace(start = -1.3, stop = 2.5, num = 64)
    


# Convertit une liste de coordonnées cartésiennes (x, y) en coordonnées polaires (rayon, angle).
def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return n.array([polar(coord) for coord in cartesian_coordinates])

    """
    result = np.zeros([len(cartesian_coordinates), 2])

    for i in range(len(cartesian_coordinates)):

        r = np.sqrt(cartesian_coordinates[i][0] ** 2 + cartesian_coordinates[i][1] ** 2)
        teta = np.arctan2(cartesian_coordinates[i][1], cartesian_coordinates[i][0])

        polar_coordinate = (r, teta)

        a[i] = polar_coordinate

    return result"""



# Trouve l’index de la valeur la plus proche d’un nombre fournit dans un array.
def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()  # argmin => index de la plus petite valeur
    # return sorted([(i, values[i]) for i in range(values.size)], key= lambda element : abs(element[1] - number))[0][0]



def sinusoidal(x):
    return x**2 * np.sin(1 / x**2) + x


def create_graph(x, y):
    plt.plot(x, y, 'o', markersize = 2.5)  # 'o' => forme du marker qui représentera la fonction  
    plt.legend(['data'], loc = 'best')     # markersize => taille des marker
    plt.show()


def axis_definition(function, start, end, nbr_sample):
    x = np.linspace(-1, 1, num = 250)  # num => nbr d'élément voulu
    y = sinusoidal(x)

    return x, y


# Créer un graphe de y=𝑥^2 sin⁡(1∕𝑥^2 )+𝑥 dans l’intervalle [-1, 1] avec 250 points.
def first_function() -> None:
    create_graph(*axis_definition(sinusoidal, -1, 1, 250))
    # *axi_ => split un tuple (x, y) en mettant chacune des variables dans la fonction create_graph(x, y)
    


def exponential(x):
    return np.exp(-x ** 2)


def integral(x, y) -> tuple:
    return quad(exponential, x, y)  # Pas besoin de donner des paramètre à quad pour la fonction


# Évaluer l’intégrale ∫_(−∞)^∞ 𝑒^(−𝑥^2) 𝑑𝑥. Afficher dans un graphique ∫𝑒^(−𝑥^2) 𝑑𝑥 pour x = [-4, 4].
def second_function():
    a, b = -4, 4  # Bornes de l'intégrale

    x = x = np.linspace(a, b, num = 100)
    y = integral(x)



if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici

    #print(linear_values())

    #first_function()
    print(integral(-np.inf, np.inf))
    #second_function()