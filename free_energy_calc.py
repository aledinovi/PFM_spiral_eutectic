# Written by Alex DiNovi
# Equations for free energies come from doi:10.1016/j.commatsci.2017.01.007

import numpy as np
import math
import sys
import matplotlib.pyplot as plt


def calculate_free_energy(u: 'numpy.ndarray[float]', c:'numpy.ndarray[float]'):
    """
    Calculates the free energies for a 2d phase field model for dislocation dynamics with solid-liquid phase transition.

    Args:
        u: ndarray[float]
            Solid phase variable.
        c: ndarray[float]
            Liquid phase variable.

    Returns:
        free_energy: values of free energies at different points
    """
    # Constants to be defined
    K = 1
    undercooling = 0.1

    ##### Begining of operative code #####
    free_energy = {}

    # Initialize arrays for solid and liquid free energies
    solid_free_energy = np.zeros_like(u)
    liquid_free_energy = np.zeros_like(c)

    # Loop over array indices to calculate solid and liquid free energies
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            solid_free_energy[i, j] = ((1.0/8.0)*((u[i,j]**2 - 1)**2)) + (c[i,j]*np.log(np.abs(c[i,j])) - c[i,j])-(np.log(K)*c[i,j]) - (undercooling)
            print("Log c: " + str(np.log(np.abs(c[i,j])))+ ", and c:" + str(c[i,j]))

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            liquid_free_energy[i, j] = (0.5*(u[i,j]**2)) + ((c[i,j]*np.log(np.abs(c[i,j]))-c[i,j]))

    # Calculate the free-energy density at each point
    free_energy_density = np.zeros_like(u)
    for i in range(free_energy_density.shape[0]):
        for j in range(free_energy_density.shape[1]):
            h_phi = (abs(u[i,j])**(2))*(3-(2*abs(u[i,j])))
            free_energy_density[i, j] = (h_phi*solid_free_energy[i,j]) + ((1-h_phi)*liquid_free_energy[i,j])



    return free_energy_density


