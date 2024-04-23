from free_energy_calc import *
from jacobian_converge import *

import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import random

import sys
import traceback
import warnings

M = 1
M_tilde = 1
W = 1

def numerical_derivative(f, x):
    """
    Compute the numerical derivative df/dx.

    Parameters:
        f (ndarray): 2D array representing the function values.
        x (ndarray): 2D array representing the corresponding x values.

    Returns:
        ndarray: Numerical derivative df/dx.
    """
    df_dx = np.gradient(f, axis=1) / np.gradient(x, axis=1)
    return df_dx



def gradient_2d(arr):
    """
    Calculates the gradient of a 2D array.

    Parameters:
    arr (ndarray): Input 2D array.

    Returns:
    ndarray, ndarray: Returns two arrays representing the gradient along the first and second dimension respectively.
    """
    grad_x = np.gradient(arr, axis=0)
    grad_y = np.gradient(arr, axis=1)
    return grad_x, grad_y

def divergence_2d(arr):
    """
    Calculates the divergence of a 2D vector field.

    Parameters:
    arr[0] (ndarray): Input array representing the x-component of the vector field.
    arr[1] (ndarray): Input array representing the y-component of the vector field.

    Returns:
    ndarray: Returns an array representing the divergence of the vector field.
    """
    div_x = np.gradient(arr[0], axis=0)
    div_y = np.gradient(arr[1], axis=1)
    return div_x + div_y



def laplacian_2d(arr):
    laplacian = np.zeros_like(arr)
    laplacian += np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4 * arr
    return laplacian


def calculate_phase_field_dislocation_2d(number_iterations=2000, input_file=None, output_file=None):
    grid_number = 26                            # N
    grid_size = 0.01                            # h
    time_step = 5e-3                            # dt
    physical_dimension = grid_number*grid_size  # L

    # Initialize phase field
    u = np.ones((grid_number, grid_number))
    c = np.zeros((grid_number, grid_number))

    scale = 0.001
    noise1 = np.random.normal(0, scale, size=u.shape)
    noise2 = np.random.normal(0, scale, size=c.shape)
    u = u + noise1
    c = c + noise2


    # Iteration begins
    time = 0
    for iteration in range(number_iterations + 1):
        u_1 = np.copy(u)
        u_2 = np.copy(u_1)
        
        c_1 = np.copy(c)
        c_2 = np.copy(c_1)

        for sub_iteration in range(2):
            print("Iteration: " + str(sub_iteration))
            free_energy = calculate_free_energy(u_1, c_1)

            change_c = divergence_2d(M_tilde*gradient_2d(numerical_derivative(free_energy, c)))
            change_u = divergence_2d(M_tilde*gradient_2d(numerical_derivative(free_energy, u)- W*W*laplacian_2d(u)))

            u_2 = np.copy(u_1)
            c_2 = np.copy(c_1)
            u_1 = u - change_u * time_step * 0.5
            c_1 = c - change_c * time_step * 0.5
            delta_u = (np.abs(u_2 - u_1)).max()
            delta_c = (np.abs(c_2 - c_1)).max()
            
            # Math error
            if delta_u == 0. or delta_c == 0.:
                raise ValueError('delta_phi = NaN')
                return
            # Normal convergence
            if delta_u < 0.0001 and delta_c < 0.0001:
                break
            # Failure to converge
            if sub_iteration > 19:
                print('Failed to converge: Reduce time_step (dt)!')
                break
        #gradient = (np.abs(gibbs_free_energy)).max()  
        #g_data[iteration] = gradient

        # Update phase field
        u -= free_energy * time_step
        c -= free_energy * time_step
        time += time_step
        print("Time: " + str(time))
    return u,c

u,c = calculate_phase_field_dislocation_2d(number_iterations=2, input_file=None, output_file=None)
