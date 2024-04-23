# Function that converges some mesh using the jacobian relaxation method
# Does not integrate over timesteps, simply takes a lattice and converges
# it to criterion.

import math
import sys
import matplotlib.pyplot as plt
import numpy as np


def jacobi_relaxation(spatial_vals, SPACING, DISTANCE, TOL, R, MAX_ITER):
    GRID_SIZE_X, GRID_SIZE_Y= my_array.shape
    spatial_vals = np.zeros((GRID_SIZE_X, GRID_SIZE_Y))
    n_iter = 0
    for iterations in range(MAX_ITER):  
        new_vals = np.copy(spatial_vals)
        convergence_checker = 0

        for i in range(1, GRID_SIZE_X-1):

            for j in range(1, GRID_SIZE_Y-1):
                # Calculate new value at location
                old_val = new_vals[i,j]
                new_vals[i,j] = 0.25*(spatial_vals[i+1,j] + spatial_vals[i-1,j] + spatial_vals[i,j+1] + spatial_vals[i,j-1])
                # New max for convergence
                if abs(new_vals[i,j]-old_val) > convergence_checker:
                    convergence_checker = abs(new_vals[i,j]-old_val)

        
        # Check for the convergence
        if (iterations % 10 == 0):
            print (print("Iteration: " + str(iterations)) + ", Convergence criterion: "+ str(convergence_checker))

        if iterations > 5 and convergence_checker < TOL:
            spatial_vals = new_vals
            print("CONVERGED!")
            break

        spatial_vals = new_vals
        n_iter = iterations
    
    return spatial_vals,n_iter