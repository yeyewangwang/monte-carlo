import numpy as np

CUBIC_TC = 4.5 # in epsilon/k source: Shroeder p.352

SQUARE_TC = 2.27 # in epsilon/k errors: Shroader p:353


def specific_heat(energy_arr, temp, num_particles):
    return (np.array(
        energy_arr).std() / temp)** 2 / num_particles


def susceptibility(mag_arr, temp, num_particles):
    return (np.array(
        mag_arr).std())** 2 / (temp * num_particles)