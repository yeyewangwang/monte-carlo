"""
Citation: This is a copy of Professor Plumb's code, with
modifications.
"""

import numpy as np


def mc_step(lattice,
            state,
            T, h, acceptedMoves, energy, magnetization, rng=None):
    lattice_size = lattice.num_sites
    # Best practice is to use a random number generator
    # instance
    if rng:
        randomPositions = lattice_size * rng.random(
            lattice_size)
        randomArray = rng.random(lattice_size)
    else:
        randomPositions = lattice_size * np.random.random(lattice_size)
        randomArray = np.random.random(lattice_size)
    # Track random number generator progress
    rng_progress = 2 * lattice_size

    # The index of energy in lattice.
    for k in range(lattice_size):
        i = int(randomPositions[k])

        neighbor_energies = 0
        for n in lattice.nearest_neighbors(i):
            neighbor_energies += state[n]
        dE = state[i] * 2 * neighbor_energies + state[i] * h

        # TODO: support only one dimension
        if dE <= 0 or np.exp(-dE / T) > randomArray[k]:
            acceptedMoves += 1
            # Flip the spin
            flipped_spin = -state[i]
            state[i] = flipped_spin
            energy, magnetization = energy + dE, \
                                    magnetization + 2 * flipped_spin
    return rng_progress, acceptedMoves, energy, magnetization


def mc_step_dist(lattice, T, h,
                 acceptedMoves, energy, magnetization,
                 state=None, region=None, smcArrProps=None
                 ):
    """
    TODO: update with random number generator
    Ensuring correctness for Ray: the key is to make sure
    the Ray object reference is one of the parameters,
    rather than being contained in a different Python
    object, because Ray could be doing its own syntactical
    analysis.
    """
    lattice_size = len(region)
    # print(lattice_size)

    randomPositions = lattice_size * np.random.random(lattice_size)
    randomArray = np.random.random(lattice_size)

    # The index of energy in lattice.
    for k in range(lattice_size):
        i = int(randomPositions[k])

        if i in smcArrProps.multicore_imap:
            i, i_shared = smcArrProps.multicore_imap[i], True
        else:
            i, i_shared = i, False

        dE = 0
        for j in lattice.nearest_neighbors(i):
            dE += smcArrProps.get(j)

        if i_shared:
            val_i = state.buf[i]
        else:
            val_i = smcArrProps.arr[i]
        dE = 2 * val_i * dE + h * val_i

        # TODO: support only one dimension
        if dE <= 0 or np.exp(-dE / T) > randomArray[k]:
            acceptedMoves += 1
            val_i_updated = -val_i

            # Flip the spin
            if i_shared:
                state.buf[i] = -1 * val_i_updated
            else:
                smcArrProps.arr[i] = -val_i_updated

            energy += dE
            magnetization += 2 * val_i_updated
    return acceptedMoves, energy, magnetization

