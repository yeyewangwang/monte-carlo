from pylab import *  # plotting library
import numpy as np


def plot_lattice(lattice,
                 ising_states,
                 style="tartan",
                 indices=False,
                 ax=None,
                 fig=None):
    positions = lattice.data.positions
    spin_directions = ising_states
    upspins = positions[spin_directions == 1]
    downspins = positions[spin_directions == -1]

    if fig is None:
        fig = plt.figure()
        if positions.shape[1] >= 3:
            ax = fig.add_subplot(projection='3d')
            # Because 'equal' is not implemented in 3d, we have
            # to do with "3d".
            ax.set_aspect('auto')
        else:
            ax = fig.add_subplot()
            ax.set_aspect('equal')

    # Citation: code snippet based on Dylan Jones
    # https://github.com/dylanljones/lattpy/blob/master/lattpy/plotting.py
    # The author owes latte to the original author
    upspins = np.atleast_2d(upspins)
    # Fix 1D case
    if upspins.shape[1] == 1:
        upspins = np.hstack(
            (upspins, np.zeros((upspins.shape[0], 1))))

    # Citation: tartan color scheme from
    # https://www.schemecolor.com/orange-deep-navy.php

    print(*upspins.T)
    ax.scatter(*upspins.T, s=5, marker='^', c="#0B0055")

    downspins = np.atleast_2d(downspins)
    # Fix 1D case
    if downspins.shape[1] == 1:
        downspins = \
            np.hstack((downspins, np.zeros((downspins.shape[0], 1))))

    scat = ax.scatter(*downspins.T, s=5, marker='v',
                      c="#F86302")
    # Manualy update data-limits
    # ax.ignore_existing_data_limits = True
    datalim = scat.get_datalim(ax.transData)
    ax.update_datalim(datalim)

    if indices:
        for i in range(
                lattice.data.positions.shape[
                    0]):
            # print(i)
            x_pos, y_pos = \
            lattice.data.positions[i][:2]
            ax.text(x_pos, y_pos, i, horizontalalignment='center')

    if fig == None:
        plt.show()
    return ax