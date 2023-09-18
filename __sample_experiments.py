"""
This software is based on Luis Sena's code on Medium
Link: https://luis-sena.medium.com/sharing-big-numpy-arrays-
across-python-processes-abf0dc2a0ab2

TODO: use this for experiments
"""
import ray
import multiprocessing
import time
import itertools
import numpy as np
from constants import CUBIC_TC, SQUARE_TC
from latt_zoo import hexagon
from model import sweep, Model
from __sample_exp_params import MODELS, NUM_STEPS, EXP_NAME


@ray.remote
def run(T, h,
        steps=100):
    latt = hexagon(SHAPE)
    # Set periodic boundary condition in the x and y directions.
    latt.set_periodic(axis=[0, 1])
    states = np.ones(latt.num_sites)
    sweep(latt, T=T, h=h, states_ref=states, steps=steps)
    return states


def run_normal(T,
               h,
               steps=100):
    # TODO: change
    latt = hexagon(SHAPE_4)
    # Set periodic boundary condition in the x and y directions.
    # latt.set_periodic(axis=[0, 1])
    states = np.ones(latt.num_sites)
    # All random numbers are generated with a seed of 1.
    rng = np.random.default_rng(1)
    a, e, m, rng_progress = sweep(latt, T=T, h=h, states_ref=states, steps=steps, rng=rng)
    return states, a, e, m, rng_progress


def run():
    start_time = time.time_ns()
    results, ams, es, ms, rng_ps = [], [], [], [], []

    for m in MODELS[1:]:
        m.init_simulation_state()

        m.load()
        print(m.rng)
        print(m.rng_progress)
        print("rounds", m.rng_progress / m.lattice.num_sites / 2)
        m.roll_rng(m.rng_progress)

        acc, ene, mag = m.sweep(steps=NUM_STEPS, save_freq=2500)

        results.append(m.state)
        ams.append(acc)
        es.append(ene)
        ms.append(mag)
        rng_ps.append(m.rng_progress)
        # m.save(time=False)

    # TODO: change
    res = np.array([[m.temperature for m in MODELS],
                     ams,
                     es, ms]).T
    print(res)
    # TODO: change
    # np.savetxt(EXP_NAME + "_props" + ".csv", res, delimiter=",")

    return results, (time.time_ns() - start_time) / 1_000_000_000

_, t = run()
print(t)