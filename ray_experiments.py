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
from hex_exp_params import TEMPERATURES, LATTICE_STRUCTURE, SHAPE, EXP_NAME, NUM_STEPS
# from exp_params_sq import TEMPERATURES, LATTICE_STRUCTURE, SHAPE, EXP_NAME
from hex_exp_params import TEMPERATURES_4, SHAPE_4, EXP_NAME_4, NUM_STEPS_4
from __sample_exp_params import MODELS

NUM_WORKERS = multiprocessing.cpu_count() - 2
# ray init can take 3 seconds or more to load
ray.init(num_cpus=NUM_WORKERS)

@ray.remote
def run(T, h,
        steps=100):
    latt = hexagon(SHAPE)
    # Set periodic boundary condition in the x and y directions.
    latt.set_periodic(axis=[0, 1])
    states = np.ones(latt.num_sites)
    sweep(latt, T=T, h=h, states_ref=states, steps=steps)
    return states

def run_normal(T, h,
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

def benchmark():
    futures = []

    models = []
    for t in TEMPERATURES_4:
        m = Model(shape=SHAPE_4,
                  lattice_structure=LATTICE_STRUCTURE,
                  temperature=t,
                  build=False)
        models.append(m)

    start_time = time.time_ns()
    # for t in TEMPERATURES:
    #     futures.append(run.remote(T=t, h=0,
    #                               steps=10))
    # results = ray.get(futures)

    results, ams, es, ms, rng_ps = [], [], [], [], []
    # TODO: change
    for t in TEMPERATURES_4:
        # TODO: change
        states, accepted, e, m, rng_progress = run_normal(T=t, h=0,
                                  steps=NUM_STEPS_4)
        results.append(states)
        ams.append(accepted)
        es.append(e)
        ms.append(m)
        rng_ps.append(rng_progress)

    for i, s in enumerate(results):
        # TODO: change
        models[i].set_simulation_state(hexagon(SHAPE_4), s, None, rng_ps[i])
        models[i].save(time=False)

    # TODO: change
    res = np.array([TEMPERATURES_4, ams, es, ms]).T
    print(res)
    # TODO: change
    np.savetxt(EXP_NAME_4 + "_props" + ".csv", res, delimiter=",")

    return results, (time.time_ns() - start_time) / 1_000_000_000

_, t = benchmark()
print(t)