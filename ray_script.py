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
from dist_ising_lattice import DistIsingLattice
from latt_zoo import square


NUM_WORKERS = multiprocessing.cpu_count() - 2
# ray init can take 3 seconds or more to load
ray.init(num_cpus=NUM_WORKERS)
LATTICE_SHAPE = (100, 100)
lattice = square(LATTICE_SHAPE)
model = DistIsingLattice(lattice=lattice)
CPU_GRID = (2, 3)
model.build_smc(CPU_GRID)

shared_state_prop = model.smc_arr.shared_arr
mc_imap = model.smc_arr.multicore_imap
cpu_allotments = model.cpu_allotments


def mc_step_dist(lattice, T, h,
                 acceptedMoves, energy, magnetization,
                 obj_ref=None, region=None, smcArrProps=None
                 ):
    """
    Ensuring correctness for Ray: the key is to make sure the Ray
    object reference is one of the parameters, rather than
    being contained in a different Python object, because
    Ray could be doing its own syntactical analysis.
    """
    lattice_size = len(region)
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

        val_i = obj_ref[i] if i_shared else smcArrProps.arr[i]
        dE = 2 * val_i * dE + h * val_i

        # TODO: support only one dimension
        if dE <= 0 or np.exp(-dE / T) > randomArray[k]:
            acceptedMoves += 1
            val_i_updated = -val_i

            # Flip the spin
            if i_shared:
                obj_ref[i] = -val_i_updated
            else:
                smcArrProps.arr[i] = -val_i_updated

            energy += dE
            magnetization += 2 * val_i_updated
    return acceptedMoves, energy, magnetization


@ray.remote
def run(lattice, T, h,
        obj_ref=None, region=None, smcArrProps=None,
        steps=100):
    accepted_moves, energy, magnetization = 0, [], []
    for _ in range(steps):
        accepted_moves, energy, magnetization = mc_step_dist(lattice, T, h,
                     accepted_moves, energy, magnetization,
                     obj_ref=obj_ref, region=region,
                     smcArrProps=smcArrProps
                     )
    return accepted_moves, energy, magnetization


def benchmark():
    futures = []
    start_time = time.time_ns()

    shared_size = len(model.smc_arr.multicore_imap)
    shared_mem = SharedMemory(name='SharedMem',
                              size=shared_size,
                              create=True,
                              )
    obj_ref = ray.put(shared_state_prop)

    for cpu_id in itertools.product(*[range(d) for d in CPU_GRID]):
        futures.append(run.remote(lattice, T=4, h=0,
                                  obj_ref=obj_ref,
                                  region=model.cpu_allotments[cpu_id],
                                  smcArrProps=model.smc_arr,
                                  steps=100))
    # results = ray.get(futures)
    shared_mem.unlink()
    return (time.time_ns() - start_time) / 1_000_000

print(benchmark(), "ms")
