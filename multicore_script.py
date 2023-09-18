"""
This software is based on Luis Sena's code on Medium
Link: https://luis-sena.medium.com/sharing-big-numpy-arrays-
across-python-processes-abf0dc2a0ab2
"""

import time
# import ray
import numpy as np
from latt_blocks import LattBlocks
from plot_lattice import plot_lattice
from latt_zoo import square
from model import sweep

import itertools
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures.process import ProcessPoolExecutor
import multiprocessing
import concurrent

NUM_WORKERS = multiprocessing.cpu_count() - 2
# ray init can take 3 seconds or more to load
# ray.init(num_cpus=NUM_WORKERS)
LATTICE_SHAPE = (10, 10)
lattice = square(LATTICE_SHAPE)
model = LattBlocks(lattice=lattice)
CPU_GRID = (1, 1)
model.build_smc(CPU_GRID)

shared_state_prop = model.smc_arr.shared_arr
mc_imap = model.smc_arr.multicore_imap
cpu_allotments = model.cpu_allotments




def unlink():
    """This function is intended for debugging"""
    SharedName = "DistMcStates1"
    shared_mem = SharedMemory(name=SharedName,
                              create=False)
    shared_mem.close()
    shared_mem.unlink()


def benchmark():
    cpu_grid = CPU_GRID
    futures = []
    start_time = time.time_ns()

    shared_size = len(model.smc_arr.multicore_imap)
    SharedName = "DistMcStates1"

    shared_mem = SharedMemory(name=SharedName,
                              size=shared_size,
                              create=True)
    dst = np.ndarray(shape=(shared_size,), dtype=np.float32,
                     buffer=shared_mem.buf)
    dst[:] = np.ones(shared_size)
    with ProcessPoolExecutor(
            max_workers=np.product(cpu_grid)) as executor:
        for cpu_id in itertools.product(*[range(d) for d in cpu_grid]):
            futures.append(executor.submit(sweep, lattice,
                                           T=4, h=0,
                                           obj_ref=SharedName,
                                           region=model.cpu_allotments[cpu_id],
                                           smcArrProps=model.smc_arr,
                                           steps=1000))
    # results = ray.get(futures)

    futures, _ = concurrent.futures.wait(futures)
    # print([f.result() for f in futures])
    # Assign the shared properties to the array.
    model.smc_arr.arr[list(model.smc_arr.multicore_imap.keys())] = dst

    arr = np.ndarray(shape=(shared_size,), dtype=np.float32,
                     buffer=shared_mem.buf)
    print(arr)
    shared_mem.close()
    shared_mem.unlink()

    print(model.smc_arr.arr)
    plot_lattice(lattice, model.smc_arr.arr)
    return (time.time_ns() - start_time) / 1_000_000

if __name__ == '__main__':
    print(benchmark(), "ms")
