from multiprocessing.shared_memory import SharedMemory
from update import mc_step, mc_step_dist
from latt_zoo import square, hexagon, cubic
import numpy as np
import datetime


def sweep(lattice, T, h,
          multicore=False,
          states_ref=None, region=None, smcArrProps=None,
          steps=10, rng=None):
    if multicore:
        shm = SharedMemory(name=states_ref, create=False)
    # array_shape = (len(model.smc_arr.multicore_imap), 0)
    # np_array = np.ndarray(array_shape, dtype=np.float64,
    #                       buffer=shm.buf)

    accepted_moves = 0
    energy, magnetization = ene_and_mag(lattice, states_ref)
    rng_progress = 0
    if multicore:
        for _ in range(steps):
            am, en, mag = mc_step_dist(lattice, T, h,
                accepted_moves, energy, magnetization,
                states_ref=shm, region=region,
                smcArrProps=smcArrProps, rng=rng)
            accepted_moves += am
            energy += en
            magnetization += mag
    else:
        for _ in range(steps):
            rng_progress_new, am, en, mag = mc_step(lattice, states_ref, T, h,
                                       accepted_moves,
                                       energy,
                                       magnetization, rng=rng)
            accepted_moves = am
            energy = en
            magnetization = mag
            rng_progress += rng_progress_new

    # Close the shared memory
    if multicore:
        shm.close()
    return accepted_moves, energy, magnetization, rng_progress


class Model:
    """
    Manage an instance of a Metropolis monte carlo
    calculation. This does not yet support multi-processor
    computing.
    """
    def __init__(self,
                 shape=(10, 10),
                 lattice_structure="square",
                 temperature=10.0,
                 field=0,
                 random_generator_seed=1,
                 build=False,
                 t_max_delta=1e-3,
                 f_max_delta=1e-3,
                 periodic=None):
        self.shape = shape
        self.lattice_structure = lattice_structure
        self.periodic = periodic

        self.temperature = temperature
        self.temperature_max_delta = t_max_delta

        self.field = field
        self.field_max_delta = f_max_delta

        self.random_generator_seed = random_generator_seed
        self.save_freq = 20
        self.previous_snapshot_filename = None

        self.num_sweeps = 0

        self.lattice = None
        self.state = None

        self.rng = None
        self.rng_progress = 0

        if build:
            self.init_simulation_state()

    def sweep(self, steps=10, save_freq=1000):
        """
        Steps: number of sweeps across the entire lattice
        Save_freq: frequency for saving a snapshot.

        Returns: number of accepted moves, history of energy
        after each sweep, history of magnetization after each
        sweep.
        """
        if save_freq <= 100:
            print(f"save_freq={save_freq} is too small.")

        accepted_moves, energy, magnetization = 0, [], []
        for i in range(steps):
            acc, ene, mag, rng_progress = sweep(self.lattice,
                                                self.temperature,
                                                self.field,
                                                states_ref=self.state,
                                                steps=1,
                                                rng=self.rng)
            self.rng_progress += rng_progress
            accepted_moves += acc
            energy.append(ene)
            magnetization.append(mag)

            if (i + 1) % save_freq == 0:
                self.save(time=False)
        return accepted_moves, energy, magnetization

    # State initiation and resetting
    def init_simulation_state(self):
        if self.lattice_structure == "square":
            self.lattice = square(self.shape)
        elif self.lattice_structure == "hexagon":
            self.lattice = hexagon(self.shape)
        elif self.lattice_structure == "cubic":
            self.lattice = cubic(self.shape)
        if self.periodic:
            self.lattice.set_periodic(self.periodic)
        self.state = np.ones(self.lattice.num_sites)
        self.rng = np.random.default_rng(self.random_generator_seed)
        self.rng_progress = 0

    def set_simulation_state(self,
                             lattice,
                             state,
                             rng,
                             rng_progress):
        self.lattice = lattice
        self.state = state
        self.rng = rng
        self.rng_progress = rng_progress

    def reinit_rng(self):
        self.rng = np.random.default_rng(
            self.random_generator_seed)

    def roll_rng(self, up_range):
        for _ in range(up_range // 1000):
            self.rng.random(1000)
        self.rng.random(up_range % 1000)

    def raise_to(self, temp=0):
        self.temperature = temp

    def cool_to(self, temp=0):
        self.temperature = temp

    def heat(self, d_temp=0):
        self.raise_to(self.temperature + d_temp)

    def size(self):
        return self.lattice.num_sites

    # Model organization, saving, and loading
    def params_dict(self):
        """
        Output a parameter dictionary of the numerical
        model.
        For instance:
            {"T": 298.15, "random_state": 235711}

        Random_state is the state of the random number
        generator. It needs to be specifically handled.
        This is NOT for the state of the physical system.
        """
        return {
            "lattice_structure": self.lattice_structure,
            "shape": self.shape,
            "temperature": self.temperature,
            "random_state": self.random_generator_seed,
            "rng_progress": self.rng_progress,
            "num_sweeps":self.num_sweeps
        }

    def time_stamp(self):
        """
        [Github Copilot] Return a time stamp in ISO format.
        """
        stamp = datetime.datetime.now().isoformat()
        return stamp

    def make_snapshot_filename(self, fileprefix="",
                               time=False, sweep_num=True):
        if time:
            fname = fileprefix + "{str}_shape={s}_temp={t}_{time}".format(
            str=self.lattice_structure,
            s=shape_str(self.shape),
            t=self.temperature, time=self.time_stamp())
        else:
            fname =fileprefix + "{str}_shape={s}_temp={t}".format(
                str=self.lattice_structure,
                s=shape_str(self.shape),
                t=self.temperature)

        num_sweeps = str(int(
                self.rng_progress / self.lattice.num_sites / 2))
        return  fname + ("" if not sweep_num else "_s=" + num_sweeps)

    def save(self, fileprefix="", time=True, sweep_num=True):
        # You technically will want to save random
        # generator's current state. Then you continue
        # training.
        num_sweeps = int(self.rng_progress / self.lattice.num_sites / 2)
        filename = self.make_snapshot_filename(fileprefix,
                                               time=time,
                                               sweep_num=sweep_num)
        # print("saving at num_sweeps= " + str(num_sweeps))

        np.savez(
            filename,
            state=self.state,
            **self.params_dict()
        )
        self.previous_snapshot_filename = filename

    def load(self, fileprefix="",  not_stored=[]):
        """
        [Github Copilot] Load a snapshot of the system.

        not_stored: a list containing names not stored
        in the file.
        """
        fileName = self.make_snapshot_filename(
                fileprefix, time=False)
        loaded_dict = np.load(
            fileName + ".npz"
        )

        state = loaded_dict["state"]
        self.lattice_structure = loaded_dict["lattice_structure"]
        shape_str = loaded_dict["shape"]
        self.temperature = loaded_dict["temperature"]
        random_state = loaded_dict["random_state"]

        self.num_sweeps = loaded_dict["num_sweeps"]
        self.shape = tuple(shape_str)
        self.random_generator_seed = random_state

        self.state = state

        if "rng_progress" not in not_stored:
            self.rng_progress = loaded_dict["rng_progress"]
        else:
            self.rng_progress = 0


def shape_str(shape):
    return "x".join([str(s) for s in shape])


def shape_tup(shape_str):
    return tuple([int(s) for s in shape_str.split("x")])


def ene_and_mag(lattice, states):
    ene, mag = 0, 0
    for i in range(lattice.num_sites):
        neighbor_energies = 0
        for n in lattice.nearest_neighbors(i):
            neighbor_energies += states[n]
        ene += -states[i] * neighbor_energies / 2
        mag += states[i]
    return ene, mag
