# This code chops a lattice into blocks.
import numpy as np
import itertools


class SelectiveMultiCoreArray:
    """
    An array where certain indices are shared across
    processor cores, but the rest of the array
    are replicated on each core.

    Use:
    random_arr = np.random.random((int(2e8),))
    smc_arr = SelectiveMultiCoreArray(random_arr)
    obj_ref = ray.put(smc_array.shared_arr)
    """
    def __init__(self,
                 arr):
        self.arr = arr
        # This array must be in a cached location across
        # cores.
        self.shared_arr = np.zeros(())
        # Map index in self.properties to index in multicore
        # states
        self.multicore_imap = {}

    def get(self, i):
        if i in self.multicore_imap:
            return self.shared_arr[self.multicore_imap[i]]
        else:
            return self.arr[i]

    def put(self, i, val):
        if i in self.multicore_imap:
            self.shared_arr[self.multicore_imap[i]] = val
            self.arr[i] = val
        else:
            self.arr[i] = val

    def share(self, i):
        if i not in self.multicore_imap:
            self.shared_arr = np.append(self.shared_arr, i)
            self.multicore_imap[i] = len(self.shared_arr) - 1

    def share_and_put(self, i, val):
        self.share(i)
        self.put(i, val)


class DistIsingLattice:
    """
    Class to keep track of properties and the lattice
    together
    """
    def __init__(self,
                 lattice,
                 num_properties=1):
        """
        Class
        """
        self.lattice = lattice

        self.state_property = np.ones((self.lattice.num_sites,
                                        num_properties))
        self.smc_arr = SelectiveMultiCoreArray(self.state_property)
        # Maps the CPU index to the lattice nodes it is
        # responsible for
        self.cpu_allotments = {}

    def build_smc(self, cpu_grid_shape):
        """
        Assign nodes to cores and find the nodes that
        should be shared across cores.
        """
        cpu_num = np.prod(cpu_grid_shape)

        cell_size = self.lattice.cell_size
        section_atol = cell_size * 0.52

        limits = self.lattice.limits()
        section_lens = []
        for d, num_sections in enumerate(cpu_grid_shape):
            pos_min, pos_max = limits[0, d], limits[1, d]
            section_lens.append((pos_max - pos_min) / cpu_grid_shape[d])

        cpu_allotments = {}
        for coords in itertools.product(*[range(d) for d in cpu_grid_shape]):
            cpu_allotments[tuple(coords)] = []
        for i in range(self.lattice.positions.shape[0]):
            p = self.lattice.positions[i]
            # The coordinates of the CPU in a grid
            cpu_coords = []
            for d, col_val in enumerate(p):
                prop_val = 1

                if col_val - np.floor(col_val / section_lens[d]) * section_lens[d] < section_atol[d]:
                    self.smc_arr.share_and_put(i, prop_val)
                if np.ceil(col_val / section_lens[d]) * section_lens[d] - col_val  < section_atol[d]:
                    self.smc_arr.share_and_put(i, prop_val)

                cpu_index = int(
                    np.floor(col_val / section_lens[d]))
                if cpu_index >= cpu_grid_shape[d]:
                    cpu_index -= 1
                cpu_coords.append(cpu_index)

            if np.prod(cpu_coords) >=cpu_num:
                print(cpu_coords, i)
            cpu_allotments[tuple(cpu_coords)].append(i)
        self.cpu_allotments = cpu_allotments
        return self.smc_arr, cpu_allotments


    def extend(self, **kwargs):
        """
        Extend the lattice object and the properties array
        together.
        """
        old_numsites = self.lattice.num_sites
        self.lattice.extend(**kwargs)
        num_newsites = self.lattice.num_sites - old_numsites
        self.state_property = np.vstack((self.state_property,
                                         np.ones((num_newsites,
                                              self.state_property.shape[1]))))

    def plot_smc_locs(self):
        axes = self.lattice.plot()
        axes.scatter(*self.lattice.positions[[] + list(self.smc_arr.multicore_imap.keys())].T,
                    s=5, marker='o', c="red")
        return axes


# TODO: check whether this should still be here.
class LatticeBlock:
    """
    Keep a lattice as distinct blocks. Each block may be
    computed on a single computer core.

    Each block has its own list of indices.
    Each block has a list of edge nodes, whose properties
    are located on a shared CPU cache, shared among cores.
    """



