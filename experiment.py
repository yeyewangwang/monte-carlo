from Ising import Ising2D
import numpy as np
import time


# TODO: Use xarray for tensors.
# TODO: Get ready for model save and load procedure
#  for 1 million nodes, etc.

class Experiment:
    """
    1. Manages grid search across various model parameters
    2. Save in the middle of a grid search and resume
    3. Save the results of the grid search to a file
    """

    def __init__(self, ts,
                 steps,
                 dimensions=2,
                 ls=[10],
                 res_path="result"):
        num_observed = 6
        self.ts = ts
        self.dimensions = dimensions
        # self.ls = [2, 4, 8, 16, 32, 64, 128]
        self.ls = ls
        self.steps = steps  # [1000, 4000, 5000, 10000]

        # Initiate indexes for the Grid Search
        self.ts_curr = 0
        self.ls_curr = 0
        self.steps_curr = 0


        # The last model object, provided for occasional
        # plotting purposes.
        self.last_model = None

        # Temperatures, length, number of steps,
        # number of observed measurements
        self.results = np.zeros((len(ts), len(self.ls),
                                 len(steps), num_observed))
        self.res_path = res_path

        self.quantities = {
            0: "Temperature",
            1: "Mean Energy",
            2: "Mean Magnetization",
            3: "Specific Heat",
            4: "Susceptibility",
            5: "Monte Carlo Steps"
        }

        self.initial_seed = 0

    def getData(self,
                time_lengths=[],
                side_lengths=[],
                step_numbers=[]):
        print("The quantities are",
              self.quantities)

        return self.results[
               time_lengths,
               side_lengths,
               step_numbers, :]

    def getExperimentState(self):
        return "T={}, L={}, Steps={}".format(self.ts[self.ts_curr],
                                             self.ls[self.ls_curr],
                                             self.steps[self.steps_curr])



    def load(self):

        self.results = np.load(self.res_path)

    def run(self, i=0, j=0, k=0):
        total_time = 0
        total_steps = 0
        while i < len(self.ts):
            t = self.ts[i]
            while j < len(self.ls):
                l = self.ls[j]
                print(instance_name(t, l))
                model = Ising2D(l, t, 0, self.dimensions)

                while k < len(self.steps):
                    print(self.getExperimentState())
                    step = self.steps[k]
                    start_time = time.time()
                    model.steps(number=step)
                    end_time = time.time()
                    self.results[i][j][k] = \
                        model.observables_array()

                    ### Checkpoint at the end of each run.
                    self.last_model = model
                    model.save()

                    total_time += end_time - start_time
                    total_steps += step
                    print("Average time per step",
                          total_time / total_steps)
                    print("Estimated hours to 1 million "
                          "steps", ((1e6 - total_steps) *
                                    total_time / total_steps) / 60 / 60)
                    self.steps_curr = k
                    k += 1

                np.save(self.res_path, self.results)
                self.ls_curr = j
                j += 1
            self.ts_curr = i
            i += 1

    def resume(self):
        """
        Resume the numerical experiment.
        """
        self.run(i=self.ts_curr,
                 j=self.ls_curr,
                 k=self.steps_curr)


def instance_name(t, size):
    """The name for one instance of the model"""
    return "T={t}_L={l}".format(t=t, l=size)

