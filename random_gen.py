# TODO: use this to wrap the three important random
# number states

class CustomRandomGen():
    def __init__(self, rng, seed, rng_progress=0):
        # The generator object and progress tracker should
        # be in synchrony.
        self.rng = rng
        self.rng_progress = 0
        # Initialization parameter
        self.rng_seed = seed

