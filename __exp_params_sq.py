"""
Constants for hexagonal structure experiment
"""
from constants import SQUARE_TC, CUBIC_TC
from model import  Model
import numpy as np

SHAPE = (32, 32)
LATTICE_STRUCTURE = "square"
TEMPERATURES = np.hstack((np.arange(SQUARE_TC - 2, SQUARE_TC + 2, 0.5),np.arange(10, 200, 20)))
EXP_NAME = "sq_exp"

MODELS = []
for t in TEMPERATURES:
    m = Model(shape=SHAPE,
              lattice_structure=LATTICE_STRUCTURE,
              temperature=t,
              build=False)
    MODELS.append(m)