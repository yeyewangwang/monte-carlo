"""
Constants for hexagonal structure experiment
"""
from constants import SQUARE_TC, CUBIC_TC
from model import  Model
import numpy as np

LATTICE_STRUCTURE = "hexagon"
# EXP_NAME = "__hex_exps"
EXP_NAME = "hex_exp_3"
PREFIX_1 = "__hex_exps/steps=10"
EXP_NAME_4 = "hex_exp_4"
SHAPE_1 = (32, 32)
SHAPE = (100, 100)
SHAPE_4 = (100, 100)

TEMPERATURES_1 = np.arange(SQUARE_TC - 1, CUBIC_TC + 1.5, 0.5)
TEMPERATURES = np.arange(3.5, 7, 0.1)
TEMPERATURES_4 = np.round(np.arange(4.1, 4.7, 0.1), 2)

NUM_STEPS_1 = 10
NUM_STEPS = 10000
NUM_STEPS_4 = 10000

MODELS_1 = []
MODELS = []
MODELS_4 = []
for t in TEMPERATURES_4:
    m_4 = Model(shape=SHAPE_4,
              lattice_structure=LATTICE_STRUCTURE,
              temperature=t,
              build=False)
    MODELS_4.append(m_4)

for t in TEMPERATURES_1:
    m_1 = Model(shape=SHAPE_1,
              lattice_structure=LATTICE_STRUCTURE,
              temperature=t,
              build=False)
    MODELS_1.append(m_1)

for t in TEMPERATURES:
    m = Model(shape=SHAPE,
              lattice_structure=LATTICE_STRUCTURE,
              temperature=t,
              build=False)
    MODELS.append(m)


