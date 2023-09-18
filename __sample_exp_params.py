"""
Constants for hexagonal structure experiment
"""
from constants import SQUARE_TC, CUBIC_TC
from model import Model

LATTICE_STRUCTURE_1 = "square"
LATTICE_STRUCTURE_2 = "cubic"
# EXP_NAME = "__hex_exps"
EXP_NAME = "sample_exp_2"
PREFIX_1 = "exps/steps=100000"

SHAPE_1 = (128, 128)
SHAPE_2 = (32, 32, 32)

TEMPERATURES_1 = SQUARE_TC + 0.1
TEMPERATURES_2 = CUBIC_TC + 0.1

CURR_STEPS = 95000
NUM_STEPS = 100000 - CURR_STEPS


MODELS = []
SQ_MODEL = Model(shape=SHAPE_1,
                 lattice_structure=LATTICE_STRUCTURE_1,
                 temperature=TEMPERATURES_1,
                 build=False)
MODELS.append(SQ_MODEL)

CUBIC_MODEL = Model(shape=SHAPE_2,
                    lattice_structure=LATTICE_STRUCTURE_2,
                    temperature=TEMPERATURES_2,
                    build=False)
MODELS.append(CUBIC_MODEL)

