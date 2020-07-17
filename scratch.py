import numpy as np
from pyNN.random import NumpyRNG

import lib.simulation as S
from lib.model import build_PN, build_population
from lib.data import Alphabet

# Data
dataset = Alphabet("experiment_1/omniglot/python/images_background", 0)

print(dataset[:2,0].shape)

def split_PN_iKC_model(inputs, **params):
    # Populations
    p_PN = build_PN(inputs)
    p_iKC = build_population(len(p_PN) * 10, params['neuron'], "iKC")
    p_eKC = build_population(params['eKC'], params['neuron'], 'eKC')

    # Connections
    


# Simulation parameters
T_SNAPSHOT = 50
DELTA_T = 0.1
RNG = NumpyRNG(seed=42)

D_INPUT = 105 * 105
N_CLASS = 2
N_SAMPLES = 5
N_EKC = int(np.ceil(np.log2(N_CLASS)))

PARAMS = S.default_params(
    DELTA_T, T_SNAPSHOT, D_INPUT, N_SAMPLES, N_EKC, rng=RNG
)