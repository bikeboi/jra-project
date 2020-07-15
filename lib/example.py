import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pynn_genn as sim
from pyNN.random import NumpyRNG, RandomDistribution

from model import default_MB
import simulation as S

# Use GTK3Agg for "plt.show" functionality
matplotlib.use("GTK3Agg")

# Constants
T_SNAPSHOT = 50 # ms
DELTA_T = 0.01 # ms
RNG = NumpyRNG(seed=69)

D_INPUT = 10
N_SAMPLES = 6
N_EKC = 1


# Parameters
PARAMS = S.default_params(
    DELTA_T, T_SNAPSHOT, D_INPUT, N_SAMPLES, N_EKC,
    rng=RNG,
)

# Model building
MODEL_SETUP = default_MB

# Inputs
"""
Input Set:
[
    [ 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 1, 0, 0, 0, 0, 0]
]
"""
INPUT_SET = np.zeros((2,D_INPUT))
INPUT_SET[0,0] = 1
INPUT_SET[1,3] = 1 


# Setup simulation
MB, intervals = S.setup_experiment(INPUT_SET, PARAMS, model_setup=MODEL_SETUP)

# Run simulation
results = S.run_experiment(MB, intervals, PARAMS)

iKC_activity = results['activity']['iKC'][0]

fig = S.plot_results(results['data'], fr"Results")
plt.show()

print(iKC_activity)

# Cleanup
S.cleanup()