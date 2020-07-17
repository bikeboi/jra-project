import numpy as np
from scipy.spatial.distance import cosine
import matplotlib
import matplotlib.pyplot as plt

import pynn_genn as sim
from pyNN.random import NumpyRNG, RandomDistribution

import simulation as S
import metrics
from model import default_MB

# Use GTK3Agg for "plt.show" functionality
matplotlib.use("GTK3Agg")

# Constants
T_SNAPSHOT = 50 # ms
DELTA_T = 0.01 # ms
RNG = NumpyRNG(seed=69)

D_INPUT = 10
N_SAMPLES = 10
N_EKC = 2


# Parameters
PARAMS = S.default_params(
    DELTA_T, T_SNAPSHOT, D_INPUT, N_SAMPLES, N_EKC,
    rng=RNG,
)

# Model building
MODEL_SETUP = default_MB

# Inputs
INPUT_SET = np.array([
    [ 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [ 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [ 0, 0, 0, 0, 0, 0, 0, 1, 0]
], dtype='float')


# Setup simulation
MB, spike_labels, intervals = S.setup_experiment(INPUT_SET, PARAMS, model_setup=MODEL_SETUP)

# Log inter and intraclass
distance = { "iKC": { "intra": [], "inter": [] } , "eKC": { "intra": [], "inter": [] } } 

def log_distance(t):
    for pop in ["iKC", "eKC"]:
        activity = results['activity'][pop][0]

        inter = metrics.interclass(activity, spike_labels, D=metrics.cosine_distance)
        intra = metrics.intraclass(activity, spike_labels, D=metrics.cosine_distance)

        distance[pop]["inter"] = inter.mean()
        distance[pop]["intra"] = intra.mean()
    
    return t + 10.0


# Run simulation
results = S.run_experiment(MB, intervals, PARAMS, intermediate_logging_step=25)

for res in results:
    print(res['activity']['PN'])


"""

fig = S.plot_results(results['data'], fr"Results")
plt.show()

print()
"""
# Cleanup
S.cleanup()