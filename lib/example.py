import numpy as np
from scipy.spatial.distance import cosine
import matplotlib
import matplotlib.pyplot as plt

import pynn_genn as sim
from pyNN.random import NumpyRNG, RandomDistribution

import simulation as S
import metrics
from model import default_MB, default_params
from embedding import spike_encode

# Use GTK3Agg for "plt.show" functionality
matplotlib.use("GTK3Agg")

# Constants
T_SNAPSHOT = 50 # ms
DELTA_T = 0.01 # ms
RNG = NumpyRNG(seed=69)

D_INPUT = 10
N_SAMPLES = 10
N_EKC = 2

# Inputs
INPUT_SET = np.random.binomial(1, 0.1, (2, N_SAMPLES, D_INPUT))
spike_coding, spike_labels = spike_encode(INPUT_SET, t_snapshot=T_SNAPSHOT, start_time=T_SNAPSHOT)


# Parameters
PARAMS = default_params(
    DELTA_T, T_SNAPSHOT, D_INPUT, N_SAMPLES, N_EKC,
    rng=RNG,
)

# Model building
MODEL_SETUP = default_MB


# Setup simulation
MB = S.setup_experiment(spike_coding, model_setup=MODEL_SETUP, **PARAMS)

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


intervals = np.arange(T_SNAPSHOT, T_SNAPSHOT * N_SAMPLES, T_SNAPSHOT)

# Run simulation
results = S.run_experiment(MB, intervals, **PARAMS)

fig = S.plot_results(results['data'], fr"Results")
plt.show()

print()

# Cleanup
S.cleanup()
