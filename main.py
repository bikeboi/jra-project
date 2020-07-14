import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pynn_genn as sim
from pyNN.random import NumpyRNG, RandomDistribution

from embedding import spatiotemporal
from model import build_MB
from simulation import get_params, run_experiment, setup_experiment, plot_results


matplotlib.use("GTK3Agg")

# Constants
T_SNAPSHOT = 50 # ms
DELTA_T = 0.01 # ms
RNG = NumpyRNG(seed=69)
N_INPUT = 20
N_SAMPLES = 10

STEPS = (
    T_SNAPSHOT # Initial cooling buffer
    + T_SNAPSHOT * N_SAMPLES # Training period
    + T_SNAPSHOT * 4 # Cooling period
    + T_SNAPSHOT * (N_SAMPLES * 0.3) # Test period
    + T_SNAPSHOT # Final cooling buffer
)

N_EKC = 2
MUTUAL_INHIBITION = 1e-8

# Parameter overrides
synapse_params = {
    "g_PN_iKC": RandomDistribution('normal', [5.25, 1.25]),
    "g_eKC_inh": MUTUAL_INHIBITION,
    "g_iKC_eKC": RandomDistribution('normal', [5.25, 0.025]),
    "g_LHI_iKC": 8.75
}

# Parameters
PARAMS = get_params(
    STEPS, DELTA_T, T_SNAPSHOT, RNG, N_INPUT, N_SAMPLES, N_EKC,
    synapse=synapse_params
)

# Supervision flag
SUPERVISION = True # Not working for now

# Inputs
INPUT_SET = np.zeros((2,N_INPUT))
INPUT_SET[0,0] = 1
INPUT_SET[0,3] = 1
INPUT_SET[1,3] = 1 
INPUT_SET[1,6] = 1

RECORDINGS = {
    "PN": ["spikes"],
    "iKC": ["spikes", "v"],
    "eKC": ["spikes", "v"]
}

# Setup simulation
model, proj_1, proj_2, input_spikes = setup_experiment(INPUT_SET, PARAMS, RECORDINGS)

# Evaluation
simulation_intervals = np.arange(T_SNAPSHOT, STEPS-T_SNAPSHOT, T_SNAPSHOT)

def activity_metrics(spikes):
    activity = []
    for start_time in simulation_intervals:
        activity.append([ len(spiketrain[(start_time <= spiketrain) & (start_time + T_SNAPSHOT > spiketrain)]) for spiketrain in spikes ])

    activity = np.array(activity).T

    return activity

# Supervision
""" Make this more general
if SUPERVISION:
    supervision_intervals = np.arange(T_SNAPSHOT, T_SNAPSHOT+T_SNAPSHOT*N_SAMPLES, T_SNAPSHOT)
    eKCs = model.get_population("eKC")
    t_epsilon = 2 # ms
    t_pulse = 10 # ms
    inject_targets = np.argmax(activity_metrics(input_spikes), axis=0)

    pulse = lambda t: sim.StepCurrentSource(
        times=[t+t_epsilon, t+t_epsilon+t_pulse], 
        amplitudes=[1.0, 0.0]
    )

    for t,target in zip(supervision_intervals, inject_targets):
        pulse(t).inject_into(eKCs[target:target+1])
"""

# Run simulation
results = run_experiment(model, PARAMS)
iKC_eKC_weights = proj_2.get('weight', format='array')


# Plot results
#print("Final iKC-eKC weights:")
#print(iKC_eKC_weights)

fig = plot_results(results, fr"Results - Supervised ($inhibition = {MUTUAL_INHIBITION}$)")
plt.show()

sim.end()