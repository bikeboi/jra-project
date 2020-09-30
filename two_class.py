from lib.util import probability_conn_list, ProgBar, WeightLogger, calculate_steps, MushroomBody
from lib.cells import IF_curr_exp_adapt
from lib.embedding import spike_encode
from lib.data import Alphabet
from lib.analysis import calculate_activity

import os
import shutil
import sys
from neo.io import PickleIO
from pyNN.random import RandomDistribution
import pynn_genn as sim
import numpy as np





def build_model(input_spikes, n_eKC, delta_t, neuron_params, syn_params, supervision=None, rng=None):
    # Derived parameters
    n_PN = len(input_spikes)
    n_iKC = n_PN * 10

    # Default synapse parameters
    # RandomDistribution('normal', (0.5, 0.1), rng=rng)
    g_PN_iKC = syn_params['PN_iKC']['weight']
    t_PN_iKC = syn_params['PN_iKC']['delay']  # 5.0

    # RandomDistribution('normal', (1.0, 0.5), rng=rng)
    g_iKC_eKC = syn_params['iKC_eKC']['weight']
    t_iKC_eKC = syn_params['iKC_eKC']['delay']  # 5.0

    g_sWTA_eKC = syn_params['sWTA_eKC']['weight']
    t_sWTA_eKC = delta_t

    g_sWTA_iKC = syn_params['sWTA_iKC']['weight']
    t_sWTA_iKC = delta_t

    stdp = sim.STDPMechanism(
        weight_dependence=sim.AdditiveWeightDependence(
            **syn_params['iKC_eKC']['wd_params']),
        timing_dependence=sim.SpikePairRule(
            **syn_params['iKC_eKC']['td_params']),
        weight=g_iKC_eKC,
        delay=t_iKC_eKC
    )

    # Neuron type
    tau_threshold = 120.0
    # neuron = sim.IF_curr_exp(**neuron_params)
    neuron = IF_curr_exp_adapt(tau_threshold=tau_threshold, **neuron_params)

    # Populations
    pop_PN = sim.Population(
        n_PN, sim.SpikeSourceArray(spike_times=input_spikes),
        label="PN"
    )

    pop_iKC = sim.Population(
        n_iKC, neuron,
        label="iKC"
    )

    pop_eKC = sim.Population(
        n_eKC, neuron,
        label="eKC"
    )

    # Supervision
    if supervision:
        supervision(pop_eKC)

    # Projections
    def n_conn(target): return int(syn_params['sparsity'] * len(target))

    # PN -> iKC
    proj_PN_iKC = sim.Projection(
        pop_PN, pop_iKC,
        # sim.FixedProbabilityConnector(syn_params['sparsity']),  # conn_PN_iKC,
        sim.FixedNumberPostConnector(n_conn(pop_iKC)),
        sim.StaticSynapse(weight=g_PN_iKC, delay=t_PN_iKC),
        label="PN_iKC"
    )

    # iKC -> eKC
    proj_iKC_eKC = sim.Projection(
        pop_iKC, pop_eKC,
        # sim.FixedProbabilityConnector(syn_params['sparsity']),  # conn_iKC_eKC,
        sim.FixedNumberPreConnector(n_conn(pop_iKC)),
        stdp,
        label="iKC_eKC"
    )

    # Lateral connection matrix
    def lateral_conn_matrix(p): return 1 - np.identity(len(p))

    # sWTA (eKC)
    proj_sWTA_eKC = sim.Projection(
        pop_eKC, pop_eKC,
        # sim.ArrayConnector(lateral_conn_matrix(pop_eKC)),
        sim.AllToAllConnector(allow_self_connections=False),
        sim.StaticSynapse(weight=g_sWTA_eKC, delay=t_sWTA_eKC),
        receptor_type='inhibitory',
        label="sWTA_eKC"
    )

    # sWTA (iKC)
    pop_iKC_inh = sim.Population(
        int(len(pop_iKC) / 10), sim.IF_curr_exp(tau_m=delta_t, tau_syn_I=delta_t, tau_syn_E=delta_t), label="iKC_inh")

    proj_iKC_inh = sim.Projection(
        pop_iKC, pop_iKC_inh,
        sim.AllToAllConnector(),
        sim.StaticSynapse(weight=1.0, delay=0.1),
        receptor_type='excitatory',
    )

    proj_sWTA_iKC = sim.Projection(
        pop_iKC_inh, pop_iKC,
        sim.AllToAllConnector(allow_self_connections=False),
        sim.StaticSynapse(weight=g_sWTA_iKC, delay=t_sWTA_iKC),
        receptor_type='inhibitory',
        label="sWTA_iKC"
    )

    return MushroomBody(pop_PN, pop_iKC, pop_eKC, proj_PN_iKC, proj_iKC_eKC)


def run(inputs, labels, supervision_setup=None, runs=1, spike_jitter=0, version=0, weight_log_freq=50, n_eKC=100,
        verbosity=1, neuron_params={}, syn_params={}):
    # Simulation parameters
    delta_t = 0.1
    t_snapshot = 50

    # Derive steps
    n_sample = len(inputs)
    steps = calculate_steps(n_sample, t_snapshot)

    # Setup the experiment
    sim.setup(delta_t)

    # Input encoding
    input_spikes = spike_encode(inputs, t_snapshot, spike_jitter=spike_jitter)

    # Build the model
    model = build_model(input_spikes, n_eKC, delta_t,
                        neuron_params, syn_params)

    # Supervision
    intervals = np.arange(0, n_sample * t_snapshot, t_snapshot)
    if supervision_setup:
        sup = supervision_setup(intervals, labels)
        sup(model.pop["eKC"])

    # Record variables
    model.record({
        "PN": ["spikes"],
        "iKC": ["spikes"],
        "eKC": ["spikes"]
    })

    # Log sim params to console
    if verbosity > 0:
        print(" -- steps:", steps)
        for name in ["PN", "iKC", "eKC"]:
            print(f" -- n_{name}: {len(model.pop[name])}")

    # Storing results
    results_path = f"results/two_class_{version}"
    # Clean out previous results
    shutil.rmtree(results_path, ignore_errors=True)
    os.makedirs(f"results/two_class_{version}")  # Make results directory

    io_PN = PickleIO(filename=f"results/two_class_{version}/PN.pickle")
    io_iKC = PickleIO(filename=f"results/two_class_{version}/iKC.pickle")
    io_eKC = PickleIO(filename=f"results/two_class_{version}/eKC.pickle")

    weight_logger = WeightLogger(
        model.proj['iKC_eKC'], weight_log_freq, f"results/two_class_{version}/weights.npy")
    progress_bar = ProgBar(steps)

    # Per-run variables
    for __ in range(runs):
        # Run simulation
        sim.run(
            steps,
            callbacks=[
                weight_logger,
                progress_bar
            ]
        )

        # Logging
        weight_logger.reset()
        sim.reset()

    model.pop["PN"].write_data(io_PN)
    model.pop["iKC"].write_data(io_iKC)
    model.pop["eKC"].write_data(io_eKC)

    weight_logger.finalize()

    # Useful params to return back to the caller
    sim_params = {
        "steps": steps,
        "t_snapshot": t_snapshot,
        "intervals": np.arange(0, n_sample * t_snapshot, t_snapshot),
        "labels": labels,
    }

    # Save params to disk
    np.savez(f"results/two_class_{version}/params", **sim_params)

    sim.end()

    return sim_params


"""
# Run the experment
if __name__ == "__main__":
    args = [int(arg) for arg in sys.argv[1:]]
    version, n_class, downscale, jitter = args
    inputs, labels = get_inputs(n_class, downscale)
    run(inputs, labels, spike_jitter=jitter, version=version)
"""
