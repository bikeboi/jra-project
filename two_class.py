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
import cv2


# Get input samples
def get_inputs(n_class, downscale=1, remove=False):

    # Get n_class character sets from the "Alphabet of the Magi" alphabet
    data_dir = "omniglot/python/images_background"
    dataset = Alphabet(data_dir, 0)
    inputs = dataset[:n_class, :]

    # Invert
    inputs[:] = (inputs == 0)

    # Centering
    # TODO: Write algorithm to center images
    for c in range(inputs.shape[0]):
        for s in range(inputs.shape[1]):
            img = np.asarray(inputs[c, s, :, :])
            h, w = img.shape
            whr = np.where(img > 0)
            cy, cx = np.mean(whr[0]), np.mean(whr[1])
            Tx = np.asarray([[1, 0, -cx + w/2],[0, 1, -cy + h/2]])
            img[:] = cv2.warpAffine(img, Tx, (img.shape[1], img.shape[0]))
            inputs[c, s, :, :] = img
            

    # Downsample
    inputs = inputs[:, :, ::downscale, ::downscale]
            
    # Flatten
    _, s, h, w = inputs.shape
    
    # Remove half of the active pixels
    if remove and n_class == 2:
        inputs[0, :, h//2:, :] = 0 
        inputs[1, :, :h//2, :] = 0 

    inputs = inputs.reshape(-1, s, w * h)

    return inputs


def build_model(input_spikes, n_eKC, delta_t, neuron_params, syn_params, rng=None):

    print("Building Model")

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
    tau_threshold = 120.0  # ms (tuned in ./lib/example.py)
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

    # Projections
    # PN -> iKC
    proj_PN_iKC = sim.Projection(
        pop_PN, pop_iKC,
        sim.FixedProbabilityConnector(0.05),  # conn_PN_iKC,
        sim.StaticSynapse(weight=g_PN_iKC, delay=t_PN_iKC),
        label="PN_iKC"
    )

    # iKC -> eKC
    proj_iKC_eKC = sim.Projection(
        pop_iKC, pop_eKC,
        sim.FixedProbabilityConnector(0.05),  # conn_iKC_eKC,
        stdp,
        label="iKC_eKC"
    )

    # Lateral connection matrix
    def lateral_conn_matrix(p): return 1 - np.identity(len(p))

    ## sWTA (eKC)
    proj_sWTA_eKC = sim.Projection(
        pop_eKC, pop_eKC,
        sim.ArrayConnector(lateral_conn_matrix(pop_eKC)),
        sim.StaticSynapse(weight=g_sWTA_eKC, delay=t_sWTA_eKC),
        receptor_type='inhibitory',
        label="sWTA_eKC"
    )

    # sWTA (iKC)
    pop_iKC_inh = sim.Population(50, neuron, label="iKC_inh")

    proj_iKC_inh = sim.Projection(
        pop_iKC, pop_iKC_inh,
        sim.AllToAllConnector(),
        sim.StaticSynapse(weight=g_sWTA_iKC, delay=delta_t),
        receptor_type='excitatory',
    )

    proj_sWTA_iKC = sim.Projection(
        pop_iKC_inh, pop_iKC,
        sim.AllToAllConnector(),
        sim.StaticSynapse(weight=g_sWTA_iKC, delay=t_sWTA_iKC),
        receptor_type='inhibitory',
        label="sWTA_iKC"
    )

    return MushroomBody(pop_PN, pop_iKC, pop_eKC, proj_PN_iKC, proj_iKC_eKC)


def initialize_model(mb: MushroomBody, neuron_params, proj_params):
    # Initialize neuron parameters
    mb.pop["iKC"].set(**neuron_params)
    mb.pop["eKC"].set(**neuron_params)

    # Projections
    # PN->iKC params
    mb.proj['PN_iKC'].set(**proj_params['PN_iKC'])
    mb.proj['iKC_eKC'].initialize(**proj_params['iKC_eKC'])


def run(inputs, runs=1, spike_jitter=0, version=0, weight_log_freq=50, neuron_params={}, syn_params={}):
    # Simulation parameters
    delta_t = 0.1
    t_snapshot = 50
    n_eKC = 500

    # Derive steps
    steps = calculate_steps(inputs.shape[1], t_snapshot)

    # Setup the experiment
    print("Setting up")
    sim.setup(delta_t)

    # Input encoding
    input_spikes, labels, samples = spike_encode(
        inputs, t_snapshot, t_snapshot, spike_jitter=spike_jitter)

    # Build the model
    model = build_model(input_spikes, n_eKC, delta_t,
                        neuron_params, syn_params)

    model.record({
        "PN": ["spikes"],
        "iKC": ["spikes"],
        "eKC": ["spikes"]
    })

    # Log sim params to console
    print(" -- steps:", steps)
    for name in ["PN", "iKC", "eKC"]:
        print(f" -- n_{name}: {len(model.pop[name])}")

    # Run
    results_path = f"results/two_class_{version}"
    # Clean out previous results
    shutil.rmtree(results_path, ignore_errors=True)
    os.makedirs(f"results/two_class_{version}")  # Make results directory

    io_PN = PickleIO(filename=f"results/two_class_{version}/PN.pickle")
    io_iKC = PickleIO(filename=f"results/two_class_{version}/iKC.pickle")
    io_eKC = PickleIO(filename=f"results/two_class_{version}/eKC.pickle")

    print("Initializing weight logger...")
    weight_logger = WeightLogger(
        model.proj['iKC_eKC'], weight_log_freq, f"results/two_class_{version}/weights.npy")
    progress_bar = ProgBar(steps)

    print("Running simulation..\n")
    for __ in range(runs):
        # initialize_model(model, neuron_params, syn_params)
        sim.run(
            steps,
            callbacks=[
                weight_logger,
                progress_bar
            ]
        )
        weight_logger.reset()
        sim.reset()

    print("\n\nDone")

    print("Saving results...")

    model.pop["PN"].write_data(io_PN)
    model.pop["iKC"].write_data(io_iKC)
    model.pop["eKC"].write_data(io_eKC)

    weight_logger.finalize()

    # Useful params to return back to the caller
    sim_params = {
        "steps": steps,
        "t_snapshot": t_snapshot,
        "intervals": np.arange(t_snapshot, inputs.shape[0]*inputs.shape[1]*t_snapshot, t_snapshot),
        "labels": labels,
        "samples": samples
    }

    # Save params to disk
    print("Saving simulation params...")
    np.savez(f"results/two_class_{version}/params", **sim_params)

    sim.end()

    return sim_params


# Run the experment
if __name__ == "__main__":
    args = [int(arg) for arg in sys.argv[1:]]
    version, n_class, downscale, jitter = args
    inputs = get_inputs(n_class, downscale)
    run(inputs, spike_jitter=jitter, version=version)
