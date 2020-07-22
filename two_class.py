from lib.util import probability_conn_list, ProgBar, WeightLogger, calculate_steps, log_weights_callback, MushroomBody
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


# Get input samples
def get_inputs(n_class, downscale=1):

    # Get n_class character sets from the "Alphabet of the Magi" alphabet
    data_dir = "omniglot/python/images_background"
    dataset = Alphabet(data_dir, 0)
    inputs = dataset[:n_class, :]

    # Centering
    # TODO: Write algorithm to center images

    # Downsample
    inputs = inputs[:, :, ::downscale, ::downscale]

    # Binarize
    inputs = inputs.round()

    # Invert
    inputs = 1 - inputs

    # Flatten
    __, c, w, h = inputs.shape
    inputs = inputs.reshape(-1, c, w * h)

    return inputs


def build_model(input_spikes, n_eKC, rng=None):
    # Derived parameters
    n_PN = len(input_spikes)
    n_iKC = n_PN * 20

    # Synapse parameters
    g_PN_iKC = RandomDistribution('normal', (0.5, 0.05), rng=rng)
    t_PN_iKC = 5.0

    # g_iKC_eKC = RandomDistribution('normal', (0.125, 0.01), rng=rng)
    t_iKC_iKC = 5.0

    wd = sim.AdditiveWeightDependence(0.3, 1.0)
    td = sim.SpikePairRule()

    stdp = sim.STDPMechanism(
        timing_dependence=td, weight_dependence=wd,
        delay=t_iKC_iKC
    )

    # Neuron type
    neuron = sim.IF_curr_exp()

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
    proj_PN_iKC = sim.Projection(
        pop_PN, pop_iKC,
        sim.FixedProbabilityConnector(0.05),  # conn_PN_iKC,
        sim.StaticSynapse(weight=g_PN_iKC, delay=t_PN_iKC),
        label="PN_iKC"
    )

    proj_iKC_eKC = sim.Projection(
        pop_iKC, pop_eKC,
        sim.FixedProbabilityConnector(0.05),  # conn_iKC_eKC,
        stdp,
        label="iKC_eKC"
    )

    return MushroomBody(pop_PN, pop_iKC, pop_eKC, proj_PN_iKC, proj_iKC_eKC)


def run(inputs, run_id=0):
    # Simulation parameters
    delta_t = 0.1
    t_snapshot = 50
    n_eKC = 100

    # Derive steps
    steps = calculate_steps(inputs.shape[1], t_snapshot)

    # Setup the experiment
    print("Setting up")
    sim.setup(delta_t)

    # Input encoding
    input_spikes, __ = spike_encode(inputs, t_snapshot, t_snapshot)

    # Build the model
    model = build_model(input_spikes, n_eKC)

    model.record({
        "PN": ["spikes"],
        "iKC": ["spikes"],
        "eKC": ["spikes"]
    })

    # Log sim params to console
    print(" -- Steps:", steps)
    for name in ["PN", "iKC", "eKC"]:
        print(f" -- n_{name}: {len(model.pop[name])}")

    # Run
    results_path = f"results/two_class_{run_id}"
    # Clean out previous results
    shutil.rmtree(results_path, ignore_errors=True)
    os.makedirs(f"results/two_class_{run_id}")  # Make results directory

    io_PN = PickleIO(filename=f"results/two_class_{run_id}/PN.pickle")
    io_iKC = PickleIO(filename=f"results/two_class_{run_id}/iKC.pickle")
    io_eKC = PickleIO(filename=f"results/two_class_{run_id}/eKC.pickle")

    weight_logs = []
    weight_logger = log_weights_callback(model.proj['PN_iKC'], weight_logs, 10)

    weight_logger = WeightLogger(
        model.proj['iKC_eKC'], 10, f"results/two_class_{run_id}/weights.npy")
    progress_bar = ProgBar(steps)

    print("Running simulation..\n")
    sim.run(
        steps,
        callbacks=[
            weight_logger,
            progress_bar
        ]
    )
    print("\n\nDone")

    print("Saving results to disk...")

    model.pop["PN"].write_data(io_PN)
    model.pop["iKC"].write_data(io_iKC)
    model.pop["eKC"].write_data(io_eKC)

    weight_logger.finalize()

    # Useful params to return back to the caller
    sim_params = {
        "steps": steps,
        "t_snapshot": t_snapshot,
        "intervals": np.arange(t_snapshot, inputs.shape[1]*t_snapshot, t_snapshot)
    }

    sim.end()

    return sim_params


# Run the experment
if __name__ == "__main__":
    args = [int(arg) for arg in sys.argv[1:]]
    run_id, n_class, downscale = args
    inputs = get_inputs(n_class, downscale)
    run(inputs, run_id)

