import numpy as np
import pynn_genn as sim
from pyNN.random import RandomDistribution
from pyNN.utility.plotting import Figure, Panel
from pyNN.utility import ProgressBar

from model import build_MB
from embedding import spatiotemporal

# Setup experiment
def setup_experiment(input_set, params, recordings={}):
    # Setup sim
    sim.setup(params['delta_t'])
    np.random.seed(42) # Magic number ¯\_(ツ)_/¯

    # Train period
    train_ixs = np.random.choice( # Sample inputs
        len(input_set),
        size=params['n_samples'],
        replace=params['n_samples'] > len(input_set)
    )
    train = input_set[train_ixs].T
    train_spikes = spatiotemporal(train, params['t_snapshot'], params['t_snapshot'])

    test_ixs = np.random.choice( # Sample inputs
        len(input_set),
        size=int(params['n_samples']*.3),
        replace=int(params['n_samples']*.3) > len(input_set)
    )
    test = input_set[test_ixs].T
    test_spikes = spatiotemporal(
        test, 
        params['t_snapshot'],
        params['t_snapshot'] + params['t_snapshot'] * params['n_samples'] + 4 * params['t_snapshot']
    ) 

    # Combine them all
    input_spikes = []
    for tr,ts in zip(train_spikes, test_spikes):
        input_spikes.append(np.concatenate([tr, ts]))

    # Build Model
    model, proj_1, proj_2 = build_MB(input_spikes, **params)
    model.record(["spikes"])
    model.get_population("eKC").record("v")

    return model, proj_1, proj_2, input_spikes

def run_experiment(model, params):
    # pb = ProgressBar(char='=')

    # Run simulation
    #print("--\nRunning Simulation")
    #step = params['steps'] / 10
    #for i in range(10):
    #    sim.run_until(step*i)
    #    pb.set_level(i/10)

    sim.run(params['steps'])
    print("Done\n--")

    return  { pop.label:  pop.get_data() for pop in model.populations }


# Plotting
def plot_results(results, title):
    plot_settings = {
        "lines.linewidth": 1.5,
        "lines.markersize": 18,
        "font.size": 14,
    }

    panels = [ Panel(result.segments[0].spiketrains, data_labels=[k], yticks=True) for k,result in results.items()  ]
    panels[-1].options.update({ "xticks": True })

    # eKC signal
    panels.append(
        Panel(
            results['eKC'].segments[0].analogsignals[0], 
            xticks=True, yticks=True, 
            xlabel="time(ms)", ylabel="membrane potential (mV)")
    )

    fig = Figure(
        *panels,
        title=title,
        settings=plot_settings
    )

    return fig

# Mushroom Body parameters
def get_params(
    steps, delta_t, t_snapshot, rng, n_input, n_samples, n_eKC, s_iKC=20,
    neuron={}, synapse={}, plasticity={}
    ):

    neuron_params = PARAM_HH_DEFAULT()
    syn_params = PARAM_SYN_DEFAULT(rng)
    stdp_params = PARAM_STDP_DEFAULT()

    neuron_params.update(neuron)
    syn_params.update(synapse)
    stdp_params.update(plasticity)

    return {
        # Population parameters
        "population": {
            "n_iKC": s_iKC * n_input,
            "n_LHI": int(0.2 * n_input),
            "n_eKC": n_eKC,
            "neuron_type": sim.HH_cond_exp(**neuron_params)
        },

        # Synapse parameters
        "synapse": syn_params,

        # Plasticity parameters
        "plasticity": stdp_params,

        # Simulation parameters
        "steps": steps,
        "delta_t": delta_t,
        "t_snapshot": t_snapshot,
        "n_input": n_input,
        "n_samples": n_samples,
        "rng": rng
    }

# Defaults
PARAM_HH_DEFAULT = lambda: ({
    'gbar_Na': 7.15,
    'gbar_K': 1.43,
    'g_leak': 0.0267,

    'e_rev_Na': 50.0,
    'e_rev_K': -95.0,
    'e_rev_leak': -63.56,

    'cm': 0.3
})

PARAM_SYN_DEFAULT = lambda rng: ({
    # PN -> iKC
    "p_PN_iKC": 0.15,
    "g_PN_iKC": RandomDistribution('normal', (4.545, 1.25), rng=rng),
    "t_PN_iKC": 2.0,

    # LHI -> iKC
    "g_LHI_iKC": 8.75,
    "t_LHI_iKC": 3.0,

    # iKC -> eKC
    "p_iKC_eKC": 1.0,
    "p_active": 0.2,
    "g_iKC_eKC_active": RandomDistribution('normal', (1.25, 0.25), rng=rng),
    "g_iKC_eKC_inactive": RandomDistribution('normal', (0.125, 0.025), rng=rng),
    "t_iKC_eKC": 10.0,

    # eKC lateral inhibition
    "g_eKC_inh": 75.0,
    "t_eKC_inh": 5.0
})

PARAM_STDP_DEFAULT = lambda: {
        # Constants
        "c_10": 10 ** 5,
        "c_01": 20,
        "c_11": 5,

        # Time constants
        "t_l": 10 ** 5,
        "t_shift": 10,

        # Initial weight
        "g_0": 0.125,

        # Max strength
        "g_max": 3.75,
}

