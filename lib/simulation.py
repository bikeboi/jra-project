import numpy as np
import pynn_genn as sim
from pyNN.random import RandomDistribution, NumpyRNG
from pyNN.utility.plotting import Figure, Panel
from pyNN.utility import ProgressBar

from embedding import generate_spike_arrays

# Setup experiment
def setup_experiment(input_set, params, model_setup, labels=None):
    # Setup sim
    sim.setup(params['delta_t'])
    np.random.seed(42) # Magic number ¯\_(ツ)_/¯

    # Input spikes
    intervals = gen_intervals(params['t_snapshot'], params['n_samples'], params['t_snapshot'])
    input_spikes, spike_labels = generate_spike_arrays(
        input_set, np.arange(len(input_set)) if labels is None else labels,
        intervals, 
        params['t_snapshot'], 
        params['snapshot_noise'] if params['snapshot_noise'] else 0.0)

    # Build Model
    MB = model_setup(input_spikes, **params)
    MB['model'].record(["spikes"])
    MB['model'].get_population("eKC").record("v")

    return MB, spike_labels, intervals

def run_experiment(MB, intervals, params):
    # Save initial weights
    initial_weights = { name: proj.get('weight', format='array') for name, proj in MB['projections'].items() }

    sim.run(params['steps'])
    print("Done\n--")

    # Save final weights
    final_weights = { name: proj.get('weight', format='array') for name, proj in MB['projections'].items() }

    return  {
        "data": { pop.label:  pop.get_data() for pop in MB['model'].populations },
        "activity": { pop.label: [ calculate_activity(seg.spiketrains, intervals) for seg in pop.get_data().segments ] for pop in MB['model'].populations },
        "weights": { "initial": initial_weights, "final": final_weights }
    }


# Plotting
def plot_results(data, title):
    plot_settings = {
        "lines.linewidth": 1.5,
        "lines.markersize": 18,
        "font.size": 14,
        "axes.xmargin": 0
    }

    panels = [ Panel(result.segments[0].spiketrains, data_labels=[k], yticks=True) for k,result in data.items()  ]
    panels[-1].options.update({ "xticks": True })

    # eKC signal
    panels.append(
        Panel(
            data['eKC'].segments[0].analogsignals[0], 
            xticks=True, yticks=True, 
            xlabel="time(ms)", ylabel="membrane potential (mV)",
        ),
    )

    fig = Figure(
        *panels,
        title=title,
        settings=plot_settings
    )

    return fig


def cleanup():
    print("Ending Simulation...")
    sim.end()

# Utility
## Activity metric
def calculate_activity(spiketrains, intervals):
    activity = []
    t_snapshot = intervals[-1] - intervals[-2] # Derive snapshot from interval difference
    for start_time in intervals:
        activity.append([ len(spiketrain[(start_time <= spiketrain) & (start_time + t_snapshot > spiketrain)]) for spiketrain in spiketrains ])

    activity = np.array(activity).T

    return activity


## Generate intervals
def gen_intervals(t_snapshot, n_samples, offset):
    return np.arange(offset, n_samples * t_snapshot + 1, t_snapshot)

def train_intervals(t_snapshot, n_samples):
    return gen_intervals(t_snapshot, n_samples, t_snapshot)

## Mushroom Body parameters
def default_params(
    delta_t, t_snapshot, d_input, n_samples, n_eKC, s_iKC=20, test_frac=0.3, rng=NumpyRNG(),
    neuron={}, synapse={}, plasticity={}
    ):

    """
    Generate simulation and model parameters

    :delta_t: Simulation timestep
    :t_snapshot: Duration of an input snapshot
    :d_input: Dimensionality of input samples
    :n_samples: Number of training samples to show
    :n_eKC: Number of eKC neurons
    :s_iKC: Multiplier for number of iKC neurons wrt. eKC neurons
    :test_frac: Fraction of test samples wrt. training samples
    :rng: RNG object for random number generation
    :neuron: Neuron parameter overrides
    :synapse: Synapse parameter overrides
    :plasticity: STDP mechanism overrides
    """

    steps = (
        t_snapshot # Initial pause
        + t_snapshot * n_samples # Input period 1
        #+ t_snapshot * 4 # Cooling period
        #+ t_snapshot * n_samples # Input period 2
        + t_snapshot # Final pause
    )

    neuron_params = PARAM_HH_DEFAULT()
    syn_params = PARAM_SYN_DEFAULT(rng)
    stdp_params = PARAM_STDP_DEFAULT(rng)

    neuron_params.update(neuron)
    syn_params.update(synapse)
    stdp_params.update(plasticity)

    return {
        # Population parameters
        "population": {
            "n_iKC": s_iKC * d_input,
            "n_eKC": n_eKC,
        },

        # Neuron parameters
        "neuron": sim.IF_curr_exp(),

        # Synapse parameters
        "synapse": syn_params,

        # Plasticity parameters
        "plasticity": stdp_params,

        # Simulation parameters
        "steps": steps,
        "delta_t": delta_t,
        "t_snapshot": t_snapshot,
        "n_samples": n_samples,
        "rng": rng,
        "snapshot_noise": 0.0
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
    "g_PN_iKC": RandomDistribution('normal', (5.0, 1.25), rng=rng),
    "t_PN_iKC": 2.0,

    # iKC -> eKC
    "p_iKC_eKC": 1.0,
})

PARAM_STDP_DEFAULT = lambda rng: {
    "weight_dependence": sim.AdditiveWeightDependence(w_min=0.01, w_max=1.0),
    "timing_dependence": sim.SpikePairRule(),
    "weight": RandomDistribution('normal', (0.125, 0.2), rng=rng),
    "delay": 10.0
}

