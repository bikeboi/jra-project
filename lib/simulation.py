import numpy as np
import pynn_genn as sim
from pyNN.random import RandomDistribution, NumpyRNG
from pyNN.utility.plotting import Figure, Panel
from pyNN.utility import ProgressBar

from model import default_MB

# Setup experiment
def setup_experiment(input_spikes, model_setup=default_MB, eKC_signal=False, eKC_conductance=False, **params):
    # Setup sim
    sim.setup(params['delta_t'])

    # Build Model
    MB = model_setup(input_spikes, **params)
    MB['model'].record(["spikes"])

    if eKC_conductance:
        MB['model'].get_population('eKC').record('gsyn_exc')

    if eKC_signal:
        MB['model'].get_population("eKC").record("v")

    return MB

def run_experiment(MB, intervals, **params):

    # Progress bar
    pb = ProgressBar()
    steps = params['steps']

    # Log initial values
    print("Running experiment\n")

    # No intermediate logging
    def progbar(t):
        pb.set_level(t/steps)
        return t + params['t_snapshot']
        
    sim.run(params['steps'], callbacks=[progbar])
    logs = log_data(MB, intervals, params['t_snapshot'])

    print()
    print("\nTotal Steps:", f"{steps}ms")

    return logs


# Plotting
def plot_results(data, title, eKC_signal=False, eKC_conductance=False, hide=[], plot_params={}):
    plot_settings = {
        #"lines.linewidth": 1.5,
        #"lines.markersize": 18,
        #"font.size": 14,
        "axes.xmargin": 0
    }

    plot_settings.update(plot_params)

    print("Plotting panels...")

    panels = [ Panel(result.segments[0].spiketrains, data_labels=[k], yticks=True) for k,result in data.items() if not k in hide  ]
    panels[-1].options.update({ "xticks": True })

    # eKC signal
    if eKC_signal:
        panels.append(
            Panel(
                data['eKC'].segments[0].analogsignals[0], 
                xticks=True, yticks=True, 
                xlabel="time(ms)", ylabel="membrane potential (mV)",
            ),
        )
    
    else:
        panels[-1].options.update({ "xlabel": "time(ms)" })

    print("Generating Figure...")
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

## Log data
def log_data(MB, active_intervals, t_snapshot):
    return {
        "data": { pop.label:  pop.get_data() for pop in MB['model'].populations },
        "activity": { pop.label: [ calculate_activity(seg.spiketrains, active_intervals, t_snapshot) for seg in pop.get_data().segments ] for pop in MB['model'].populations },
    }

## Activity metric
def calculate_activity(spiketrains, intervals, t_snapshot):
    activity = []
    for train in spiketrains:
        int_act = [ len([t for t in train if t >= i and t < i + t_snapshot]) for i in intervals ]
        activity.append(int_act)

    activity = np.array(activity)

    return activity


## Generate intervals
def gen_intervals(t_snapshot, n_samples, offset):
    return np.arange(offset, n_samples * t_snapshot + 1, t_snapshot)

def train_intervals(t_snapshot, n_samples):
    return gen_intervals(t_snapshot, n_samples, t_snapshot)