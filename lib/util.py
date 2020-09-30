import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pynn_genn as sim
import pyqtgraph as pg
from pyNN.utility import ProgressBar
from neo.io import PickleIO


def fetch_results(experiment_name, version):
    prefix = f"results/{experiment_name}_{version}"
    return {name: PickleIO(f"{prefix}/{name}.pickle").read_block() for name in ["PN", "iKC", "eKC"]}


def fetch_params(experiment_name, version):
    path = f"results/{experiment_name}_{version}/params.npz"
    return np.load(path, allow_pickle=True)


def fetch_weights(experiment_name, version):
    return np.load(f"results/{experiment_name}_{version}/weights.npy")


def calculate_steps(n_sample, t_snapshot):
    return (
            t_snapshot  # Initial buffer
            + n_sample * t_snapshot  # Active period
            + t_snapshot  # End buffer
    )


def probability_conn_list(n_pre, n_post, p, *params):
    """Generate connection list with probabilistic connections

    :n_pre: Size of pre-synaptic population
    :n_post: Size of post-synaptic population
    :p: Connection probability
    :params: Synapse params
    """

    possible_conns = n_pre * n_post
    expected_conns = int(p * possible_conns)

    c1 = np.random.choice(n_pre, expected_conns)
    c2 = np.random.choice(n_post, expected_conns)

    return [(c1[i], c2[i], *params) for i in range(expected_conns)]


# Mushroom Body container class
class MushroomBody:
    """Wrapper class for the MB model"""

    def __init__(self, PN, iKC, eKC, PN_iKC, iKC_eKC):
        self.pop = {"PN": PN, "iKC": iKC, "eKC": eKC}
        self.proj = {"PN_iKC": PN_iKC, "iKC_eKC": iKC_eKC}
        self._all = sim.Assembly(PN, iKC, eKC)

    def pop_set(self, **params):
        self._all.set(**params)

    def record(self, records):
        """
        Record population variables

        :records: Dictionary of key=population, val=array of variables to record
        """
        for target, variables in records.items():
            self.pop[target].record(variables)

    def write_data(self, io):
        for pop in self.pop.values():
            pop.write_data(io)


# Callbacks

class WeightLogger:
    """Weight Logging"""

    def __init__(self, projection, log_freq, filepath):
        self.projection = projection
        self.log_freq = log_freq
        self.buffer = []
        self.filepath = filepath

    def __call__(self, t):
        if t < 1:
            return t + 1
        # Fetch weights
        weights = self.projection.get('weight', format='array')

        # Perform some reductions for optimization
        weights = weights.flatten()  # Flatten (don't need spatial info)
        weights = weights[~np.isnan(weights)]  # Ignore NaNs

        self.buffer.append(weights)

        return t + self.log_freq - 1

    def save(self):
        np.save(self.filepath, np.array(self.buffer))


# Progress Bar
class ProgBar:
    """Progress Bar"""

    def __init__(self, steps, tick_freq=10):
        """
        :steps: Total number of steps
        :tick_freq: Update frequency
        """

        self.bar = ProgressBar()
        self.steps = steps
        self.tick_freq = tick_freq

    def __call__(self, t):
        """
        :t: Current timestep
        """
        self.bar.set_level(t / self.steps)
        return t + self.tick_freq


class Results:
    def __init__(self, base_dir, versions):
        self.base_dir = base_dir
        prefixes = [f"{self.base_dir}/{version}" for version in versions]
        self.params = [np.load(f"{prefix}_params.npz") for prefix in prefixes]
        self.data = [[PickleIO(f"{pref}_{r}") for r in range(p['runs'])] for pref, p in
                     zip(prefixes, self.params)]

    def get_run(self, ix):
        ix = np.atleast_1d(np.array(ix))
        if len(ix) == 1:
            return self.data[ix[0]]
        elif len(ix) == 2:
            vix,rix = ix
            return self.data[vix][rix]


# PLotting
def plot_rasters(model: sim.Assembly, duration, t_snapshot=10):
    sns.set_style("white")
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))

    plt.subplots_adjust(hspace=0.5)

    for ax, pop in zip(axs, model.populations):
        spiketrains = pop.get_data().segments[0].spiketrains
        ax.eventplot(spiketrains, colors='k')
        ax.set_title(pop.label)
        ax.set_xlabel("step (ms)")
        ax.set_ylabel("neuron index")
        ax.set_xlim(left=-t_snapshot, right=duration + t_snapshot)

    return fig
