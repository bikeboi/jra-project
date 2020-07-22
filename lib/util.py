import numpy as np
import pynn_genn as sim
from pyNN.utility import ProgressBar
from neo.io import PickleIO


def retrieve_results(experiment_name, version=0):
    prefix = f"results/{experiment_name}_{version}"
    return {name: PickleIO(f"{prefix}/{name}.pickle").read_block() for name in ["PN", "iKC", "eKC"]}


def log_weights_callback(projection, log, freq):
    def callback(t):
        weights = projection.get('weight', format='array')
        log.append(weights)
        return t + freq

    return callback


def calculate_steps(n_input, t_snapshot):
    return (
        t_snapshot  # Initial buffer
        + n_input * t_snapshot  # Active period
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

    def __init__(self, PN, iKC, eKC, PN_iKC, iKC_eKC):
        self.pop = {"PN": PN, "iKC": iKC, "eKC": eKC}
        self.proj = {"PN_iKC":  PN_iKC, "iKC_eKC": iKC_eKC}
        self._all = sim.Assembly(PN, iKC, eKC)

    def record(self, records):
        for target, variables in records.items():
            self.pop[target].record(variables)

    def write_data(self, io):
        for pop in self.pop.values():
            pop.write_data(io)


# Callbacks

class WeightLogger:
    """Weight Logging
    """

    def __init__(self, projection, log_freq, filepath):
        self.projection = projection
        self.log_freq = log_freq
        self.log = []
        self.filepath = filepath
    
    def __call__(self, t):
        weights = self.projection.get('weight', format='array')
        self.log.append(weights)
        return t + self.log_freq
    
    def finalize(self):
        np.save(self.filepath, np.array(self.log))


# Progress Bar
class ProgBar:
    """Progress Bar class
    """

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
        self.bar.set_level(t/self.steps)
        return t + self.tick_freq
