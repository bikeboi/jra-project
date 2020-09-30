import numpy as np
from neo.io import PickleIO
from scipy.spatial.distance import euclidean, cosine


# Fetching collected data
def fetch_results(name, params, group='ekc'):
    prefix = f"results/{name}"
    data = [PickleIO(f"{prefix}/data_{group}_{r}.pickle").read()[0].segments[0].spiketrains for r in range(params['runs'])]
    return data


def compute_rates(runs, params, train=False):
    t_snap, intervals = params['t_snapshot'], (params['train_intervals'] if train else params['test_intervals'])
    return np.array([calculate_activity(trains, intervals, t_snap) for trains in runs])


def overlap_distance(activity, labels):
    per_class = {c: len(labels[labels == c]) for c in np.unique(labels)}
    total_activity = activity.sum()

    return {c: a / total_activity for c, a in per_class.items()}


# Utility
# Average class vectors
def calculate_class_vectors(activity, labels, return_class_activity=False):
    pcs = per_class_activity(activity, labels)
    cvs = {c: act.mean(axis=1) for c, act in pcs.items()}

    if return_class_activity:
        return cvs, pcs
    else:
        return cvs


def per_class_activity(activity, labels):
    return {c: activity[:, labels == c] for c in np.unique(labels)}


# Calculate activity from spiketrains
def calculate_activity(spiketrains, intervals, t_interval):
    activity = np.empty((len(spiketrains), len(intervals)))

    for i, t in enumerate(intervals):
        neuron_activity = np.array(
            [len(train.time_slice(t, t + t_interval)) for train in spiketrains])
        activity[:, i] = neuron_activity

    return activity


# Utility functions
# Numerically stable cosine distance
def cosine_distance(a, b):
    epsilon = np.finfo(np.float32).eps
    return cosine(a + epsilon, b + epsilon)
