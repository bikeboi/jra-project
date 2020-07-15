import numpy as np
from scipy.spatial.distance import euclidean

# Experiment metrics

def interclass(activity, labels, D=euclidean):
    class_vectors = calculate_class_vectors(activity, labels)

    # Distances
    D_inter = {}
    for c,v in class_vectors.items():
        distance = np.mean([ D(v,v_other) for c_other,v_other in class_vectors.items() if c != c_other ])
        D_inter[c] = distance
    
    return D_inter


def intraclass(activity, labels, D=euclidean):
    class_vectors, per_class = calculate_class_vectors(activity, labels, return_class_activity=True)
    per_class = per_class_activity(activity, labels)

    D_intra = {}
    for c,v in class_vectors.items():
        class_activity = per_class[c]
        distance = np.mean([ D(v, active) for active in class_activity.T ])
        D_intra[c] = distance
    
    return D_intra


def overlap_distance(activity, labels):
    per_class = { c: len(labels[labels == c]) for c in np.unique(labels) }
    total_activity = activity.sum()

    return { c: a / total_activity for c,a in per_class.items() }


# Utility
## Average class vectors
def calculate_class_vectors(activity, labels, return_class_activity=False):
    pcs = per_class_activity(activity, labels)
    cvs = { c: act.mean(axis=1) for c,act in pcs.items() }

    if return_class_activity:
        return cvs, pcs
    else:
        return cvs

def per_class_activity(activity, labels):
    return { c: activity[:,labels == c] for c in np.unique(labels) }