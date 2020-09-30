import numpy as np
from pyNN.parameters import Sequence
from skimage.filters import apply_hysteresis_threshold


def spike_encode(inputs, labels, t_snapshot=50, offset=0, spike_jitter=0):
    # Threshold inputs
    samples = inputs.round()

    # Encode into spike snapshots
    snapshots = []
    for bin_train in samples.T:
        spikes = [(i * t_snapshot) + offset for i, x in enumerate(bin_train) if x > 0]
        snapshots.append(spikes)

    return snapshots


"""
input_set = np.array([
    [[1, 0], [1, 0]],
    [[0, 1], [0, 1]],
])

enc, lab, __ = spike_encode(input_set)
print(lab)
for train in enc:
    print(train)
"""
