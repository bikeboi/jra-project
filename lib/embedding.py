import numpy as np
from pyNN.parameters import Sequence

def spike_encode(inputs, t_snapshot=50, start_time=0):

    n_per_class = inputs.shape[1]
    n_class = inputs.shape[0]
    labels = np.concatenate([ np.full(n_per_class, c) for c in range(n_class) ], axis=0)
    
    # Randomly sample from input set
    samples = np.concatenate(inputs, axis=0)
    n_sample = len(samples)
    ixs = np.arange(len(samples))
    np.random.shuffle(ixs)
    
    samples = samples[ixs]
    labels = labels[ixs]
    
    # Encode into spike snapshots
    intervals = np.arange(start_time + 1, n_sample * t_snapshot, t_snapshot)
    snapshots = []
    for i,pattern in zip(intervals, samples):
        masked = pattern * i
        snapshot = [ np.array([x - 1]) if x != 0 else np.array([]) for x in masked ]
        snapshots.append(snapshot)
    
    # Transpose and eliminate empty
    snapshots = np.transpose(np.array(snapshots, dtype='object'))
    snapshots = [Sequence([ s[0] for s in train if len(s) > 0 ]) for train in snapshots ]
    
    return snapshots, labels

"""
input_set = np.array([
    [[1, 0], [1, 0]],
    [[0, 1], [0, 1]],
])

enc, lab = spike_encode(input_set)
for train in enc:
    print(train)
"""