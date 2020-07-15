import numpy as np

def generate_spike_arrays(input_set, labels, intervals, snapshot_duration=50, noise=0.0):
    """
    Generate spike array times from set of binary inputs and spike intervals

    :input_set: Set of binary arrays coding for snapshot patterns
    :intervals: Times (in ms) of each spike during simulation
    :snapshot_duration: Duration of a single snapshot
    :noise: Amount of gaussian noise to vary 
    """

    n_inputs = len(input_set)
    n_intervals = len(intervals)

    arrays = [ ]

    # Sample all inputs at least once
    inputs = np.random.choice(n_inputs, n_inputs, replace=False)

    # Resample gap between n_interval > n_input
    n_diff = n_intervals - n_inputs
    diff = np.random.choice(n_inputs, n_diff, replace=True)

    inputs = np.concatenate([inputs,diff], axis=0)

    # Select from inputs
    samples = input_set[inputs]
    sample_labels = labels[inputs]

    for inp,i in zip(samples,intervals):
        snapshot = generate_snapshot(i, inp, snapshot_duration, noise=noise)
        arrays.append(snapshot)
    
    return list(map(to_spike_array, np.concatenate(arrays, axis=1).tolist())), sample_labels

def generate_snapshot(start_time, input, duration=50, noise=0.0):
    snapshot = input * start_time
    jitter = np.random.normal(0, noise, snapshot.shape)
    out = np.clip(snapshot + jitter, start_time, start_time+duration) * input

    return out.reshape(-1,1)


def dummy_snapshot(start, d_input, rate=0.1):
    spikes = np.zeros((d_input, 1))
    active = np.random.choice(d_input, int(d_input*0.1), replace=False)
    spikes[active] = 1.0

    return spikes


def to_spike_array(arr):
    times = []
    for spike in arr:
        if spike > 0:
            times.append(spike)
    
    return times