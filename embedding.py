import numpy as np

def spatiotemporal(code, scale, offset=0.0):
    """Map binary matrix spike code to spike times

    :code: Rank 2 tensor with. Dim 0 = Spatial, Dim 1 = Temporal
    :scale: Time scale to map spikes to
    """

    __, t = code.shape
    t_range = scale * np.arange(0,t) + offset

    # Spatiotemporal coding
    spike_times = []
    for spikes in code:
        times = t_range[spikes.nonzero()]
        spike_times.append(times)
    
    return spike_times