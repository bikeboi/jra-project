"""
Sandbox to test out random ideas
"""

import imageio
from skimage.transform import downscale_local_mean
from skimage.filters import apply_hysteresis_threshold
import pickle
import glob
import matplotlib.pyplot as plt
from functools import reduce
import brian2.numpy_ as np
from brian2.only import *

plt.switch_backend("GTK3Agg")

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'image.cmap': 'gray',
})


# Helper functions
def load_images(n_class):
    """
    Load images from Omniglot dataset
    :param n_class: Number of characters to load
    :return: n_class x n_per_class x d_image size dataset
    """
    char_dirs = glob.glob("omniglot/Korean/**")
    images = np.array([[preprocess(imageio.imread(f)) for f in glob.glob(f"{char_dir}/*")] for i, char_dir in
                       enumerate(char_dirs)]).astype('float')[:n_class]

    return images


def preprocess(img):
    image = 1 - img.astype('float') / 255.
    image = downscale_local_mean(image, (10, 10))
    image = apply_hysteresis_threshold(image, 0.2, 0.4)
    return image.flatten()


def get_trains(sm: SpikeMonitor):
    return [train / ms for train in sm.spike_trains().values()]


def spike_encode(image, label, n_class, t, t_var):
    """
    Encode image into spike pattern
    :param image: 1-D - flattened image
    :param label: int - image level
    :param n_class: int - number of classes
    :param t: float - time of pattern onset
    :param t_var: float -  Variance in pattern onset time per neuron
    :return: Matrix of spike patterns
    """

    d_img = len(image)
    segment_ix = label * d_img
    sample_offset = 500

    # Generate indices and times
    ixs = np.flatnonzero(image) + segment_ix
    ts = np.clip(np.random.normal(t * sample_offset, t_var, len(ixs)), 0, None)

    # Iterations
    n_iter = 5
    iter_offset = 50
    ixs = np.tile(ixs, n_iter)
    ts = np.concatenate([ts + x * iter_offset for x in range(n_iter)])

    return ixs, ts


def encode_dataset(images, t_var=0, shuffle=True):
    """
    Encode dataset of images to spikes
    :param images: n_class x n_sample x d_image size dataset
    :param t_var: Variance in per-neuron spike onset time per-sample
    :param shuffle: Shuffle samples
    :return: spike encoded images
    """

    n_class, n_per_class, d_image = images.shape

    samples = images.reshape(n_class * n_per_class, d_image)  # Flatten dataset
    labels = np.concatenate([np.full(n_per_class, l) for l in range(n_class)])  # Generate labels

    if shuffle:
        ixs = np.random.choice(len(samples), len(samples), replace=False)
        samples = samples[ixs]
        labels = labels[ixs]

    spike_ixs = []
    spike_ts = []
    for i, (img, label) in enumerate(zip(samples, labels)):
        ixs, ts = spike_encode(img, label, n_class, i, t_var)
        spike_ixs.append(ixs)
        spike_ts.append(ts)

    spike_ixs = np.concatenate(spike_ixs).reshape(-1, 1)
    spike_ts = np.concatenate(spike_ts).reshape(-1, 1)

    return np.concatenate([spike_ixs, spike_ts], axis=1), labels


# Neuron model
NEURON = """
dv/dt = (-v + v_rest + Isyn)/tau : volt
dIsyn/dt = -Isyn/tau_syn : volt
theta = -50*mV : volt
v_rest = -55*mV : volt
tau = 50*ms : second
tau_syn = 2*ms : second
"""

INHIBITOR = """
dcount/dt = -count/(10*ms) : 1
thresh = 50 : 1
"""

start_scope()
defaultclock.dt = 0.1*ms

# Data
N_CLASS = 2
images = load_images(N_CLASS)
D_IMAGE = images.shape[-1]
spikes, labels = encode_dataset(images)

# Populations
n_pn = N_CLASS * D_IMAGE
n_kc = 15 * n_pn

pns = SpikeGeneratorGroup(n_pn, spikes[:, 0], spikes[:, 1] * ms)
sm_pn = SpikeMonitor(pns)

kcs = NeuronGroup(n_kc, NEURON, threshold='v > theta', reset='v = v_rest', method='exact')
kcs.v = 'v_rest'
sm_kc = SpikeMonitor(kcs)
vm_kc = StateMonitor(kcs, 'v', record=[0])
pm_kc = PopulationRateMonitor(kcs)

inh = NeuronGroup(1, INHIBITOR, threshold='count > thresh', reset='count = 0', method='exact')
vm_inh = StateMonitor(inh, 'count', True)

# Connections
pn_kc = Synapses(pns, kcs, 'w : volt', on_pre='Isyn += w', delay=2*ms)
pn_kc.connect(p=0.1)
pn_kc.w = 25 * mV

kc_inh = Synapses(kcs, inh, on_pre='count += 1')
kc_inh.connect()

inh_kc = Synapses(inh, kcs, 'g : volt', on_pre="""
Isyn = -g
v = v_rest
""")
inh_kc.connect()
inh_kc.g = 55 * mV

duration = 5*second #N_CLASS * 20 * second / 2
run(duration)

pn_spikes = get_trains(sm_pn)
kc_spikes = get_trains(sm_kc)

input_rates = np.concatenate(pn_spikes)
output_rates = np.concatenate(kc_spikes)

print(sm_kc.num_spikes)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

ax1.eventplot(pn_spikes)
ax1.set_title("Input Spikes")
ax2.eventplot(kc_spikes)
ax2.set_title("KC Spikes")
ax3.plot(vm_inh.t / ms, vm_inh.count[0])
ax3.set_title("Inhibitor Count")
ax3.set_ylabel("Number of spikes counted")
ax3.set_xlabel("time [ms]")

plt.show()
