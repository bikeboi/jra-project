import sys
import glob
import imageio
import numpy as np
from skimage.transform import downscale_local_mean
from skimage.filters import apply_hysteresis_threshold
from sklearn.model_selection import StratifiedShuffleSplit
from brian2 import SpikeMonitor, ms


# Fetching data
def load_images(n_class):
    """
    Load images from Omniglot dataset
    :param n_class: Number of characters to load
    :return: n_class x n_per_class x d_image size dataset
    """
    char_dirs = glob.glob("../omniglot/Korean/**")
    images = np.array([[preprocess(imageio.imread(f)) for f in glob.glob(f"{char_dir}/*")] for i, char_dir in
                       enumerate(char_dirs)]).astype('float')[:n_class]

    return images


def preprocess(img):
    image = 1 - img.astype('float') / 255.
    image = downscale_local_mean(image, (10, 10))
    image = apply_hysteresis_threshold(image, 0.2, 0.4)
    return image.flatten()


# Get spike trains
def get_trains(sm: SpikeMonitor):
    return [train / ms for train in sm.spike_trains().values()]


# Encode images into spiketrains
def spike_encode(image, label, n_class, t, t_var, n_iter=3, sample_period=500):
    """
    Encode image into spike pattern
    :param image: 1-D - flattened image
    :param label: int - image level
    :param n_class: int - number of classes
    :param t: float - time of pattern onset
    :param t_var: float -  Variance in pattern onset time per neuron
    :param n_iter: int - Number of repetitions per sample
    :param sample_offset: int - Time taken per sample
    :return: Matrix of spike patterns
    """

    d_img = len(image)
    segment_ix = label * d_img

    # Generate indices and times
    ixs = np.flatnonzero(image) + segment_ix
    ts = np.full(len(ixs), t * sample_period)

    # Iterations
    iter_period = 50
    ixs = np.tile(ixs, n_iter)
    ts = np.concatenate([ts + x * iter_period for x in range(n_iter)])

    # Temporal noise
    ts = np.clip(ts + np.random.normal(0, t_var, ts.shape), 0, None)

    return ixs, ts


def encode_dataset(images, train_ratio, n_epoch, t_var=0, n_iter=3, sample_period=500):
    """
    Encode dataset of images to spikes
    :param images: n_class x n_sample x d_image size dataset
    :param train_ratio: Fraction of inputs to be used for training
    :param n_epoch: Number of training periods before testing
    :param t_var: Variance in per-neuron spike onset time per-sample
    :return: spike encoded images
    """

    # Retrieve Samples
    n_class, n_per_class, d_image = images.shape
    samples = images.reshape(n_class * n_per_class, d_image)  # Flatten dataset
    labels = np.concatenate([np.full(n_per_class, l) for l in range(n_class)])  # Generate labels

    # Train-Test Split
    xs, ys = [], []
    (train, test), = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio).split(samples, labels)
    xs = np.concatenate([np.tile(samples[train], [n_epoch,1]),  # Repeat train data for n_epoch
                         samples[test]])
    ys = np.concatenate([np.tile(labels[train], n_epoch),
                         labels[test]])


    # Spike Encoding
    spike_ixs = []
    spike_ts = []
    for i, (img, label) in enumerate(zip(xs, ys)):
        ixs, ts = spike_encode(img, label, n_class, i, t_var, n_iter=n_iter, sample_period=sample_period)
        spike_ixs.append(ixs)
        spike_ts.append(ts)

    spike_ixs = np.concatenate(spike_ixs).reshape(-1, 1)
    spike_ts = np.concatenate(spike_ts).reshape(-1, 1)

    return np.concatenate([spike_ixs, spike_ts], axis=1), ys


class ProgressBar(object):
    def __init__(self, toolbar_width=40):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % (" " * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1))  # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed - self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("\n")
