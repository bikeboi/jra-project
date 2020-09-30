import os
import numpy as np
from PIL import Image
from skimage.transform import downscale_local_mean
from skimage.filters import apply_hysteresis_threshold


# Utilities for working with data
# Get input samples
def get_inputs(data_dir, n_class, n_epoch, downscale=1, test_frac=0.33, shuffle=True, transform=None):
    # Get n_class character sets from the "Alphabet of the Magi" alphabet
    dataset = Alphabet(data_dir, 0)
    inputs = dataset[:n_class, :]

    n_per_class = inputs.shape[1]
    n_sample = n_class * n_per_class

    # Centering
    # TODO: Write algorithm to center images

    # Invert values
    inputs = 1 - inputs

    # Downsample
    inputs = np.array([[downscale_local_mean(im, (downscale, downscale)) for im in cs] for cs in inputs])  # Downscale
    inputs = np.array([[apply_hysteresis_threshold(im, 0.25, 0.4) for im in cs] for cs in inputs])  # Thresholding

    # Flatten
    inputs = inputs.reshape(n_sample, -1)
    labels = np.concatenate([np.full(n_per_class, i) for i in range(n_class)])

    # Shuffle
    if shuffle:
        ixs = np.arange(n_sample)
        np.random.shuffle(ixs)
        inputs = inputs[ixs]
        labels = labels[ixs]

    if transform:
        inputs, labels = transform(inputs, labels, n_class=n_class)

    # Train Test Split
    n_test = int(n_sample * test_frac)
    train_x, train_y = inputs[n_test:], labels[n_test:]
    test_x, test_y = inputs[:n_test], labels[:n_test]

    # Repeat for epochs
    train_xs = []
    train_ys = []
    for __ in range(n_epoch):
        train_xs.append(train_x)
        train_ys.append(train_y)
        ixs = np.arange(len(train_x))
        np.random.shuffle(ixs)
        train_x = train_x[ixs]
        train_y = train_y[ixs]

    return (
        np.concatenate(train_xs, axis=0), np.concatenate(train_ys, axis=0),
        test_x, test_y
    )


# Transforms
def separable(xs, ys, n_class=2):
    d_in = xs.shape[-1]
    step = int(d_in / n_class)

    for i,s in enumerate(np.arange(0, d_in, step)):
        xs[ys == i, :s] = 0
        xs[ys == i, s+step:] = 0
        xs[ys == i, s:s+step] = np.random.binomial(1, .5, xs[ys == i, s:s+step].shape)

    return xs, ys


class Dataset:

    def __init__(self, **params):
        self.params = params

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, ix):
        raise NotImplementedError


class Alphabet(Dataset):
    def __init__(self, root_dir, id, **params):
        super().__init__(**params)

        name = sorted(os.listdir(root_dir))[id]
        data_dir = f"{root_dir}/{name}"
        char_paths = [
            f"{data_dir}/{path}" for path in sorted(os.listdir(data_dir))]

        self.paths = np.array([np.array(
            [f"{char_path}/{style_path}" for style_path in sorted(os.listdir(char_path))]) for char_path in char_paths])

        # Metadata
        self.num_chars = len(os.listdir(data_dir))

    def __getitem__(self, ix):
        if type(ix) == type((0,)):
            char, style = ix
            image_paths = np.atleast_2d(self.paths[char, style])
        else:
            image_paths = np.atleast_2d(self.paths[ix])

        images = []
        for char in image_paths:
            styles = np.array(
                [np.asarray(Image.open(style_path), dtype=np.float) for style_path in char])
            images.append(styles)

        return np.array(images)
