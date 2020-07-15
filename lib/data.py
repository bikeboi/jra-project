import os
import numpy as np
from PIL import Image

# Utilities for working with data

class Dataset:
    
    def __init__(self, **params):
        self.params = params
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, ix):
        raise NotImplementedError



class Omniglot(Dataset):

    def __init__(self, data_dir, train=True, invert=True):
        self.clamped = [None, None, None]
        self.invert = invert

        root = data_dir + ("/train" if train else "/test")
        
        self.alphabets = sorted(os.listdir(root))

        # Path lookup table
        self.lookup_ = { alpha:
            [
                [
                    f"{root}/{alpha}/{char}/{style}" for k,style in enumerate(sorted(os.listdir(f"{root}/{alpha}/{char}")))
                ]
                for j,char in enumerate(sorted(os.listdir(f"{root}/{alpha}")))
            ]
            for i,alpha in enumerate(sorted(os.listdir(f"{root}")))
        }
    
    def __getitem__(self, ix):
        alphabets = np.atleast_1d(self.alphabets[ix[0]])

        to_slice = lambda x: slice(x,x+1) if type(x) == type(0) else x

        char = to_slice(ix[1]) if len(ix) > 1 else slice(None)
        style = to_slice(ix[2]) if len(ix) > 2 else slice(None)

        def load_img(path):
            return np.asarray(Image.open(path), dtype='float')

        paths = [ np.array(self.lookup_[alphabet])[char, style].tolist() for alphabet in alphabets ]

       
        return [ np.squeeze(np.array([ [ load_img(s) for s in c ] for c in a ])) for a in paths ]


ROOT = "experiment_1/data/omniglot"

dataset = Omniglot(ROOT)
sample = dataset[0,:2,0]

print(sample[0].shape)