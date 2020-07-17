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


class Alphabet(Dataset):
    def __init__(self, root_dir, id, **params):
        super().__init__(**params)

        name = sorted(os.listdir(root_dir))[id]
        data_dir = f"{root_dir}/{name}"
        char_paths = [ f"{data_dir}/{path}" for path in sorted(os.listdir(data_dir)) ]

        self.paths = np.array([ np.array([ f"{char_path}/{style_path}" for style_path in sorted(os.listdir(char_path)) ]) for char_path in char_paths ])

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
            styles = np.array([ np.asarray(Image.open(style_path), dtype=np.float) for style_path in char ])
            images.append(styles)
        
        return np.array(images).squeeze()
