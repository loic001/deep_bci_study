import os
import json
import numpy as np
from torch.utils.data import Dataset

class MemmapDataset(Dataset):
    def __init__(self, dir, name):
        self.dir = dir

        with open(os.path.join(self.dir, '{}_memmap_dict.json'.format(name)), encoding='utf-8') as f:
            memmap_dict = json.loads(f.read())
        self.data_shape = tuple(memmap_dict['data_shape'])
        self.target_shape = tuple(memmap_dict['target_shape'])
        self.data_memmap_file = os.path.join(dir, '{}_data.mem'.format(name))
        self.target_memmap_file = os.path.join(dir, '{}_target.mem'.format(name))
        self.data = np.memmap(self.data_memmap_file , dtype='float32', mode='r', shape=self.data_shape)
        # self.data = np.full((20000, 55, 600), 5.0)
        print(self.data.nbytes)
        self.target = np.memmap(self.target_memmap_file, dtype='long', mode='r', shape=self.target_shape)
        assert self.data.shape[0] == self.target.shape[0]
        self.total_size = self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx].__array__(), self.target[idx])

    def __len__(self):
        return self.total_size
