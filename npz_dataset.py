# -*- coding: utf-8 -*-

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class NPZ_Dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        sample = np.load(str(file_path))
        data = sample['piano_roll']
        torch_array = torch.from_numpy(data).float()
        return torch_array


""" 
class NPZLoader(dataloader.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.files = list(Path(path).glob('*/*.npz'))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        numpy_array = np.load(str(self.files[item]))['piano_roll']
        torch_array = torch.from_numpy(numpy_array)
        if self.transform is not None:
            torch_array = self.transform(torch_array)
        return torch_array, 0
"""  
    