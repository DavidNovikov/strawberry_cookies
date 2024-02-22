# general config
import torch
import os


def get_cfg():
    dct = {
        'lr':  0.0001,
        'n_epochs': 10,
        'device': 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu',
        'os': os.name,
        'run_name': 'run17'
    }

    return dct
