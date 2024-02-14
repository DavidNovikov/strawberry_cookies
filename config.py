# general config

import torch


def get_cfg():
    dct = {
        'lr':  0.001,
        'n_epochs': 5,
        'device': 'mps' if torch.has_mps else 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    return dct
