# general config
import torch
import os


# val loss: 0.1037 - 0.2 sched
# def get_cfg():
#     dct = {
#         'lr':  0.00001,
#         'n_epochs': 500,
#         'device': 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu',
#         'os': os.name,
#         'run_name': 'run25'
#     }
# 
#     return dct



def get_cfg():
    dct = {
        'lr':  0.00001,
        'n_epochs': 20,
        'device': 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu',
        'os': os.name,
        'run_name': 'run30'
    }

    return dct
