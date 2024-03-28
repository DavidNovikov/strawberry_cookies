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


def make_new_exp():
    """
        This function creates a new directory to store the training results and returns it
    """
    if not os.path.isdir('runs'):
        os.mkdir('runs')
    runs = os.listdir('runs')
    new_exp_dir = f'runs/exp{len(runs)}'
    os.mkdir(new_exp_dir)
    return new_exp_dir

def get_cfg():
    new_exp_dir = make_new_exp()
    dct = {
        'lr':  0.0000001,
        'n_epochs': 200,
        'device': 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu',
        'os': os.name,
        'save_dir': new_exp_dir,
        'data_dir': 'samples_maestro_remove_poor_resolution_0416_drop_high_feq/',
        'run_name': 'adding_noise_range_back_samples_maestro_remove_poor_resolution_0416_drop_high_feq_lr_p0000001_epochs_200',
        'noise': 11,
        'noise_range': 44,
        'rec_loss_w': 0.9,
        'rec_from_noise_loss_w': 0.1,
        'idem_loss_w': 1.0,
        'tight_loss_w': 0.1,
        'gamma_for_lr': 0.9,
        'steps_for_lr': 25,
    }

    return dct
