import os

import wandb
from torch import optim
import torch

from models import Net
from train import train
from _test import test
from config import get_cfg
#from data_handler import data_set_split
from data_handler_npz import data_set_split
from generator import generator


def create_train_param(cfg, train_data_loader, valid_data_loader):
    # (f, f_copy, opt, data_loader, n_epochs)
    n_epochs = cfg['n_epochs']
    f = Net()
    f_copy = Net()
    opt = optim.Adam(f.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=cfg['steps_for_lr'], gamma=cfg['gamma_for_lr'])
    return f, f_copy, opt, train_data_loader, valid_data_loader, n_epochs, scheduler


def main_learn():

    cfg = get_cfg()
    wandb.init(project='dl4cvproj', name=cfg['run_name'])
    wandb.login(key='33da02b9d54fe7947827d5cce1404a6e7f8ebcd5')
#    path_to_data = f'{os.getcwd()}\\png_files' if cfg['os'] == 'nt' else f'{os.getcwd()}/png_files'
    path_to_data = f'{os.getcwd()}\\{cfg["data_dir"]}' if cfg["os"] == 'nt' else f'{os.getcwd()}/{cfg["data_dir"]}'
    data_loader_train, data_loader_valid, data_loader_test = data_set_split(
        path_to_data,
        batch_size=128)
    param = create_train_param(cfg, data_loader_train, data_loader_valid)
    train(*param, cfg)
    with torch.no_grad():
        test(param[1], data_loader_test, cfg)
    with torch.no_grad():
        generator(param[1], cfg)


if __name__ == '__main__':
    main_learn()
