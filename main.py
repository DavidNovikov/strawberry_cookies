import os

import wandb
from torch import optim

from models import Net
from train import train
from test import test
from config import get_cfg
from data_handler import data_set_split
from generator import generator


def create_train_param(cfg, train_data_loader, valid_data_loader):
    # (f, f_copy, opt, data_loader, n_epochs)
    n_epochs = cfg['n_epochs']
    f = Net()
    f_copy = Net()
    opt = optim.Adam(f.parameters(), lr=cfg['lr'])
    return f, f_copy, opt, train_data_loader, valid_data_loader, n_epochs


def main_learn():

    cfg = get_cfg()
    wandb.init(project='dl4cvproj', name=cfg['run_name'])
    wandb.login(key='5687569e35bdb10f530f4efa1312a7169e5cb3c3')
    path_to_data = f'{os.getcwd()}\\png_files' if cfg['os'] == 'nt' else f'{os.getcwd()}/png_files'
    data_loader_train, data_loader_valid, data_loader_test = data_set_split(
        path_to_data,
        batch_size=128)
    param = create_train_param(cfg, data_loader_train, data_loader_valid)
    train(*param, cfg['device'])
    test(param[1], data_loader_test, cfg['device'])
    #generator(param[1], cfg['device'])


if __name__ == '__main__':
    main_learn()
