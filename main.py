import os
from torch import optim

from models import encoder_decoder_net_notes_first
from train import train
from test import test
from config import get_cfg
from data_handler import data_set_split
from generator import generator


def create_train_param(cfg, train_data_loader, valid_data_loader):
    # (f, f_copy, opt, data_loader, n_epochs)
    n_epochs = cfg['n_epochs']
    f = encoder_decoder_net_notes_first()
    f_copy = encoder_decoder_net_notes_first()
    opt = optim.Adam(f.parameters(), lr=cfg['lr'])
    return f, f_copy, opt, train_data_loader, valid_data_loader, n_epochs


def main_learn():
    cfg = get_cfg()
    path_to_data = f'{os.getcwd()}\\png_files' if cfg['os'] == 'nt' else f'{os.getcwd()}/png_files'
    data_loader_train, data_loader_valid, data_loader_test = data_set_split(
        path_to_data,
        batch_size=2)
    param = create_train_param(cfg, data_loader_train, data_loader_valid)
    train(*param, cfg['device'])
    test(param[1], data_loader_test, cfg['device'])
    generator(param[1], cfg['device'])


if __name__ == '__main__':
    main_learn()
