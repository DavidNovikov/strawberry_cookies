from torch import optim

from models import encoder_decoder_net
from train import train
from test import test
from config import get_cfg
from data_handler import data_set_split


def create_train_param(cfg, train_data_loader, valid_data_loader):
    #(f, f_copy, opt, data_loader, n_epochs)
    n_epochs = cfg['n_epochs']
    f = encoder_decoder_net(1)
    f_copy = encoder_decoder_net(1)
    opt = optim.Adam(f.parameters(), lr=cfg['lr'])
    return f, f_copy, opt, train_data_loader, valid_data_loader, n_epochs




def main_learn():
    cfg = get_cfg()
    data_loader_train, data_loader_valid, data_loader_test = data_set_split(
        "C:\\Users\\97254\\Desktop\\Masters-semester-A\\Deep_Learning_for_Computer_Vision_Fundamentals_and_Applications\\final_project\\code\\png_files\\",
    batch_size = 2)
    param = create_train_param(cfg, data_loader_train, data_loader_valid)
    train(*param)
    test(param[1], data_loader_test)



if __name__ == '__main__':
    main_learn()


