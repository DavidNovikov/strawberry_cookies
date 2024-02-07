from train import train_loop
from test import valid_loop
from config import get_cfg

def main_learn():
    cfg = get_cfg()
    model = train_loop(cfg)
    valid_loop(model)



if __name__ == '__main__':
    main_learn()