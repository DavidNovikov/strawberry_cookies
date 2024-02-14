# a class which reads the data files and return tensors,
# 1) data loader => get_item()

import torch
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms


def data_set_split(path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, batch_size=32, shuffle=True):
    # Define transformations to be applied to the images

    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor()
    ])

    # Create a dataset from the images at the specified path
    dataset = ImageFolder(root=path, transform=transform)  # you can add a transformation here

    # Calculate the sizes of the training, validation, and test sets
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_valid = int(valid_ratio * num_samples)
    num_test = num_samples - num_train - num_valid

    # Split the dataset into training, validation, and test sets
    train_set, valid_set, test_set = torch.utils.data.random_split(
        dataset, [num_train, num_valid, num_test],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Create data loaders for each set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader


class Config:
    seed = 2509
    num_workers = 1
    num_folds = 8

    num_measurments = 200
    num_r = 100

    # num_Res = 4
    # num_P = 180

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Params related to documentation
    result_path = os.path.dirname(__file__) + '/results'
    create_results_dir = True
    create_code_copy = True  # save a copy of the code in the results folder to easily match code changes to results
    save_results = True
    exp_name = 'test'


    path_data = r'/home/projects/yonina/kvichy/data/radar/SISO_data_pp/'


    # my changes

    learning_rate = 0.0002
    num_epochs = 10**3
    val_interval = 10
    batch_size = 32



    def __init__(self):
        print('Init config successfully')