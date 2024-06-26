import torch
import os
import torch
import os
import torch.nn.functional as F
import wandb



def symmetric_bce(x, y):
    # return (F.binary_cross_entropy(x, y) + F.binary_cross_entropy(y, x)) / 2
    return (x-y).pow(2).mean()


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


def save_model_and_meta_data(exp_dir, losses, model, best_loss, current_epoch_loss, optimizer):
    """
        This is a checkpointing function, 
        if there is a better model, update the best model, 
        always save the last model
    """
    torch.save({'model_sd': model.state_dict(),
                'opt_sd': optimizer.state_dict(),
                'running_losses': losses},
               f'{exp_dir}/last.pt')

    if current_epoch_loss < best_loss:
        torch.save({'model_sd': model.state_dict(),
                    'opt_sd': optimizer.state_dict(),
                    'running_losses': losses},
                   f'{exp_dir}/best.pt')


def print_testing_to_console(losses):
    """
        Print the testing results to the console
    """
    rec_loss = f"\treconstruction loss:{losses['loss_rec']}\n"
    idem_loss = f"\tidempotent loss:{losses['loss_idem']}\n"
    print(f'test loss:{rec_loss}{idem_loss}')


def test(f, data_loader, device):
    """
        This function runs over the training data and reports the reconstruction and idempotent loss
    """
    f = f.to(device)
    f.eval()
    total_epoch_loss_rec = 0
    total_epoch_loss_idem = 0
    shape = (1, 1, 88, 88)
    for x, _ in data_loader:
        x = x.transpose(1, 2)
        x = x.to(device)
        # z = torch.bernoulli(torch.full(shape, 0.1))
        z = torch.randn_like(x)
        # z = (z - z.min()) / (z.max() - z.min())
        z = z.to(device)

        # apply f to get all needed
        fx = f(x)
        fz = f(z)
        ffz = f(fz)

        # calculate losses
        loss_rec = F.binary_cross_entropy(fx, x)
        loss_idem = symmetric_bce(ffz, fz)

        # optimize for losses
        loss = loss_rec + loss_idem

        # accumulate the loss
        total_epoch_loss_rec += loss_rec
        total_epoch_loss_idem += loss_idem

    print("########################################################")
    print("test")
    print_testing_to_console({'loss_rec': total_epoch_loss_rec / len(data_loader),
                              'loss_idem': total_epoch_loss_idem / len(data_loader)})
