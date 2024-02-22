import torch
import os
import torch.nn.functional as F
import wandb


def wasserstein_distance(p_samples, q_samples):
    """
    Compute the 1-dimensional Wasserstein distance between two distributions.

    Args:
        p_samples (torch.Tensor): Samples from the first distribution.
        q_samples (torch.Tensor): Samples from the second distribution.

    Returns:
        torch.Tensor: The Wasserstein distance.
    """
    # Sort samples
    p_samples_sorted, _ = torch.sort(p_samples)
    q_samples_sorted, _ = torch.sort(q_samples)

    # Compute the distance
    distance = torch.abs(p_samples_sorted - q_samples_sorted).mean()

    return distance

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


def print_training_to_console(losses):
    """
        Print the training results to the console
    """
    epoch = f"epoch:{len(losses['loss_rec'])}\n"
    rec_loss = f"\treconstruction loss:{losses['loss_rec'][-1]}\n"
    idem_loss = f"\tidempotent loss:{losses['loss_idem'][-1]}\n"
    tight_loss = f"\ttighting loss:{losses['loss_tight'][-1]}\n"
    print(f'{epoch}{rec_loss}{idem_loss}{tight_loss}')


def print_validation_to_console(losses):
    """
        Print the training results to the console
    """
    rec_loss = f"\treconstruction loss:{losses['loss_rec']}\n"
    idem_loss = f"\tidempotent loss:{losses['loss_idem']}\n"
    print(f'{rec_loss}{idem_loss}')


def train(f, f_copy, opt, train_data_loader, valid_data_loader, n_epochs, device):
    """
        This function runs n_epochs epochs and saves the training loss at every epoch
    """
    f = f.to(device)
    f_copy = f_copy.to(device)


    new_exp_dir = make_new_exp()
    tight_loss_coefficient = 0.1

    # losses is the loss per loss type per epoch
    losses = {'loss_rec': [],
              'loss_idem': [],
              'loss_tight': []}
    # best loss is the best loss, if we get a better loss we should update the best.pt
    best_loss = {'total_loss': torch.inf,
                 'loss_rec': torch.inf,
                 'loss_idem': torch.inf,
                 'loss_tight': torch.inf}

    for epoch in range(n_epochs):
        f.train()
        # keep track of the total loss to compare it to the best loss
        total_epoch_loss_rec = 0
        total_epoch_loss_idem = 0
        total_epoch_loss_tight = 0
        shape = (1, 1, 88, 88)
        for x, _ in train_data_loader:
            # z = torch.bernoulli(torch.full(shape, 0.1))
            z = torch.randn_like(x)
            z = (z - z.min()) / (z.max() - z.min())

            # put the data on the device
            x = x.to(device)
            z = z.to(device)

            # apply f to get all needed
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f(f_z)
            f_fz = f_copy(fz)

            # calculate losses
            loss_rec = F.binary_cross_entropy_with_logits(fx, x)
            loss_idem = F.binary_cross_entropy_with_logits(f_fz, fz)
            loss_tight = -F.binary_cross_entropy_with_logits(ff_z, f_z)

            # optimize for losses
            loss = loss_rec + loss_idem + loss_tight * tight_loss_coefficient
            opt.zero_grad()
            loss.backward()
            opt.step()

            # accumulate the loss
            total_epoch_loss_rec += loss_rec
            total_epoch_loss_idem += loss_idem
            total_epoch_loss_tight += loss_tight


        total_epoch_loss_rec = total_epoch_loss_rec / len(train_data_loader)
        total_epoch_loss_idem = total_epoch_loss_idem / len(train_data_loader)
        total_epoch_loss_tight = total_epoch_loss_tight / \
            len(train_data_loader)
        # append the individual losses
        losses['loss_rec'].append(total_epoch_loss_rec)
        losses['loss_idem'].append(total_epoch_loss_idem)
        losses['loss_tight'].append(total_epoch_loss_tight)
        total_epoch_loss = total_epoch_loss_rec + \
            total_epoch_loss_idem + total_epoch_loss_tight

        # checkpointing
        save_model_and_meta_data(
            new_exp_dir, losses, f, best_loss['total_loss'], total_epoch_loss, opt)

        # if we had a better model save it
        if total_epoch_loss < best_loss['total_loss']:
            best_loss['total_loss'] = total_epoch_loss
            best_loss['loss_rec'] = total_epoch_loss_rec
            best_loss['loss_idem'] = total_epoch_loss_idem
            best_loss['loss_tight'] = total_epoch_loss_tight

        # print the training results to the console
        print("########################################################")
        print("train")
        print_training_to_console(losses)

        wandb.log({'Epoch:': epoch,
                   'Training Loss:': total_epoch_loss, 'Train total_epoch_loss_rec': total_epoch_loss_rec,
        'Train total_epoch_loss_idem': total_epoch_loss_idem, 'Train total_epoch_loss_tight': total_epoch_loss_tight})


        valid(f, valid_data_loader, device, epoch)


def valid(f, data_loader, device, epoch):
    """
        This function runs over the training data and reports the reconstruction and idempotent loss
    """
    f = f.to(device)
    f.eval()
    total_epoch_loss_rec = 0
    total_epoch_loss_idem = 0
    shape = (1, 1, 88, 88)
    for x, _ in data_loader:
        z = torch.bernoulli(torch.full(shape, 0.1))
        # z = torch.randn_like(x)
        # z = (z - z.min()) / (z.max() - z.min())

        # put the data on the device
        x = x.to(device)
        z = z.to(device)

        # apply f to get all needed
        fx = f(x)
        fz = f(z)
        ffz = f(fz)

        # calculate losses
        loss_rec = F.binary_cross_entropy_with_logits(fx ,x)
        loss_idem = F.binary_cross_entropy_with_logits(ffz , fz)

        # optimize for losses
        loss = loss_rec + loss_idem

        # accumulate the loss
        total_epoch_loss_rec += loss_rec
        total_epoch_loss_idem += loss_idem

    wandb.log({ 'Epoch:': epoch,'Validation Loss:':  total_epoch_loss_rec / len(data_loader) +  total_epoch_loss_idem / len(data_loader)
                , 'Validation total_epoch_loss_rec:': total_epoch_loss_rec / len(data_loader),
                'Validation total_epoch_loss_idem:':  total_epoch_loss_idem / len(data_loader)})

    print("########################################################")
    print("valid")
    print_validation_to_console({'loss_rec': total_epoch_loss_rec / len(data_loader),
                                 'loss_idem': total_epoch_loss_idem / len(data_loader)})
