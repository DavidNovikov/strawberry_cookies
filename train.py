import torch


def train(f, f_copy, opt, data_loader, n_epochs):
    for epoch in range(n_epochs):
        for x in data_loader:
            z = torch.randn_like(x)

            # apply f to get all needed
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f(f_z)
            f_fz = f_copy(fz)

            # calculate losses
            loss_rec = (fx - x).pow(2).mean()
            loss_idem = (f_fz - fz).pow(2).mean()
            loss_tight = -(ff_z - f_z).pow(2).mean()

            # optimize for losses
            loss = loss_rec + loss_idem + loss_tight * 0.1
            opt.zero_grad()
            loss.backward()
            opt.step()
