import torch
import matplotlib.pyplot as plt
import wandb


def generator(trained_model, device):

    trained_model = trained_model.to(device)
    trained_model.eval()

    shape = (1, 88, 1, 88)
    #out = torch.bernoulli(torch.full(shape, 0.1))
    out = torch.randn(shape)
    out = out.to(device)

    for i in range(10):

        img_np = out.transpose(1,2).squeeze().detach().cpu().numpy()
        plt.imshow(img_np, cmap='gray')  # Assuming it's a grayscale image
        plt.axis('off')  # Hide axis
        plt.show()
        wandb.log({"example_image{i}": wandb.Image(img_np)})


        out = trained_model(out)
        # Convert the image tensor to a numpy array and remove the first two dimensions

        # Plot the image
