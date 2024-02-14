import torch
import matplotlib.pyplot as plt


def generator(trained_model, device):

    trained_model = trained_model.to(device)
    trained_model.eval()

    shape = (1, 1, 88, 88)
    #out = torch.bernoulli(torch.full(shape, 0.1))
    out = torch.randn(shape)
    out = (out - out.min()) / (out.max() - out.min())
    out = out.to(device)

    for i in range(10):

        img_np = out.squeeze().detach().numpy()
        plt.imshow(img_np, cmap='gray')  # Assuming it's a grayscale image
        plt.axis('off')  # Hide axis
        plt.show()

        out = trained_model(out)
        # Convert the image tensor to a numpy array and remove the first two dimensions

        # Plot the image
