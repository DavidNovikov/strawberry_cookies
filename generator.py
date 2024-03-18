import torch
import matplotlib.pyplot as plt
import wandb
from img2midi import image2midi
import numpy as np
from PIL import Image



def imshow(x, norm=False):
    if torch.is_tensor(x):
        x = x.detach().cpu()
    if x.shape[0] <= 3:
        x = torch.einsum('chw->hwc', x)
    if x.shape[-1] == 1:
        plt.imshow(x.squeeze(-1), cmap='gray')
    else:
        plt.imshow(x)
    plt.show()




def generator(trained_model, device, song_length=132):
    trained_model = trained_model.to(device)
    trained_model.eval()

    # Generate initial random noise
    shape = (1, 88, 1, 88)

    initial_noise = torch.randn(shape).to(device)

    # Generate the initial song from the random noise
    generated_song = trained_model(initial_noise).transpose(1,2)
    #plt.imshow((generated_song > 0.25).float().detach().cpu().squeeze(dim=0).squeeze(dim=0), cmap='gray')

    # Iterate to generate the full song
    for _ in range(5):
        shape = (1, 88, 1, 44)
        new_noise = torch.randn(shape).to(device)
        prev_song = generated_song[:,:,:,-44:].transpose(1,2)
        concatenated_tensor = torch.cat((prev_song, new_noise), dim=3)
        #plt.imshow((concatenated_tensor.transpose(1,2)>0.25).float().detach().cpu().squeeze(dim=0).squeeze(dim=0), cmap='gray')
        new_song = trained_model(concatenated_tensor).transpose(1,2)
        #new_song_two = trained_model(new_song_two.transpose(1,2)).transpose(1,2)
        generated_song = torch.cat((generated_song, new_song[:,:,:,-44:]), dim=3)
        #print(new_song_two)
        
    final = (generated_song>0.25).float().detach().cpu().squeeze(dim=0).squeeze(dim=0)
    final_np = final.numpy()
    plt.imsave("final_image.png", final_np, cmap='gray')


    plt.imshow(final, cmap='gray')
    wandb.log({"final": wandb.Image(final)})
    image2midi("final_image.png")