import torch
import matplotlib.pyplot as plt
import wandb
from img2midi import image2midi
import numpy as np
from PIL import Image
from models import Net


def remove_surrounded_nonzeros_rows(tensor, n=1):
    # Get the dimensions of the input tensor
    rows, cols = tensor.shape
    
    for k in range(n+1):

      # Iterate over each row
      for i in range(rows):
          # Initialize a count for consecutive non-zero elements
  
          # Iterate over each element in the row
          for j in range(cols):
              # If the current element is zero, reset the count
              if torch.any(tensor[i, j : j + k] != 0)  and ((j != 0 and tensor[i, j-1 :j] ==0) or j == 0) and \
                  ((j +k < cols and tensor[i, j+k :j+k+1] ==0) or j +k >= cols):
                  tensor[i, j: j + k] = 0

    return tensor

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




def generator(trained_model, cfg, song_length=132, epoch=None):
    device = cfg['device']
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
        shape = None
        prev_song = None
        if 'noise' in cfg:
            shape = (1, 88, 1, cfg['noise'])
        else:
            shape = (1, 88, 1, 44)
        new_noise = torch.randn(shape).to(device)
        if 'noise' in cfg:
            prev_song = generated_song[:,:,:,-(88-cfg['noise']):].transpose(1,2)
        else:
            prev_song = generated_song[:,:,:,-44:].transpose(1,2)
        concatenated_tensor = torch.cat((prev_song, new_noise), dim=3)
        #plt.imshow((concatenated_tensor.transpose(1,2)>0.25).float().detach().cpu().squeeze(dim=0).squeeze(dim=0), cmap='gray')
        new_song = trained_model(concatenated_tensor).transpose(1,2)
        print('prev_song:', prev_song.shape)
        print('new_noise:', new_noise.shape)
        print('new_song:', new_song[:,:,:,-cfg['noise']:].shape)
        print('concatenated_tensor:', concatenated_tensor.shape)
        #new_song_two = trained_model(new_song_two.transpose(1,2)).transpose(1,2)
        if 'noise' in cfg:
            generated_song = torch.cat((generated_song, new_song[:,:,:,-cfg['noise']:]), dim=3)
        else:
            generated_song = torch.cat((generated_song, new_song[:,:,:,-44:]), dim=3)
        #print(new_song_two)
        
    final = (generated_song>0.25).float().detach().cpu().squeeze(dim=0).squeeze(dim=0)
    final_cleaned = remove_surrounded_nonzeros_rows(final, 1)
    final_np = final.numpy()
    final__cleaned_np = final_cleaned.numpy()
    if 'name' in cfg:
        plt.imsave(f"final_image{cfg['name']}.png", final_np, cmap='gray')
        plt.imsave(f"final_image_cleaned{cfg['name']}.png", final__cleaned_np, cmap='gray')
    elif 'save_dir' in cfg:
        plt.imsave(f"{cfg['save_dir']}/sample_at_{epoch if epoch else 0}.png", final_np, cmap='gray')
    else:
        plt.imsave("final_image.png", final_np, cmap='gray')


    plt.imshow(final, cmap='gray')
    # wandb.log({"final": wandb.Image(final)})
    if 'name' in cfg:
        image2midi(f"final_image{cfg['name']}.png")
        image2midi(f"final_image_cleaned{cfg['name']}.png")
    elif 'save_dir' in cfg:
        image2midi(f"{cfg['save_dir']}/sample_at_{epoch if epoch else 0}.png")
    else:
        image2midi("final_image.png")
    
if __name__ == "__main__":
    
    model_paths = ['runs/exp53/best.pt']
    len_noises = [11]
    for model_path, len_noise in zip(model_paths, len_noises):
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = Net()

        model.load_state_dict(checkpoint['model_sd'])
        cfg = {'device' : 'cpu', 'name': model_path[8:10]
               , 'noise': len_noise
               }
        generator(model, cfg)