import numpy as np
import torch

def add_gaussian(img: torch.tensor, mean: float, variance: float, device:torch.device) -> torch.tensor:
    """
    Adds Gaussian noise to an image.
    Used to simulate read noise.
    Input format: CxHxW
    """
    img_in = torch.clone(img)
    img_in = img_in + torch.randn(img.shape, device=device) * torch.sqrt(variance) + mean
    img_in = torch.clip(img_in,0,1)
    return img_in

def add_poisson(img: torch.tensor, scaling: float, device: torch.device) -> torch.tensor: # TODO: Confirm everything is in order with this function. Yi-Chun found problems with photon count. Reapply the derivation from inverse direction all the way up to lux. 
    """
    Adds Poisson noise to an image.
    Used to simulate shot noise.
    Input format: CxHxW
    """
    img_in = torch.clone(img).to(device)
    img_in = img_in * scaling
    img_in = torch.poisson(img_in)
    img_in = img_in / scaling
    return img_in

def add_salt_and_pepper(img: torch.tensor, saltCount: int, pepperCount: int) -> torch.tensor:
    """
    Adds salt and pepper noise to an image.
    Useful for simulating dead and hot pixels.
    """
    l, h, w = img.shape
    for i in range(saltCount):
        y_coord=np.random.randint(0, w - 1)
        x_coord=np.random.randint(0, h - 1)
        img[:,x_coord,y_coord] = 1
    for i in range(pepperCount):
        y_coord=np.random.randint(0, w - 1)
        x_coord=np.random.randint(0, h - 1)
        img[:,x_coord,y_coord] = 0
    return img