import numpy as np
import torch
from skimage.metrics import structural_similarity

def MSE(orig: torch.tensor, test: torch.tensor, return_map: bool = False):
    """
    Mean squared error between two images.
    return_map: if True, returns the local values of MSE.
    Input format: CxHxW
    """
    assert orig.shape == test.shape
    if type(orig) == np.ndarray:
        orig_i = np.copy(orig)
        orig_i = torch.tensor(orig_i)
    else:
        orig_i = torch.clone(orig)
    if type(test) == np.ndarray:
        test_i = np.copy(test)
        test_i = torch.tensor(test_i)
    else:
        test_i = torch.clone(test)
    diff = orig_i - test_i
    if return_map:
        return torch.sum(torch.square(diff)) / diff.numel(), torch.sum(torch.square(diff), axis=0)
    else:
        return torch.sum(torch.square(diff)) / diff.numel()
    
def PSNR(img1: torch.tensor, img2: torch.tensor, return_map: bool = False):
    """
    Peak signal-to-noise ratio between two images.
    return_map: if True, returns the local values of PSNR.
    """
    assert img1.shape == img2.shape
    if type(img1) == np.ndarray:
        img1_i = np.copy(img1)
        img1_i = torch.tensor(img1_i)
    else:
        img1_i = torch.clone(img1)
    if type(img2) == np.ndarray:
        img2_i = np.copy(img2)
        img2_i = torch.tensor(img2_i)
    else:
        img2_i = torch.clone(img2)
    if return_map:
        mse, local_mse  = MSE(img1_i, img2_i, return_map = True)
        local_psnr = 10 * torch.log10(1/(local_mse+1e-10))
        psnr = 10 * torch.log10(1/(mse+1e-10))
        return psnr, local_psnr
    else:
        mse = MSE(img1, img2, return_map = False)
        psnr = 10 * torch.log10(1/(mse+1e-10))
        return psnr
    
def SSIM(img1: torch.tensor, img2: torch.tensor):
    """
    Structural similarity index between two images.
    There is no local value for SSIM.
    Input format: CxHxW
    """
    if type(img1) == np.ndarray:
        img1_i = np.copy(img1)
    else:
        img1_i = torch.clone(img1)
        img1_i = img1_i.detach().cpu().numpy()
    if type(img2) == np.ndarray:
        img2_i = np.copy(img2)
    else:
        img2_i = torch.clone(img2)
        img2_i = img2_i.detach().cpu().numpy()
    if len(img1_i.shape) == 2:
        return torch.tensor(structural_similarity(img1_i, img2_i, data_range = 1.0))
    else:
        return torch.tensor(structural_similarity(img1_i, img2_i, data_range = 1.0, channel_axis = 0)) 

def SAM(orig:torch.tensor, test:torch.tensor, return_map:bool = False):
    """
    Spectral Angle Mapper between two hyperspectral images.
    """
    assert orig.shape == test.shape
    if type(orig) == np.ndarray:
        orig_i = np.copy(orig)
        orig_i = torch.tensor(orig_i)
    else:
        orig_i = torch.clone(orig)
    if type(test) == np.ndarray:
        test_i = np.copy(test)
        test_i = torch.tensor(test_i)
    else:
        test_i = torch.clone(test)
    l,h,w = orig_i.shape
    orig_i = orig_i.reshape(l,-1)
    test_i = test_i.reshape(l,-1)
    map = torch.acos(torch.sum(orig_i * test_i, axis = 0) / (torch.linalg.norm(orig_i, axis = 0) * torch.linalg.norm(test_i, axis = 0) + 1e-10))
    map = map.reshape(h,w)
    if return_map:
        return torch.mean(map), map
    else:
        return torch.mean(map)