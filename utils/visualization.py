import numpy as np
import torch
import matplotlib.pyplot as plt

def draw_hpim(hpim_i: torch.tensor, lams_i:torch.tensor = None, draw:bool = False, method:str = '1931', normalize = False) -> np.ndarray:
    """
    Draw hyperspectral image in RGB format.
    hpim: hyperspectral image
    lams: wavelength of each band
    draw: whether to draw the image or just return the rgb array.
    method: '1931' or 'Gaussians'
    1931: use the CIE 1931 standard observer to convert the hyperspectral image to RGB image.
    Gaussians: use Gaussian functions to convert the hyperspectral image to RGB image.
    return rgb_img in HWC format. 
    """
    if type(hpim_i) == torch.Tensor:
        hpim = torch.clone(hpim_i)
        hpim = hpim.detach().cpu().numpy()
    else:
        hpim = np.copy(hpim_i)
    if type(lams_i) == torch.Tensor:
        lams = torch.clone(lams_i)
        lams = lams.detach().cpu().numpy()
    else:
        lams = np.copy(lams_i)
    l,h,w = np.shape(hpim)
    if lams_i is None:
        lams = np.linspace(400, 700, num = l)
    if method == '1931':
        data = np.reshape(hpim, (l, -1), order ='C')
        CIEXYZ_1931_table = np.load('data/CIEXYZ_1931_table.npy')
        CIEXYZ_1931_table[:, 1:] = CIEXYZ_1931_table[:, 1:]
        table_h = CIEXYZ_1931_table.shape[0]
        X = np.interp(lams, CIEXYZ_1931_table[:, 0], CIEXYZ_1931_table[:, 1], left = 0, right = 0)
        Y = np.interp(lams, CIEXYZ_1931_table[:, 0], CIEXYZ_1931_table[:, 2], left = 0, right = 0)
        Z = np.interp(lams, CIEXYZ_1931_table[:, 0], CIEXYZ_1931_table[:, 3], left = 0, right = 0)
        hs2xyz = np.stack([X, Y, Z], axis=1).T
        #xyz_data = hs2xyz @ data
        xyz2rgb = np.asarray([2.3706743, -0.9000405, -0.4706338, -0.5138850, 1.4253036, 0.0885814, 0.0052982, -0.0146949, 1.0093968]).reshape(3,3)
        hsi2rgb = xyz2rgb@hs2xyz / 9
        np.save('data/hsi2rgb.npy', hsi2rgb)
        rgb_data = hsi2rgb @ data
        #rgb_data = rgb_data / 9
        if normalize:
            rgb_data = (rgb_data - np.min(rgb_data, axis = 1, keepdims=True)) / (np.max(rgb_data, axis = 1, keepdims=True) - np.min(rgb_data, axis = 1, keepdims=True) + 1e-16)
        rgb_img = np.reshape(rgb_data, (3, h, w), order='C')
        rgb_img = np.moveaxis(rgb_img, 0, -1)
        rgb_img = np.clip(rgb_img, 0, 1)
        if draw:
            plt.figure()
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.show()
        return rgb_img
    elif method == 'Gaussians':
        R = .55*np.exp(-(lams-600)**2/2500)
        G = np.exp(-(lams-525)**2/4000)
        B = .85*np.exp(-(lams-450)**2/5000)
        wb = [0.972952272645604, 2.0642231002049565, 1.6066716258050628]
        R = R/wb[0]
        G = G/wb[1]
        B = B/wb[2]
        Rim = np.moveaxis(np.tile(R,[h,w,1]),2,0)
        Gim = np.moveaxis(np.tile(G,[h,w,1]),2,0)
        Bim = np.moveaxis(np.tile(B,[h,w,1]),2,0)
        rgb_img = np.stack([np.sum(Rim*hpim,axis=0),np.sum(Gim*hpim,axis=0),np.sum(Bim*hpim,axis=0)],axis=-1)
        rgb_img = np.clip((1/3) * rgb_img, 0,1)
        if draw:
            plt.figure()
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.show()
        return rgb_img