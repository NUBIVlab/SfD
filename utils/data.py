import os, glob

import numpy as np
import scipy as sp
import torch, torchvision
import matplotlib.pyplot as plt

data_path = '/home/mkeremaydin/Desktop/projects/data' # Replace with your data path.

def RGYPepperC(device:torch.device = torch.device('cpu')) -> tuple[torch.tensor, torch.tensor]:
    """
    hyperspectral data downloaded from:
    https://sites.google.com/site/hyperspectralcolorimaging/dataset/general-scenes
    returns format: CxHxW
    """
    f = sp.io.loadmat(os.path.join(data_path,'Pepper/RGYPepperC_2500.mat')) 
    lams = torch.tensor(torch.squeeze(f['Wavelengths']), device=device, dtype=torch.float32)
    hpim = torch.tensor(f['S'], device=device, dtype=torch.float32) 
    hpim = torch.moveaxis(hpim,1,2)
    return (lams, hpim)

def bear_and_fruit(size:int = 408, device:torch.device = torch.device('cpu'))-> tuple[torch.tensor, torch.tensor]:
    """
    hyperspectral data downloaded from:
    https://color2.psych.upenn.edu/hyperspectral/bearfruitgray/bearfruitgray.html
    returns format: CxHxW
    """
    rsz = size / 2010
    f = sp.io.loadmat(os.path.join(data_path,'BearFruitGrayB/data.mat'))
    lams = torch.squeeze(torch.tensor(f['wavelengths'].astype(np.float32), device=device, dtype=torch.float32))
    hpim0 = torch.tensor(f['img_stack'], device=device, dtype=torch.float32)
    hpim0 = hpim0[:, 5:-5, 5:-5]
    l,h,w = hpim0.shape
    hpim = torch.zeros((l,round(rsz*h), round(rsz*w)))
    for i in range(len(lams)):
        hpim[i,:,:] = torchvision.transforms.functional.resize(torch.unsqueeze(hpim0[i,:,:],0), (round(rsz*h), round(rsz*w)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
    return lams, hpim

def load_Harvard(file_to_open:int = 0, size:int = 408, device:torch.device = torch.device('cpu'), correct_camera_sensitivity:bool = False)-> tuple[torch.tensor, torch.tensor]:
    """
    hyperspectral data downloaded from:
    http://vision.seas.harvard.edu/hyperspec/d2x5g3/
    returns format: CxHxW
    """
    filepath = os.path.join(data_path, 'HarvardDataset/')
    rsz = size / 1040
    lams = torch.arange(420,721,10, device=device, dtype=torch.float32)
    file = sorted(glob.glob(filepath+'*.mat'))[file_to_open]
    data = sp.io.loadmat(file)
    image = torch.tensor(data['ref'].astype(np.float32), device=device, dtype=torch.float32)
    image = torch.permute(image, (2,0,1))
    image = image[:, :, (1392-1040):1392]
    image = image / torch.reshape(torch.amax(image, axis=(1,2)), (image.shape[0],1,1))
    l,h,w = image.shape
    hpim = torch.zeros((l,round(rsz*h), round(rsz*w)), device=device, dtype=torch.float32)
    for i in range(len(lams)):
        hpim[i,:,:] = torchvision.transforms.functional.resize(torch.unsqueeze(image[i,:,:],0), (round(rsz*h), round(rsz*w)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
    if correct_camera_sensitivity:
        cam_sens = np.genfromtxt(filepath+'calib.txt')
        hpim = hpim / torch.reshape(cam_sens, (cam_sens.shape[0], 1, 1))
    return lams, hpim

def load_KAIST(file_to_open:int = 0, size:int = 408, device:torch.device = torch.device('cpu')) -> tuple[torch.tensor, torch.tensor]:
    """
    hyperspectral data downloaded from:
    http://vclab.kaist.ac.kr/siggraphasia2017p1/kaistdataset.html
    You need to convert the original exr data to mat format. 
    returns format: CxHxW
    """
    rsz = size/2704
    lams = torch.arange(420,721,10)
    filepath = os.path.join(data_path, 'KAISTDataset/')
    file = filepath+str(file_to_open)+'.mat'
    data = sp.io.loadmat(file)
    hpim0 = torch.tensor(data['hpim'], device=device, dtype=torch.float32)
    l,h,w = hpim0.shape
    hpim = torch.zeros((l,round(rsz*h), round(rsz*w)), device=device, dtype=torch.float32)
    for i in range(len(lams)):
        hpim[i,:,:] = torchvision.transforms.functional.resize(torch.unsqueeze(hpim0[i,:,:],0), (round(rsz*h), round(rsz*w)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
    hpim = torch.clip(hpim, 0, 1)
    center_width = hpim.shape[1] //2
    hpim = hpim[:, :, center_width-size//2:center_width-size//2+size]
    return lams, hpim

def load_CAVE(file_to_open: int, size:float = 408, device:torch.device = torch.device('cpu')) -> tuple[torch.tensor, torch.tensor]:
    """
    hyperspectral data downloaded from:
    https://www.cs.columbia.edu/CAVE/databases/multispectral/
    returns format: CxHxW
    """
    rsz = size/512
    l,h,w = 31, 512, 512
    filepath = os.path.join(data_path, 'CAVEDataset/')
    folder = sorted(glob.glob(filepath+'*'))[file_to_open]
    dataname = folder.split('/')[1]
    img = torch.zeros((l,round(rsz*h), round(rsz*w)), device=device, dtype=torch.float32)
    lams = torch.arange(400, 705, 10, device=device, dtype=torch.float32)
    for j,file in enumerate(sorted(glob.glob(folder+'/*.png'))):
            img[j,:,:] = torchvision.transforms.functional.resize(torch.unsqueeze(torch.tensor(plt.imread(file)),0), (round(rsz*h), round(rsz*w)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
    return lams, img

def load_ICVL(file_to_open:int, size = 408, spectral_downsample = 1.0, device:torch.device = torch.device('cpu')) -> tuple[torch.tensor, torch.tensor]:
    """
    hyperspectral data downloaded from:
    https://icvl.cs.bgu.ac.il/hyperspectral/
    returns format: CxHxW
    """
    rsz = size / 1306
    data_path = '/data/'
    converter = lambda s: float(s.decode('utf8').strip(','))
    filepath = os.path.join(data_path, 'ICVLDataset/small_brain/')
    folder = sorted(glob.glob(filepath+'*'))[file_to_open]
    dataname = folder.split('/')[-1]
    l, h, w = None, None, None
    lambda_start = None
    lambda_end = None
    with open(filepath+dataname+'/'+dataname+'.hdr', 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'samples' in line:
                parts = line.split()
                w = int(parts[-1])
            if 'lines' in line:
                parts = line.split()
                h = int(parts[-1])
            if line.startswith('bands'):
                parts = line.split()
                l = int(parts[-1])
            if 'Wavelength' in line:
                lambda_start = int(i)+1
            if '}' in line and lambda_start is not None and i > lambda_start:
                lambda_end = int(i)-1
                break
    lams = np.loadtxt(filepath+dataname+'/'+dataname+'.hdr', dtype=np.float32, skiprows=lambda_start, converters = converter, max_rows=lambda_end-lambda_start+1)
    img = np.fromfile(filepath+dataname+'/'+dataname+'.raw', dtype=np.uint16, sep="")
    img = np.reshape(img, (h, l, w))
    img = np.flip(np.transpose(img, (1,2,0)), axis = 1)
    b,s = max(h,w), min(h,w)
    a = (b - s) //2
    if h > w:
        img = img[a:(a+s),:,:]
    else:
        img = img[:, a:(a+s),:]
    l,h,w = img.shape
    hpim = torch.zeros((519, round(rsz*h), round(rsz*w)), device=device, dtype=torch.float32)
    for i in range(img.shape[0]):
        hpim[i,:,:] = torchvision.transforms.functional.resize(torch.unsqueeze(torch.tensor(np.copy(img[i,:,:]).astype(np.float32)),0), (round(rsz*h), round(rsz*w)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
    hpim = hpim / np.power(2.0, 15)
    if spectral_downsample != 1.0:
        hpim = hpim.detach().cpu().numpy()
        c,h,w = hpim.shape
        ds_ratio = int(1/spectral_downsample)
        new_c = np.floor(c/ds_ratio).astype(np.int32)
        old_c = new_c * ds_ratio
        hpim = hpim[:old_c,:,:]
        lams = lams[:old_c]
        hpim = np.moveaxis(hpim, 0, -1)
        hpim = np.reshape(hpim, (h,w,-1,ds_ratio))
        hpim = np.sum(hpim, axis=-1)
        hpim = np.moveaxis(hpim, -1, 0)
        lams = np.reshape(lams, (-1,ds_ratio))
        lams = np.mean(lams, axis=-1)
        hpim = torch.tensor(hpim, device=device, dtype=torch.float32)
    return torch.tensor(lams, device=device, dtype=torch.float32), hpim


