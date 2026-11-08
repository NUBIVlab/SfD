import torch
import numpy as np
from hsdt import hsdt, hsdt_deep, hsdt_4 # pyright: ignore[reportMissingImports]
import utils.spectral_basis as dim

def soft_thresh(x, tau):
    '''
    Soft regularization to apply L1 regularization
    '''
    return torch.sign(x) * torch.maximum(torch.abs(x) - tau, torch.tensor(0.0, dtype=torch.float32))

class no_reg:
    '''
    No regularization applied.
    '''
    def __init__(self) -> None:
        pass

    def reg(self, x_spec):
        return x_spec

class L1_reg: #TODO: Update
    '''
    L1 regularization applied in spatial domain.
    # TODO: Requires update for spectral basis.
    '''
    def __init__(self, lam:float, fsh:int, fsw:int, h:int, w:int) -> None:
        '''
        lam: regularization weight
        fsh: focal stack height in fft domain.
        fsw: focal stack width in fft domain.
        '''
        self.lam = lam
        self.fsh = fsh
        self.fsw = fsw
        self.h = h
        self.w = w

    def reg(self, x_spec):
        '''
        x_spec: reconstruction in fft domain.
        returns the regularized reconstruction in fft domain.
        '''
        x = torch.real(torch.fft.ifft2(x_spec, s=(self.fsh, self.fsw)))[:,:self.h,:self.w]
        x = soft_thresh(x, self.lam)
        return torch.fft.fft2(x, s=(self.fsh, self.fsw))

class TV_reg: #TODO: Update
    '''
    TV regularization applied in spatial domain.
    # TODO: Requires update for spectral basis.
    '''
    def __init__(self, lam:float, fsh:int, fsw:int) -> None:
        '''
        lam: regularization weight
        fsh: focal stack height in fft domain.
        fsw: focal stack width in fft domain.
        '''
        self.lam = lam
        self.fsh = fsh
        self.fsw = fsw

    def reg(self, x_spec):
        '''
        x_spec: reconstruction in fft domain.
        returns the regularized reconstruction in fft domain.
        '''
        x = torch.real(torch.fft.ifft2(x_spec, s=(self.fsh, self.fsw)))
        grad = torch.stack((torch.roll(x, 1, dims=1) - x, torch.roll(x, 1, dims=2) - x))
        grad = grad.permute(1,2,3,0)
        grad = soft_thresh(grad, self.lam)
        
        lines0 = -torch.cumsum(grad[:, -1, :, 1], dim=1)
        lines0 += torch.tile(x[:, -1, -1].reshape(x.shape[0], 1), (1, x.shape[2]))
        recon0 = -torch.cumsum(grad[:, :, :, 0], dim=1)
        recon0 += torch.tile(lines0.reshape(x.shape[0], 1, x.shape[2]), (1,x.shape[1],1))

        lines1 = -torch.cumsum(grad[:, :, -1, 0], dim=1)
        lines1 += torch.tile(x[:, -1, -1].reshape(x.shape[0], 1), (1, x.shape[1]))
        recon1 = -torch.cumsum(grad[:, :, :, 1], dim=2)
        recon1 += torch.tile(lines1.reshape(x.shape[0], x.shape[1], 1), (1,1,x.shape[2]))
        recon = (recon0 + recon1) / 2
        return torch.fft.fft2(recon, s=(self.fsh, self.fsw))
    
class HSDT:
    '''
    Deep denoiser by Zeqiang Lai, Chenggang Yan, Ying Fu. More info at: 
    https://github.com/Zeqiang-Lai/HSDT
    '''
    def __init__(self, settings:dict, basis:dim.SpectralDim = None, config:dict=None) -> None:
        '''
        size: 's' for small model, 'm' for medium model, 'l' for large model.
        noise_type: 'gaussian' or 'octopus'
        scale: scaling based on the results of: https://arxiv.org/pdf/2002.11546.pdf
        '''
        size = settings['size']
        noise_type = settings['noise']
        scale = settings['scale']
        if size == 's':
            self.net = hsdt_4()
        elif size == 'm':
            self.net = hsdt()
        elif size == 'l':
            self.net = hsdt_deep()
        model_dir = 'data/model_weights/hsdt_' + noise_type + '_' + size + '.pth'
        self.net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu'), weights_only=False))
        self.net.to(config['device'])   
        self.net.eval()
        self.scale = scale
        self.basis = basis

    def reg(self, x):
        '''
        x_spec: reconstruction in fft domain.
        returns the denoised reconstruction in fft domain.
        '''
        #x = torch.real(torch.fft.ifft2(x_spec, s=(self.fsh, self.fsw)))[:,:self.h,:self.w]
        l, h, w = x.shape
        if self.basis is not None:
            x = self.basis.recover(x)
        # If number of channels is less than 31, put zero channels at the beggigng to make it 31.
        original_shape = x.shape[0]
        if x.shape[0] < 31:
            x = torch.cat([torch.zeros(31 - x.shape[0], x.shape[1], x.shape[2]).to(x.device), x], dim=0)
        # We need to padd the image to a multiple of 4. 
        # We will crop the image after denoising.
        pad_h = 8 - h % 8
        pad_w = 8 - w % 8
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = self.net(x[None,None,:,:,:] / self.scale) * self.scale
        x = x[0,0,:,:,:]
        x = x[:, :h, :w]
        if original_shape < 31:
            x = x[31 - original_shape:, :, :]
        if self.basis is not None:
            x = self.basis.project(x, l)
        return x



