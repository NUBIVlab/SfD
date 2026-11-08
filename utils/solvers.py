import math
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils.ip as inv
import utils.visualization as vis

class pnp_ADMM:
    '''
    pnp admm without the crop operator. see pnp_ADMM_2 for the variant with crop operator. This verison is cheaper to compute without much performance loss for smaller PSF's.  
    '''
    def __init__(self, inv_problem:inv.inverse_problem, regularizer:object, config:dict):
        self.config = config
        self.device = self.config['device']
        self.inv_problem = inv_problem
        self.reg = regularizer
        self.rho = float(self.config['rho'])
        assert self.rho >= 0.0

        M = self.inv_problem.filter_stack_spectral.to(self.device)
        A_hw = torch.moveaxis(M, [0,1,2,3], [2,3,0,1]).contiguous()
        self.H, self.W, self.Udim, self.Vdim = A_hw.shape
        self.B = self.H * self.W
        self.A_b = A_hw.reshape(self.B, self.Udim, self.Vdim)
        self.I_VB = torch.eye(self.Vdim, dtype=self.A_b.dtype, device=self.device)[None].expand(self.B, -1, -1)
        self._precompute_augmented_pinv(self.rho)
        self.X = self.init_X()
        self.U = torch.zeros_like(self.X, device=self.device)
        self.Z = self.Z_update(self.X, self.U)
        self.recon = None
        return
    
    @torch.no_grad()
    def init_X(self, empty_cache: bool = True):
        b_fft = self.inv_problem.focal_stack_spectral.to(self.device)
        b_b   = b_fft.permute(1, 2, 0).reshape(self.B, self.Udim)
        eps   = torch.finfo(self.A_b.real.dtype).eps
        rcond = max(self.Udim, self.Vdim) * eps
        F_A   = torch.linalg.pinv(self.A_b, rcond=rcond)
        x_b = (F_A @ b_b.unsqueeze(-1)).squeeze(-1)
        X0  = x_b.reshape(self.H, self.W, self.Vdim).permute(2, 0, 1).contiguous()
        del F_A, b_b, b_fft, x_b
        if empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return X0
    
    def _precompute_augmented_pinv(self, rho: float):
        sr = 0.0 if rho == 0.0 else math.sqrt(rho)
        A_aug = torch.cat([self.A_b, sr * self.I_VB], dim=1)
        eps = torch.finfo(self.A_b.real.dtype).eps
        rcond = max(self.Udim + self.Vdim, self.Vdim) * eps
        self.F_aug = torch.linalg.pinv(A_aug, rcond=rcond)
        self._rho_cached = rho

    def set_rho(self, rho: float):
        '''
        Function to change rho during run-time. Slowly decaying rho over iterations leads to better results, but runs slower. 
        Not used in the paper, not sure if it works. 
        '''
        if rho != self._rho_cached:
            self.rho = float(rho)
            self._precompute_augmented_pinv(self.rho)
    
    def _recover_from_spectral(self, x_vhw: torch.Tensor):
        recon = torch.real(torch.fft.ifft2(x_vhw, s=(self.inv_problem.fsh, self.inv_problem.fsw)))[:, :self.inv_problem.h, :self.inv_problem.w]
        return self.inv_problem.basis_object.recover(recon)

    @torch.no_grad()
    def X_update(self, Z, U):
        b_fft = self.inv_problem.focal_stack_spectral.to(self.device)            
        b_b   = b_fft.permute(1,2,0).reshape(self.B, self.Udim)                  
        c_b   = (Z - U).permute(1,2,0).reshape(self.B, self.Vdim)                

        sr = 0.0 if self.rho == 0.0 else math.sqrt(self.rho)
        y_aug = torch.cat([b_b, sr * c_b], dim=1).unsqueeze(-1) 

        x_b = (self.F_aug @ y_aug).squeeze(-1) 
        x_vhw = x_b.reshape(self.H, self.W, self.Vdim).permute(2,0,1).contiguous()
        return x_vhw
    
    def Z_update(self, x, u):
        x_spatial = torch.real(torch.fft.ifft2(x))[:, :self.inv_problem.h, :self.inv_problem.w]
        u_spatial = torch.real(torch.fft.ifft2(u))[:, :self.inv_problem.h, :self.inv_problem.w]
        z_spatial = self.reg.reg(x_spatial + u_spatial)
        z_spectral = torch.fft.fft2(z_spatial, s=(self.inv_problem.fsh, self.inv_problem.fsw))
        return z_spectral
    
    def U_update(self, x, z, u):
        return u + x - z
    
    def step(self):
        self.X = self.X_update(self.Z, self.U)
        self.Z = self.Z_update(self.X, self.U)
        self.U = self.U_update(self.X, self.Z, self.U)
        self.recon_spectral = self.Z
        return

class pnp_ADMM_2:
    """
    pnp-admm with crop operator. 
    """
    def __init__(self, inv_problem:inv.inverse_problem, regularizer:object, config:dict):
        self.config = config
        self.device = self.config['device']
        self.inv_problem = inv_problem
        self.reg = regularizer
        self.admm_mu = self.config['rho']
        self.R_divmat_method = self.config['divmat_method']
        self.recon = None
        self.num_meas, self.foc_h, self.foc_w = self.inv_problem.focal_stack.shape
        self.num_meas, self.num_dim, self.filt_h, self.filt_w = self.inv_problem.filter_stack.shape
        #self.fft_h, self.fft_w = self.foc_h + self.filt_h - 1, self.foc_w + self.filt_w - 1
        self.fft_h, self.fft_w = self.foc_h, self.foc_w
        assert self.fft_h == self.inv_problem.filter_stack_spectral.shape[2]
        assert self.fft_w == self.inv_problem.filter_stack_spectral.shape[3]
        self.X_divmat = self.precompute_X_divmat()
        self.V_divmat = self.precompute_V_divmat()
        self.x, self.u, self.v, self.xi, self.eta = self.init_Matrices()
        return
    
    def display_variables(self, meas_lams:torch.Tensor, adjust_brightness:bool = False):
        v_vis = vis.draw_hpim(self.v, meas_lams, False)
        if adjust_brightness:
            v_vis = v_vis / np.percentile(v_vis, 99)
        x_vis = self.inv_problem.basis_object.recover(self.x)
        x_vis = vis.draw_hpim(x_vis, self.inv_problem.lams, False)
        if adjust_brightness:
            x_vis = x_vis / np.percentile(x_vis, 99)
        u_vis = self.inv_problem.basis_object.recover(self.u)
        u_vis = vis.draw_hpim(u_vis, self.inv_problem.lams, False)
        if adjust_brightness:
            u_vis = u_vis / np.percentile(u_vis, 99)
        xi_vis = vis.draw_hpim(self.xi, meas_lams, False)
        if adjust_brightness:
            xi_vis = xi_vis / np.percentile(xi_vis, 99)
        eta_vis = self.inv_problem.basis_object.recover(self.eta)
        eta_vis = vis.draw_hpim(eta_vis, self.inv_problem.lams, False)
        if adjust_brightness:
            eta_vis = eta_vis / np.percentile(eta_vis, 99)
        plt.figure(figsize=(25,4))
        plt.subplot(151)
        plt.imshow(v_vis)
        plt.title('v')
        plt.subplot(152)
        plt.imshow(x_vis)
        plt.title('x')
        plt.subplot(153)
        plt.imshow(u_vis)
        plt.title('u')
        plt.subplot(154)
        plt.imshow(xi_vis)
        plt.title('xi')
        plt.subplot(155)
        plt.imshow(eta_vis)
        plt.title('eta')
        plt.tight_layout()
        plt.show()
        return


    def step(self):
        self.v = self.v_update(self.x, self.xi)
        self.x = self.x_update(self.u, self.v, self.xi, self.eta)
        self.u = self.u_update(self.x, self.eta)
        self.xi = self.xi_update(self.x, self.v, self.xi)
        self.eta = self.eta_update(self.x, self.u, self.eta)
        self.recon = self.x
        #self.display_variables(ENTER MEASUREMENT WAVELENGTHS HERE, True)
        return
    
    def _C(self, x):
        # This is the crop operator. It crops the image to the size of the focal stack.
        # Cropping is done to the center of the image.
        top = (self.fft_h - self.foc_h + 1) // 2
        left = (self.fft_w - self.foc_w + 1) // 2
        bottom = top + self.foc_h
        right = left + self.foc_w
        return x[:, top:bottom, left:right]
    
    def _CH(self, x):
        p_top = (self.fft_h - self.foc_h + 1) // 2
        p_left = (self.fft_w - self.foc_w + 1) // 2
        p_bottom = self.fft_h - self.foc_h - p_top
        p_right = self.fft_w - self.foc_w - p_left
        return torch.nn.functional.pad(x, (p_left, p_right, p_top, p_bottom))
    
    def apply_V_divmat(self, x):
        # Multiply the inner part of image (uncropped) with 1/(1+mu).
        # Multiply the outer part of image (cropped) with 1 / mu.
        top = (self.fft_h - self.foc_h + 1) // 2
        left = (self.fft_w - self.foc_w + 1) // 2
        bottom = top + self.foc_h
        right = left + self.foc_w
        x = x * 1 / self.admm_mu[0]
        x[:, top:bottom, left:right] = x[:, top:bottom, left:right] * self.admm_mu[0] / (1 + self.admm_mu[0])
        return x


    def fft_convolve_einsum(self, a, b, b_type:str='spec', crop:bool=False, return_spectral:bool=False):
        u,v,fh,fw = a.shape
        v2, ih, iw = b.shape
        assert return_spectral == False or crop == False # Cannot return spectral and crop at the same time.
        try:
            assert v == v2
        except:
            raise ValueError('Dimension mismatch along channels in fft_convolve_einsum. Channel Sizes are: ', v, v2)
        if b_type == 'spec':
            try:
                assert ih == self.fft_h and iw == self.fft_w
            except:
                raise ValueError('Invalid shape for b, Error thrown in fft_convolve_einsum. Shape is: ', ih, iw)
        else:
            if ih == self.fft_h and iw == self.fft_w:
                b = torch.fft.fft2(b)
            elif ih == self.foc_h and iw == self.foc_w:
                #b = self._CH(b)
                b = torch.fft.fft2(b, s=(self.fft_h, self.fft_w))
            else:
                raise ValueError('Invalid shape for b, Error thrown in fft_convolve_einsum. Shape is: ', ih, iw)
        try:
            c = torch.einsum('uvhw,vhw->uhw', a, b)
        except:
            print(a.shape, b.shape)
        if return_spectral:
            return c
        c = torch.real(torch.fft.ifft2(c))
        if crop:
            #import aux_funcs.visualization as vis
            #import matplotlib.pyplot as plt
            #c_rec = self.inv_problem.basis_object.recover(c)
            #c_rgb = vis.draw_hpim(c_rec, self.inv_problem.lams, False)
            #plt.figure()
            #plt.imshow(c_rgb / np.percentile(c_rgb, 99))
            #plt.show()
            c = self._C(c)
        return c
    
    def precompute_V_divmat(self):
        return 1./(self._CH(torch.ones((1, self.foc_h, self.foc_w))) + self.admm_mu[0])

    def apply_V_divmat(self, x):
        # Multiply the inner part of image (uncropped) with 1/(1+mu).
        # Multiply the outer part of image (cropped) with 1 / mu.
        return x * self.V_divmat
    
    def precompute_X_divmat(self):
        M = self.inv_problem.filter_stack_spectral
        M = torch.moveaxis(M, [0, 1, 2, 3], [2, 3, 0, 1])
        MTM = torch.matmul(torch.transpose(M.conj(), 2, 3), M)
        reg = (torch.zeros(self.inv_problem.v, self.inv_problem.v, device=self.device).fill_diagonal_(1)).reshape(1, 1, self.inv_problem.v, self.inv_problem.v).to(self.device)
        pre_inv = self.admm_mu[0]*(MTM).type(torch.complex128) + self.admm_mu[1] * reg
        if self.R_divmat_method == 'pinv':
            MTM_inv = torch.linalg.pinv(pre_inv)
        elif self.R_divmat_method == 'inv':
            MTM_inv = torch.linalg.inv(pre_inv)
        elif self.R_divmat_method == 'svd':
            MTM_inv = self.safe_invert(pre_inv, rtol=1e-15)
        MTM_inv = MTM_inv.type(torch.complex64)
        MTM_inv = torch.moveaxis(MTM_inv, [0, 1, 2, 3], [2, 3, 0, 1])
        return MTM_inv
    
    def r_calc(self, u, v, xi, eta):
        return self.fft_convolve_einsum(torch.transpose(self.inv_problem.filter_stack_spectral.conj(), 0, 1), (self.admm_mu[0] * v + xi), 'spat', False, True) + torch.fft.fft2(eta + self.admm_mu[1] * u, s=(self.fft_h, self.fft_w))


    def u_update(self, x, eta):
        return self.reg.reg(x + eta)
    
    def x_update(self, u, v, xi, eta):
        return self.fft_convolve_einsum(self.X_divmat, self.r_calc(u, v, xi, eta), 'spec', False)[:, :self.foc_h, :self.foc_w]
    
    def v_update(self, x, xi):
        return self.apply_V_divmat(self._CH(self.inv_problem.focal_stack) + self.admm_mu[0] * self.fft_convolve_einsum(self.inv_problem.filter_stack_spectral, x, 'spat', False) - xi)
    
    def xi_update(self, x, v, xi):
        return xi + self.admm_mu[0] * (v - self.fft_convolve_einsum(self.inv_problem.filter_stack_spectral, x, 'spat', False))
    
    def eta_update(self, x, u, eta):
        return eta + self.admm_mu[1] * (u - x)
    
    def init_Matrices(self):
        eta = torch.zeros((self.num_dim, self.foc_h, self.foc_w), device=self.device)
        xi = torch.zeros((self.num_meas, self.fft_h, self.fft_w), device=self.device)
        v = self.apply_V_divmat(self._CH(self.inv_problem.focal_stack))
        u = torch.zeros_like(eta)
        x = self.x_update(u, v, xi, eta)
        u = self.u_update(x, eta)
        return x, u, v, xi, eta
    


class inverse_filter:
    '''
    Solves the problem with inverse filtering in fourier domain.
    '''
    def __init__(self, inv_problem:inv.inverse_problem, lam:float = 0, config:dict = None) -> None:
        '''
        inv_problem: Inverse problem object, lets the pnpFISTA access variables and functions like _f, _H inside the inverse problem.
        lam: Regularization term to prevent division by zeros in the pressence of noise.
        '''
        self.device = config['device']
        self.inv_problem = inv_problem
        self.lam = lam
        self.recon_spectral = None 
        self.recon = None
        M = self.inv_problem.filter_stack_spectral
        M = torch.moveaxis(M, [0, 1, 2, 3], [2, 3, 0, 1]).contiguous()
        reg = (torch.zeros(self.inv_problem.u, self.inv_problem.v, device=self.device).fill_diagonal_(self.lam)).reshape(1, 1, self.inv_problem.u, self.inv_problem.v).to(self.device)
        inverse_filter = torch.linalg.pinv(M + reg).to(self.device)
        self.inverse_filter = torch.moveaxis(inverse_filter, [0, 1, 2, 3], [2, 3, 0, 1])
    
    def solve(self):
        rec_spectral = torch.einsum('uvhw,vhw->uhw', self.inverse_filter, self.inv_problem.focal_stack_spectral)
        recon = torch.real(torch.fft.ifft2(rec_spectral, s=(self.inv_problem.fsh, self.inv_problem.fsw)))
        self.recon = self.inv_problem.basis_object.recover(recon)


class inverse_filter_2:
    '''
    Computationally more expensive but more flexible. Can be used inside ADMM.
    '''
    def __init__(self, inv_problem, lam: float = 1e-6, config: dict | None = None,
                 rtol: Optional[float] = None, atol: float = 0.0):
        self.device = config['device']
        self.inv_problem = inv_problem
        self.lam = float(lam)
        self.rho = float(lam)
        self.rtol = rtol
        self.atol = float(atol)
        M = inv_problem.filter_stack_spectral.to(self.device)
        M = torch.moveaxis(M, [0, 1, 2, 3], [2, 3, 0, 1]).contiguous()
        self.H, self.W, self.U, self.V = M.shape
        self.B = self.H * self.W
        self.M_hwuv = M
        self.M_b = M.reshape(self.B, self.U, self.V)
        self.I_uv = torch.eye(self.U, self.V, dtype=M.dtype, device=self.device)
        self.I_vB  = torch.eye(self.V, dtype=M.dtype, device=self.device)[None].expand(self.B, -1, -1)
        if self.rtol is None:
            if self.atol > 0.0:
                self.rcond_default = 0.0
            else:
                eps = torch.finfo(self.M_b.real.dtype).eps
                self.rcond_default = max(self.U, self.V) * eps
        else:
            self.rcond_default = float(self.rtol)
        M_reg = self.M_hwuv + self.lam * self.I_uv
        F_hwvu = torch.linalg.pinv(M_reg, rcond=self.rcond_default)
        self.F_vuhw = F_hwvu.permute(2, 3, 0, 1).contiguous()


    def _b_fft(self):
        return self.inv_problem.focal_stack_spectral.to(self.device)

    def _recover_from_spectral(self, x_vhw: torch.Tensor):
        recon = torch.real(torch.fft.ifft2(x_vhw, s=(self.inv_problem.fsh, self.inv_problem.fsw)))
        return self.inv_problem.basis_object.recover(recon)


    @torch.no_grad()
    def solve(self, c_vhw: Optional[torch.Tensor] = None):
        A_b = self.M_b                                                 
        b_b = self._b_fft().permute(1, 2, 0).reshape(self.B, self.U)
        if c_vhw is None:
            c_b = torch.zeros(self.B, self.V, dtype=A_b.dtype, device=A_b.device)
        else:
            c_b = c_vhw.permute(1, 2, 0).reshape(self.B, self.V)
        sr = math.sqrt(self.rho)
        A_aug = torch.cat([A_b, sr * self.I_vB], dim=1)
        y_aug = torch.cat([b_b, sr * c_b], dim=1).unsqueeze(-1)
        if (self.atol == 0.0) and (self.rtol is None):
            eps = torch.finfo(A_b.real.dtype).eps
            rcond = max(self.U + self.V, self.V) * eps
            F_aug = torch.linalg.pinv(A_aug, rcond=rcond) 
        else:
            rtol = 0.0 if (self.rtol is None) else float(self.rtol)
            F_aug = torch.linalg.pinv(A_aug, atol=self.atol, rtol=rtol)
        x_b = (F_aug @ y_aug).squeeze(-1) 
        x_vhw = x_b.reshape(self.H, self.W, self.V).permute(2, 0, 1).contiguous()
        self.recon = self._recover_from_spectral(x_vhw)
        return self.recon, x_vhw
