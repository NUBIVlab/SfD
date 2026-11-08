import gc
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import utils.visualization as vis
import utils.spectral_basis as dim
from utils.optics import draw_filter_stack

class inverse_problem:
    '''
    Stores the forward model and measurements. If necessary, computes the forward model on the fly.
    Passes the necessary variables to the solver. 
    '''
    def __init__(self, x:torch.tensor=None, meas:torch.tensor=None, filter_stack:torch.tensor=None, config:dict=None) -> None:
        '''
        x: Ground truth image, C x H x W
        meas: Measurement image, C_sim x H_sim x W_sim
        filter_stack: Filter stack, U x V x H_f x W_f
        '''
        self.config = config
        self.simulated = config['simulate_measurements']
        self.device = config['device']
        if self.simulated:
            assert x is not None and meas is None and filter_stack is not None
            self.u, self.v, self.fh, self.fw = filter_stack.shape
            self.c, self.h, self.w = x.shape
            self.fsh, self.fsw  = self.h + self.fh - 1, self.w + self.fw - 1
            self.x = x
            self.x_spectral =  torch.fft.fft2(x, s=(self.fsh, self.fsw))
            self.filter_stack = filter_stack
            self.filter_stack_spectral = torch.fft.fft2(filter_stack, s=(self.fsh, self.fsw))
            self.focal_stack = self._f(self.x, is_spatial=True)
        else:
            assert meas is not None and filter_stack is not None and x is None
            self.u, self.fsh, self.fsw = meas.shape
            _, self.v, self.fh, self.fw = filter_stack.shape
            self.h, self.w = self.fsh - self.fh + 1, self.fsw - self.fw + 1
            self.x = x
            self.focal_stack = meas
            self.filter_stack = filter_stack
            self.filter_stack_spectral = torch.fft.fft2(filter_stack, s=(self.fsh, self.fsw))
        self.focal_stack_spectral = torch.fft.fft2(self.focal_stack, s=(self.fsh, self.fsw))
        self.lams = torch.arange(self.config['min_wavelength'], self.config['max_wavelength'] + 1, self.config['wavelength_resolution'], device=self.config['device'])
        self.recon_spectral = None
        self.recon = None
        self.basis = None
        # Below we store the original filter stack and the spectral filter stack for future use on cpu. 
        # If we need to restore them we can copy them back to the gpu.
        self.u_arch, self.v_arch = self.u, self.v
        self.filter_stack_arch = self.filter_stack.detach().cpu()
        self.filter_stack_spectral_arch = self.filter_stack_spectral.detach().cpu()
        self.focal_stack_spectral_arch = self.focal_stack_spectral.detach().cpu()
        self.focal_stack_arch = self.focal_stack.detach().cpu()
        self.lams_arch = self.lams.detach().cpu()

    def cr(self, x:torch.tensor, inp:str = 'spatial') -> torch.tensor: #TODO: UPDATE
        if inp == 'spectral':
            x = torch.abs(torch.fft.ifft2(x, s=(self.fsh, self.fsw)))
        top = (self.fsh - self.h + 1)  // 2
        left = (self.fsw - self.w + 1) // 2
        x_spatial_c = x[:, top:top+self.h, left:left+self.w]
        return x_spatial_c

    def cr_H(self, x:torch.tensor, inp:str = 'spatial') -> torch.tensor: #TODO: UPDATE
        if inp == 'spectral':
            x = torch.abs(torch.fft.ifft2(x, s=(self.fsh_ip, self.fsw_ip))) # Not sure abt this line. I don't think we need it anyway.
        pad_top = (self.fsh - self.h + 1) // 2
        pad_left = (self.fsw - self.w + 1) // 2
        pad_bottom = self.fsh - self.h - pad_top
        pad_right = self.fsw - self.w - pad_left
        x_padded = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return x_padded
    
    def _f(self, x:torch.tensor = None, is_spatial:bool = False) -> torch.tensor:
        '''
        Forward operator (A) for the inverse problem. 
        x: input image.
        '''
        if is_spatial:
            x_spectral = torch.fft.fft2(x, s=(self.fsh, self.fsw))
        else:
            x_spectral = x
        y_spectral = torch.einsum('uvhw,vhw->uhw', self.filter_stack_spectral, x_spectral)
        y_spatial = torch.real(torch.fft.ifft2(y_spectral, s=(self.fsh, self.fsw)))
        return y_spatial
    
    def _H(self, x_spectral:torch.tensor = None) -> torch.tensor:
        '''
        Tranpose (or hermitian) of the forward operator (= A^H)
        x: input image in spectral domain
        '''
        y_spectral = torch.einsum('uvhw,uhw->vhw', torch.conj(self.filter_stack_spectral), x_spectral)
        return y_spectral

    def fidelityLoss(self, x_hat:torch.tensor) -> float:
        '''
        Computes loss in frequency domain.
        x_hat: input image in spectral domain
        '''
        y_hat = self._f(x_hat)
        return torch.linalg.norm(y_hat - self.focal_stack)
    
    def reset_filter_stack(self, filter_stack = None):
        '''
        Resets the filter stack to the original state.
        '''
        if filter_stack is None:
            del self.filter_stack, self.filter_stack_spectral
            del self.focal_stack_spectral, self.focal_stack
            del self.lams
            gc.collect()
            torch.cuda.empty_cache()
            self.u, self.v = self.u_arch, self.v_arch
            self.filter_stack = self.filter_stack_arch.to(self.x.device)
            self.filter_stack_spectral = self.filter_stack_spectral_arch.to(self.x.device)
            self.focal_stack_spectral = self.focal_stack_spectral_arch.to(self.x.device)
            self.focal_stack = self.focal_stack_arch.to(self.x.device)
            self.lams = self.lams_arch.to(self.x.device)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            del self.filter_stack, self.filter_stack_spectral
            del self.focal_stack_spectral, self.focal_stack
            del self.lams
            gc.collect()
            torch.cuda.empty_cache()
            self.u, self.v = filter_stack.shape[0], filter_stack.shape[1]
            self.filter_stack = filter_stack.to(self.x.device)
            self.filter_stack_spectral = torch.fft.fft2(filter_stack, s=(self.fsh_ip, self.fsw_ip)).to(self.x.device)
            self.focal_stack_spectral = self.focal_stack_spectral_arch.to(self.x.device)
            self.focal_stack = self.focal_stack_arch.to(self.x.device)
            self.lams = self.lams_arch.to(self.x.device)
            gc.collect()
            torch.cuda.empty_cache()

    @torch.no_grad()
    def optimize(self, solver: object, spec:dim.SpectralDim, config:dict) -> None:
        abs_tol = config['opt_abs_tol']
        rel_tol = config['opt_rel_tol']
        if config['verbosity_level'] >= 1:
            solver_name = solver.__class__.__name__
            print(f"Optimizing with {solver_name}")
            try:
                reg_name = solver.reg.__class__.__name__
                print(f"Regularizer: {reg_name}")
            except Exception:
                print("No regularizer")
        rho = getattr(solver, "rho", getattr(solver, "admm_rho", 0.0))
        if rho < 0:
            raise ValueError("rho must be ≥ 0")
        def fidelity_from_X(X_spec):
            return self.fidelityLoss(X_spec)
        best = {
            "loss": float("inf"),
            "X": None,
            "Z": None,
            "U": None,
            "recon_spectral": None,
            "recon": None,
            "iter": 0,
        }
        bad_streak = 0
        Z_prev = solver.Z.clone()
        frob = lambda t: torch.linalg.vector_norm(t)
        n = solver.X.numel()
        for it in range(1, config['opt_max_iter'] + 1):
            solver.step()
            if it % config['opt_chk_every'] != 0:
                continue
            Xk, Zk, Uk = solver.X, solver.Z, solver.U
            r_norm = frob(Xk - Zk)
            s_norm = frob(rho * (Zk - Z_prev))
            X_norm = frob(Xk)
            Z_norm = frob(Zk)
            U_norm = frob(Uk)
            eps_pri  = (n**0.5) * abs_tol + rel_tol * max(X_norm, Z_norm)
            eps_dual = (n**0.5) * abs_tol + rel_tol * (rho * U_norm)
            loss = fidelity_from_X(solver.recon_spectral)
            if config['verbosity_level'] >= 1:
                print(f"[it {it:03d}] loss={loss:.6e}  r={r_norm:.3e} s={s_norm:.3e} | eps_pri={eps_pri:.3e} eps_dual={eps_dual:.3e}")
            if config['opt_save_intermediate']:
                recon_spatial = torch.real(torch.fft.ifft2(solver.recon_spectral))[:, :self.h, :self.w]
                recon_rgb = vis.draw_hpim(spec.recover(recon_spatial), self.lams, method=self.config['draw_method'])
                plt.imsave(self.config['result_path'] + f'/iter_{it}.png', recon_rgb)
            improved = loss <= best["loss"] * (1 + config['opt_degrade_tol'])
            if improved:
                bad_streak = 0
                best.update({
                    "loss": float(loss),
                    "X": Xk.clone(),
                    "Z": Zk.clone(),
                    "U": Uk.clone(),
                    "recon_spectral": solver.recon_spectral.clone(),
                    "recon": solver._recover_from_spectral(solver.recon_spectral).clone(),
                    "iter": it,
                })
            else:
                bad_streak += 1
            converged = (r_norm <= eps_pri) and (s_norm <= eps_dual)
            out_of_patience = bad_streak >= config['opt_patience']
            if converged or out_of_patience:
                if config['verbosity_level'] >= 1:
                    reason = "ADMM converged" if converged else f"early stop (no improvement for {config['opt_patience']} checks)"
                    print(f"Stopping at iter {it}: {reason}. Best at iter {best['iter']} with loss={best['loss']:.6e}")
                break
            Z_prev = Zk.clone()
        if best["X"] is not None:
            solver.X = best["X"]
            solver.Z = best["Z"]
            solver.U = best["U"]
            self.recon_spectral = best["recon_spectral"]
            self.recon = best["recon"]
        else:
            self.recon_spectral = solver.recon_spectral
            self.recon = solver._recover_from_spectral(self.recon_spectral)
        return

    def solve(self, solver:object) -> None:
        '''
        Solver the inverse problem.
        solver: solver object, needs to have a solve() method.
        '''
        solver_name = solver.__class__.__name__
        solver.solve()
        #self.recon_spectral = solver.recon_spectral
        self.recon = solver.recon

    def change2inv_fs(self, spec:dim.SpectralDim, ndims:int = None) -> None:
        '''
        Change the filter stack to the inverse filter stack. 
        Inverse filter stack creates results projected to the spectral basis learned from Harvard dataset.
        The channels are in the decreasing variance order.
        spec: SpectralDim object with the basis.
        '''
        self.basis = spec.basis
        self.basis_object = spec
        assert self.basis.shape[0] == self.v
        if ndims is None:
            ndims = self.basis.shape[1]
        inv_fs = torch.zeros(self.u, ndims, self.fh, self.fw, device=self.config['device'])
        for i in range(ndims):
            inv_fs[:,i,:,:] = torch.einsum('v,uvhw->uhw', self.basis[:,i], self.filter_stack)
        self.filter_stack = inv_fs
        self.filter_stack_spectral = torch.fft.fft2(self.filter_stack, s=(self.fsh, self.fsw))
        self.v = ndims
        return

    def down_sample_fs(self, bins:torch.tensor) -> None:
        '''
        When the forward model has too high spectral resolution,
        but we don't need such high resolution in our recovery, 
        we can downsample the filter stack along 'v' axis. 
        Note that this will cause model mismatch. Especially usefull when using PCA basis.
        '''
        if type(bins) == int:
            bins = torch.arange(bins, device=self.config['device'])
        self.filter_stack = torch.index_select(self.filter_stack, 1, bins)
        self.filter_stack_spectral = torch.fft.fft2(self.filter_stack, s=(self.fsh, self.fsw))
        self.v = bins.shape[0]

    def pick_measurements(self, measurements:list) -> None: #TODO: UPDATE
        '''
        Pick measurements from the focal stack. (along u axis.)
        measurements: list of indices of the measurements to be picked.
        '''
        self.filter_stack = self.filter_stack[measurements,:,:,:]
        self.filter_stack_spectral = self.filter_stack_spectral[measurements,:,:,:]
        self.focal_stack = self.focal_stack[measurements,:,:]
        self.focal_stack_spectral = torch.fft.fft2(self.focal_stack, s=(self.fsh_ip, self.fsw_ip))
        self.u = len(measurements)

    def change_X(self, x:torch.tensor) -> None: #TODO: UPDATE
        '''
        Change the input image.
        '''
        self.x = x
        self.c, self.h, self.w = x.shape
        assert self.c == self.v
        self.fsh, self.fsw  = self.h + self.fh - 1, self.w + self.fw - 1
        self.x_spectral = torch.fft.fft2(x, s=(self.fsh, self.fsw)) # WRONG
        self.focal_stack_spectral = self._f(self.x_spectral) # WRONG
        self.focal_stack = torch.real(torch.fft.ifft2(self.focal_stack_spectral))
        self.recon_spectral = None
        self.recon = None

    def draw_input_RGB(self) -> None:
        rgb = vis.draw_hpim(self.x, method=self.config['draw_method'], draw=False)
        return rgb

    def draw_result_RGB(self) -> None:
        rgb = vis.draw_hpim(self.recon, method=self.config['draw_method'], draw=False)
        return rgb

    def draw_focal_stack(self) -> None:
        fig, ax = plt.subplots(1, self.u, figsize=(3*self.u, 3))
        for i in range(self.u):
            ax[i].imshow(self.focal_stack[i,:,:].cpu(), cmap='gray')
            ax[i].set_title('Measurement ' + str(i))
            ax[i].axis('off')
        return fig