import numpy as np
import torch

class SpectralDim:
    """ 
    Class that contaons the functions for spectral dimension reduction
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = config['device']
        self.lams = torch.arange(config['min_wavelength'], config['max_wavelength'] + config['wavelength_resolution'], config['wavelength_resolution'], device=self.device)
        lams_np = self.lams.cpu().numpy()
        data = np.load(self.config['basis_file'], allow_pickle=True).item()
        basis_raw = data['basis']
        basis_interp = np.zeros((self.lams.shape[0], basis_raw.shape[1]), dtype=np.float32)
        # Basis files names have the format: spectral_basis_data_{lam_min}-{lam_max}-{lam_res}.npy
        lam_basis_min = self.config['basis_file'].split('_')[-1].split('-')[0]
        lam_basis_max = self.config['basis_file'].split('_')[-1].split('-')[2].split('.')[0]
        lam_basis_step = self.config['basis_file'].split('_')[-1].split('-')[1]
        lam_basis = np.arange(int(lam_basis_min), int(lam_basis_max) + 1, int(lam_basis_step))
        for i in range(basis_raw.shape[1]):
            basis_interp[:,i] = np.interp(lams_np, lam_basis, basis_raw[:,i])
        self.basis = torch.tensor(basis_interp, device=self.device, dtype=torch.float32)
        self.singular_values = torch.tensor(data['singular_values'], device=self.device, dtype=torch.float32)
    
    def project(self, hpim: torch.tensor, dim: int) -> torch.tensor:
        """
        Projects a hyperspectral image to a lower dimensional space.
        hpim: hyperspectral image of shape CxHxW.
        dim: dimension of the projection.
        returns: projected hyperspectral image of shape PxHxW.
        """
        l,h,w = hpim.shape
        data = torch.reshape(hpim, (l, -1))
        z = data.T @ self.basis[:, 0:dim]
        hpim_projected = torch.reshape(z.T, (dim,h,w))
        return hpim_projected
    
    def recover(self, hpim: torch.tensor) -> torch.tensor:
        """
        Recovers a hyperspectral image from its lower dimensional representation.
        hpim: hyperspectral image of shape PxHxW.
        """
        dim,h,w = hpim.shape
        l, u = self.basis.shape
        data = torch.reshape(hpim, (dim, -1))
        x_rec = (data.T @ self.basis[:, 0:dim].T).T
        hpim_recovered = torch.reshape(x_rec, (l,h,w))
        return hpim_recovered