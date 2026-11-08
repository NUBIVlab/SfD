import time
import json
from pathlib import Path

import torch

import utils.solvers as solvers
import utils.regularizers as regularizers

def config_debug():
    config = {
        'experiment_name' : 'debug',
        'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
        'min_wavelength': 400, # in nm
        'max_wavelength': 700, # in nm
        'wavelength_resolution': 10, # in nm
        'base_focal_length': 50, # in mm
        'chromatic_focal_shift_file': 'data/chromatic_focal_shift/NBK7.txt', 
        'n_camera_locs': 5, 
        'sampling_strategy': 'z-axis',
        'aperture_code_file': 'data/apertures/circ.pt', 
        'generate_filter_stack': True,
        'object_depth': 10e6, # in mm If you set this to a small value some CFS plots will look weird (Camera locations will not align with CFS plot beautifully, instead they will align with z_image plot), but method will still work.
        'aperture_diameter': 15, # in mm
        'pixel_size': 5.76e-3, # in mm
        'gaussian_noise_mean': 0.0,
        'gaussian_noise_variance': 1e-6,
        'poisson_noise_scaling': 9.6e6,
        'simulate_measurements' : True,
        'solver' : solvers.pnp_ADMM,
        'regularizer' : regularizers.HSDT,
        'regularizer_settings' : {'size' : 'l', 'noise' : 'octopus', 'scale' : 1.0},
        'opt_chk_every' : 1, 
        'opt_max_iter' : 10,
        'opt_abs_tol' : 1e-4,
        'opt_rel_tol' : 1e-3,
        'opt_patience' : 3,
        'opt_degrade_tol' : 1e-2,
        'opt_save_intermediate' : False,
        'ADMM_crop' : False, # We can disable the crop operator if: a-) PSF is very small. b-) image has a black background (dark borders)
        'rho' : 1e-16,
        'divmat_method' : 'ridge', #inv and pinv are fast. ridge is slow but stable. optim is very slow should only be used for debugging. # OUT OF DATE, no longer used.
        'dataset' : 'KAIST',
        'img_id' : 0,
        'img_size' : 512,
        'result_path' : 'results/',
        'modify_wavelengths_from_data' : True,
        'draw_method' : '1931',
        'basis_file' : 'data/basis/spectral_basis_data_420-10-720.npy',
        'n_dims': 12,
        'verbosity_level' : 2 # 0: silent, 1: some info, 2: detailed
    }
    if config['max_wavelength'] > 710:
        config['basis_file'] = 'data/basis/spectral_basis_data_420-10-720.npy'
    elif config['min_wavelength'] < 410:
        config['basis_file'] = 'data/basis/spectral_basis_data_400-10-700.npy'
    config['result_path'] = 'results/' + config['experiment_name'] + '/' + time.strftime("%Y-%m-%d %H:%M")
    Path(config['result_path']).mkdir(parents=True, exist_ok=True)
    config_path = Path(config['result_path']) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=str)
    torch.set_grad_enabled(False)
    return config

def config_simulation():
    config = {
        'experiment_name' : 'simulation',
        'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
        'min_wavelength': 400, # in nm
        'max_wavelength': 700, # in nm
        'wavelength_resolution': 10, # in nm
        'base_focal_length': 50, # in mm
        'chromatic_focal_shift_file': 'data/chromatic_focal_shift/NBK7.txt', 
        'n_camera_locs': 5, 
        'sampling_strategy': 'z-axis',
        'aperture_code_file': 'data/apertures/circ.pt', 
        'generate_filter_stack': True,
        'object_depth': 10e6, # in mm If you set this to a small value some CFS plots will look weird (Camera locations will not align with CFS plot beautifully, instead they will align with z_image plot), but method will still work.
        'aperture_diameter': 18, # in mm
        'pixel_size': 5.76e-3, # in mm
        'gaussian_noise_mean': 0.0,
        'gaussian_noise_variance': 1e-6,
        'poisson_noise_scaling': 9.6e6,
        'simulate_measurements' : True,
        'solver' : solvers.pnp_ADMM,
        'regularizer' : regularizers.HSDT,
        'regularizer_settings' : {'size' : 'm', 'noise' : 'octopus', 'scale' : 1.0},
        'opt_chk_every' : 1, 
        'opt_max_iter' : 5,
        'opt_abs_tol' : 1e-4,
        'opt_rel_tol' : 1e-3,
        'opt_patience' : 1,
        'opt_degrade_tol' : 1e-2,
        'opt_save_intermediate' : False,
        'ADMM_crop' : False, # We can disable the crop operator if: a-) PSF is very small. b-) image has a black background (dark borders)
        'rho' : 1e-10,
        'divmat_method' : 'ridge', #inv and pinv are fast. ridge is slow but stable. optim is very slow should only be used for debugging. # OUT OF DATE, no longer used.
        'dataset' : 'KAIST',
        'img_id' : 0,
        'img_size' : 512,
        'result_path' : 'results/',
        'modify_wavelengths_from_data' : True,
        'draw_method' : '1931',
        'basis_file' : 'data/basis/spectral_basis_data_420-10-720.npy',
        'n_dims': 6,
        'verbosity_level' : 2 # 0: silent, 1: some info, 2: detailed
    }
    if config['max_wavelength'] > 710:
        config['basis_file'] = 'data/basis/spectral_basis_data_420-10-720.npy'
    elif config['min_wavelength'] < 410:
        config['basis_file'] = 'data/basis/spectral_basis_data_400-10-700.npy'
    config['result_path'] = 'results/' + config['experiment_name'] + '/' + time.strftime("%Y-%m-%d %H:%M")
    Path(config['result_path']).mkdir(parents=True, exist_ok=True)
    config_path = Path(config['result_path']) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=str)
    torch.set_grad_enabled(False)
    return config

def config_hardware():
    config = {
        'experiment_name' : 'hardware',
        'device': torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
        'min_wavelength': 440, # in nm
        'max_wavelength': 720, # in nm
        'wavelength_resolution': 10, # in nm
        'focal_stack_dir': 'data/hardware/focal_stacks/',
        'gt_dir': 'data/hardware/gt/',
        'filter_stack_file': 'data/hardware/FS_LCTF_proc_corrected.npy', 
        'generate_filter_stack': False,
        'simulate_measurements' : False,
        'solver' : solvers.pnp_ADMM,
        'regularizer' : regularizers.HSDT,
        'regularizer_settings' : {'size' : 'l', 'noise' : 'octopus', 'scale' : 1.0},
        'opt_chk_every' : 1, 
        'opt_max_iter' : 5,
        'opt_abs_tol' : 1e-4,
        'opt_rel_tol' : 1e-3,
        'opt_patience' : 1,
        'opt_degrade_tol' : 1e-2,
        'opt_save_intermediate' : False,
        'ADMM_crop' : False, # We can disable the crop operator if: a-) PSF is very small. b-) image has a black background (dark borders)
        'rho' : 1e-9,
        'divmat_method' : 'ridge', #inv and pinv are fast. ridge is slow but stable. optim is very slow should only be used for debugging. # OUT OF DATE, no longer used.
        'dataset' : 'focal_stacks',
        'img_id' : 'noah',
        'result_path' : 'results/',
        'modify_wavelengths_from_data' : False,
        'draw_method' : '1931',
        'basis_file' : 'data/basis/spectral_basis_data_420-10-720.npy',
        'n_dims': 10,
        'verbosity_level' : 2 # 0: silent, 1: some info, 2: detailed
    }
    if config['max_wavelength'] > 710:
        config['basis_file'] = 'data/basis/spectral_basis_data_420-10-720.npy'
    elif config['min_wavelength'] < 410:
        config['basis_file'] = 'data/basis/spectral_basis_data_400-10-700.npy'
    config['result_path'] = 'results/' + config['experiment_name'] + '/' + time.strftime("%Y-%m-%d %H:%M")
    Path(config['result_path']).mkdir(parents=True, exist_ok=True)
    config_path = Path(config['result_path']) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=str)
    torch.set_grad_enabled(False)
    return config


def configuration_param_search():
    config = {
        'experiment_name' : 'param_search',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'min_wavelength': 400, # in nm
        'max_wavelength': 700, # in nm
        'wavelength_resolution': 10, # in nm
        'base_focal_length': 50, # in mm
        'chromatic_focal_shift_file': 'data/chromatic_focal_shift/NBK7.txt', 
        'n_camera_locs': 5, 
        'sampling_strategy': 'z-axis',
        'aperture_code_file': 'data/apertures/circ.pt', 
        'generate_filter_stack': True,
        'object_depth': 10e6, # in mm If you set this to a small value some CFS plots will look weird (Camera locations will not align with CFS plot beautifully, instead they will align with z_image plot), but method will still work.
        'aperture_diameter': 15, # in mm
        'pixel_size': 5.76e-3, # in mm
        'gaussian_noise_mean': 0.0,
        'gaussian_noise_variance': 1e-6,
        'poisson_noise_scaling': 9.6e6,
        'simulate_measurements' : True,
        'solver' : solvers.pnp_ADMM,
        'regularizer' : regularizers.HSDT,
        'regularizer_settings' : {'size' : 'l', 'noise' : 'octopus', 'scale' : 1.0},
        'opt_chk_every' : 1, 
        'opt_max_iter' : 10,
        'opt_abs_tol' : 1e-4,
        'opt_rel_tol' : 1e-3,
        'opt_patience' : 3,
        'opt_degrade_tol' : 1e-2,
        'opt_save_intermediate' : False,
        'ADMM_crop' : False, # We can disable the crop operator if: a-) PSF is very small. b-) image has a black background (dark borders)
        'rho' : [1e-16, 1e-12],
        'divmat_method' : 'ridge', #inv and pinv are fast. ridge is slow but stable. optim is very slow should only be used for debugging. # OUT OF DATE, no longer used.
        'dataset' : 'KAIST',
        'img_id' : 0,
        'img_size' : 512,
        'result_path' : 'results/',
        'modify_wavelengths_from_data' : True,
        'draw_method' : '1931',
        'basis_file' : 'data/basis/spectral_basis_data_420-10-720.npy',
        'n_dims': 12,
        'verbosity_level' : 1 # 0: silent, 1: some info, 2: detailed
    }
    if config['max_wavelength'] > 710:
        config['basis_file'] = 'data/basis/spectral_basis_data_420-10-720.npy'
    elif config['min_wavelength'] < 410:
        config['basis_file'] = 'data/basis/spectral_basis_data_400-10-700.npy'
    config['result_path'] = 'results/' + config['experiment_name'] + '/' + time.strftime("%Y-%m-%d %H:%M")
    Path(config['result_path']).mkdir(parents=True, exist_ok=True)
    config_path = Path(config['result_path']) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4, default=str)
    torch.set_grad_enabled(False)
    return config