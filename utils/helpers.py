import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt

import utils.optics as opt
import utils.ip as ip
import utils.data as data
import utils.spectral_basis as dim
import utils.visualization as vis
import utils.solvers as sol
import utils.regularizers as reg
import utils.metrics as metrics

def load_simulation_data(config):
    '''
    Loads the simulation data.
    '''
    if config['dataset'] == 'CAVE':
        lams, hpim = data.load_CAVE(config['img_id'], config['img_size'], config['device'])
    elif config['dataset'] == 'Harvard':
        lams, hpim = data.load_Harvard(config['img_id'], config['img_size'], config['device'])
    elif config['dataset'] == 'ICVL':
        lams, hpim = data.load_ICVL(config['img_id'], config['img_size'], config['device'])
    elif config['dataset'] == 'KAIST':
        lams, hpim = data.load_KAIST(config['img_id'], config['img_size'], config['device'])
    if config['modify_wavelengths_from_data']:
        config['min_wavelength'] = int(lams[0])
        config['max_wavelength'] = int(lams[-1])
        config['wavelength_resolution'] = int(lams[1]-lams[0])
    return lams, hpim

#def radiometric_correction(rec_hsi, config): # Radimetric correction is already applied to the GT images. We can skip this step. Raw data is available upon request. 
#    pass
          
def simulated_results(config):
    lams, hpim = load_simulation_data(config)
    if config['verbosity_level'] >= 1:
        rgb_gt = vis.draw_hpim(hpim, lams, draw=False, method=config['draw_method'])
        plt.imsave(config['result_path'] + '/' + 'gt.png', rgb_gt)
    optics = opt.optics(config, draw_results = (config['verbosity_level'] >= 1))
    sb = dim.SpectralDim(config)
    simulation = ip.inverse_problem(hpim, None, optics.filter_stack, config)
    fig1 = simulation.draw_focal_stack()
    fig1.savefig(config['result_path'] + '/' + 'focal_stack.png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close(fig1)
    if config['verbosity_level'] >= 2:
        print('Simulation is Done.')
    simulation.change2inv_fs(sb, config['n_dims'])
    regularizer = config['regularizer'](config['regularizer_settings'], sb, config)
    solver = config['solver'](simulation, regularizer, config)
    simulation.optimize(solver, sb, config)
    #simulation.solve(solver)
    rec_hsi = simulation.recon
    rec_rgb = vis.draw_hpim(rec_hsi, lams)
    gt_rgb = vis.draw_hpim(hpim, lams)
    plt.imsave(config['result_path'] + '/' + 'recon.png', rec_rgb)
    if config['verbosity_level'] >= 2:
        print('Reconstruction is Done.')
    psnr = metrics.PSNR(hpim, rec_hsi).cpu().item()
    return {'PSNR': psnr, 'rec_rgb': rec_rgb, 'gt_rgb': gt_rgb}

def hardware_results(config):
    lams = torch.arange(config['min_wavelength'], config['max_wavelength'] + 1, config['wavelength_resolution'], device=config['device'], dtype=torch.float32)
    data = sp.io.loadmat(config['gt_dir'] + config['img_id'] + '.mat')
    lams = torch.tensor(data['lams'].squeeze(), device=config['device'], dtype=torch.float32)
    gt = torch.tensor(data['hpim'], device=config['device'], dtype=torch.float32)
    focal_stack = torch.tensor(np.load(config['focal_stack_dir'] + config['img_id'] + '.npy').astype(np.float32)).to(config['device'])
    filter_stack = torch.tensor(np.load(config['filter_stack_file']).astype(np.float32)).to(config['device'])
    sb = dim.SpectralDim(config)
    inv = ip.inverse_problem(x=None, meas=focal_stack, filter_stack=filter_stack, config=config)
    regularizer = config['regularizer'](config['regularizer_settings'], sb, config)
    inv.change2inv_fs(sb, config['n_dims'])
    solver = config['solver'](inv, regularizer, config)
    inv.optimize(solver, sb, config)
    rec_hsi = inv.recon
    #rec_hsi = radiometric_correction(rec_hsi, config) # Radimetric correction is already applied to the GT images. We can skip this step. Raw data is available upon request. 
    print(rec_hsi.shape, lams.shape, gt.shape)
    rec_rgb = vis.draw_hpim(rec_hsi, lams)
    gt_rgb = vis.draw_hpim(gt, lams)
    plt.imsave(config['result_path'] + '/' + 'gt.png', gt_rgb)
    plt.imsave(config['result_path'] + '/' + 'recon.png', rec_rgb)
    np.save(config['result_path'] + '/' + 'recon.npy', rec_hsi.cpu().numpy())
    if config['verbosity_level'] >= 2:
        print('Reconstruction is Done.')
    psnr = metrics.PSNR(gt, rec_hsi).cpu().item()
    return {'PSNR': psnr, 'rec_rgb': rec_rgb, 'gt_rgb': gt_rgb}