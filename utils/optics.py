import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

class optics:
    """
    This class contains optics related functions. You can either generate a filter stack using:
        - Wavelengths (range and resolution)
        - focal length (including chromatic focal shift)
        - f-number, aperture diameter
        - Aperture shape.
        - scene depth (z-axis, objective lens is at z=0)
        - sensor pixel size (sensor extends from -inf to +inf in x and y directions)
    Or, use the experimentally measured PSFs directly. See config to modify.
    """
    def __init__(self, config:dict, draw_results:bool = False) -> None:
        self.config = config
        self.device = config['device']
        self.lams = torch.arange(self.config['min_wavelength'], self.config['max_wavelength']+1, self.config['wavelength_resolution'], device=self.device, dtype=torch.float32)
        if config['generate_filter_stack']:
            self.scale, self.aux_dict = self.get_scale()
            if draw_results:
                fig1 = draw_cfs_and_cameras(config, self.aux_dict)
                fig1.savefig(self.config['result_path'] + '/' + 'cfs_and_cameras.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig1)
                fig2 = draw_scale_heatmap(self.scale, self.aux_dict['camera_locs'], self.lams)
                fig2.savefig(self.config['result_path'] + '/' + 'PSF_size_for_camera_loc_vs_wavelength.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig2)
            self.filter_stack = self.get_filter_stack(self.scale)
            if draw_results:
                fig3 = draw_filter_stack(self.filter_stack, self.aux_dict['camera_locs'], self.lams)
                fig3.savefig(self.config['result_path'] + '/' + 'filter_stack.png', bbox_inches='tight', pad_inches=0)
                plt.close(fig3)
        else:
            self.filter_stack = self.load_filter_stack()

    def get_lams_vs_focal_length(self) -> torch.tensor:
        """
        Calculate the focal length of each wavelength based on base chromatic focal shift.
        Returns:
            focal_lengths: torch.tensor of shape n_wavelengths
        """
        base_chomatic_focal_shift = np.loadtxt(self.config['chromatic_focal_shift_file']) # chromatic focal shift of the lens for f=1mm.
        base_chromatic_focal_shift = np.interp(self.lams.cpu().numpy(), base_chomatic_focal_shift[:,0], base_chomatic_focal_shift[:,1])
        base_chomatic_focal_shift = torch.tensor(base_chromatic_focal_shift, device=self.device, dtype=torch.float32)
        focal_lengths = self.config['base_focal_length'] + base_chomatic_focal_shift * self.config['base_focal_length']
        return focal_lengths

    def get_camera_locs(self, z_img:torch.tensor) -> torch.tensor:
        """
        This function calculates sensor locations based on the focal lengths of each wavelength.
        We have two options:
            - Wavelengths in focus are equally spaced. (sample along wavelength axis)
            - Camera locations are equally spaced. (sample along z-axis)
        returns:
            camera_locs: torch.tensor of shape n_camera_locs 
        """
        if self.config['sampling_strategy'] == 'z-axis':
            if self.config['n_camera_locs'] > 1:
                camera_locs = torch.linspace(z_img[0], z_img[-1], self.config['n_camera_locs'], device=self.device, dtype=torch.float32)
            elif self.config['n_camera_locs'] == 1:
                camera_locs = torch.tensor([(z_img[0] + z_img[-1]) / 2], device=self.device, dtype=torch.float32)
            else:
                raise ValueError("n_camera_locs must be >= 1")
        elif self.config['sampling_strategy'] == 'wavelength':
            if self.config['n_camera_locs'] > 1:
                wavelengths__in_focus = torch.linspace(self.config['min_wavelength'], self.config['max_wavelength'], self.config['n_camera_locs'], device=self.device, dtype=torch.float32)
            elif self.config['n_camera_locs'] == 1:
                wavelengths__in_focus = torch.tensor([(self.config['min_wavelength'] + self.config['max_wavelength']) / 2], device=self.device, dtype=torch.float32)
            else:
                raise ValueError("n_camera_locs must be >= 1")
            camera_locs = torch.zeros(self.config['n_camera_locs'], device=self.device, dtype=torch.float32)
            for i, wl in enumerate(wavelengths__in_focus):
                idx = torch.argmin(torch.abs(self.lams - wl))
                camera_locs[i] = z_img[idx]
        return camera_locs
        
    def get_scale(self) -> torch.tensor:
        """
        Scale is the size of each PSF in pixels in the filter stack.
        Returns:
            scale: torch.tensor of shape n_camera_locs X n_wavelengths
            aux_dict:
                camera_locs: sensor locations in z-axis
                cfs: focal lengths of each wavelength
        """
        cfs = self.get_lams_vs_focal_length() # focal lengths of each wavelength
        z_image = 1 / (1 / self.config['object_depth'] + 1 / cfs) # image distance for each wavelength
        camera_locs = self.get_camera_locs(z_image)
        scale = torch.zeros((self.config['n_camera_locs'], len(self.lams)), device=self.device, dtype=torch.float32)
        for i, cam_loc in enumerate(camera_locs):
            # Triangle similarity to calculate scale, we will use aperture diameter, z_image and current camera location.
            scale[i] = self.config['aperture_diameter'] * (z_image - cam_loc) / z_image / self.config['pixel_size']
        aux_dict = {'camera_locs': camera_locs, 'cfs': cfs}
        return scale, aux_dict
    

    def get_filter_stack(self, scale:torch.tensor) -> torch.tensor:
        """
        This function generates the filter stack given a base psf shape and the scale tensor.
        scale: the scale tensor, stores relative sizes of each filter.
        """
        psf = torch.load(self.config['aperture_code_file'], weights_only=False).to(self.device)
        filter_size = int(torch.max(torch.abs(scale))) 
        # Make filter size odd
        if filter_size % 2 == 0:
            filter_size += 1
        # Center of the filter
        r = filter_size // 2
        u = scale.shape[0]
        v = scale.shape[1]
        filter_stack = torch.zeros((u,v,filter_size,filter_size)).to(self.device)
        for i in range(u):
            for j in range(v):
                s = int(torch.abs(scale[i,j]))
                if s in [-1,0,1]:
                    filter_stack[i,j, r, r] = 1
                    continue
                f = torch.nn.functional.interpolate(torch.unsqueeze(torch.unsqueeze(psf,0),0), size=(s, s), mode='bilinear').squeeze()
                if scale[i,j] < 0:
                    f = torch.flip(f, dims=[0,1])
                f = f / torch.sum(f)
                filter_stack[i,j,(r-int(s/2)):(r-int(s/2)+s),(r-int(s/2)):(r-int(s/2)+s)] = f
        filter_stack = filter_stack / scale.shape[1]
        return filter_stack
    
def draw_cfs_and_cameras(config, aux_dict) -> plt.Figure:
    """
    Draw chromatic focal shift and camera planes in an overlapped plot.
    Horizontal (from left) and vertical (from bottom) lines stop at the
    intersection with the CFS curve. Axes limits are set first and reused.
    """
    cfs = aux_dict['cfs'].detach().cpu().numpy()
    camera_locs = aux_dict['camera_locs'].detach().cpu().numpy()
    lams = np.arange(config['min_wavelength'],
                     config['max_wavelength'] + 1,
                     config['wavelength_resolution'])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lams, cfs, label='Chromatic Focal Shift', color='black', linewidth=2, zorder=3)
    x_min, x_max = float(lams.min()), float(lams.max())
    y_min = float(min(cfs.min(), camera_locs.min()))
    y_max = float(max(cfs.max(), camera_locs.max()))
    x_pad = 0.02 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    for cam_loc in camera_locs:
        idx = int(np.argmin(np.abs(cfs - cam_loc)))
        lam_intersect = float(lams[idx])
        y_intersect = float(cfs[idx])
        ax.hlines(y=cam_loc, xmin=x_left, xmax=lam_intersect,
                  color='red', linestyle='--', linewidth=1, zorder=2)
        ax.vlines(x=lam_intersect, ymin=y_bottom, ymax=y_intersect,
                  color='red', linestyle='--', linewidth=1, zorder=2)
        ax.plot(lam_intersect, y_intersect, marker='o', color='blue', markersize=6, zorder=4)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Focal Length / Camera Location (mm)')
    ax.set_title('Chromatic Focal Shift and Camera Locations')
    ax.legend()
    camera_legend_line = mpl.lines.Line2D(
        [], [], color='red', linestyle='--', linewidth=1,
        marker='o', markerfacecolor='blue', markeredgecolor='blue',
        label='Camera Locations'
    )
    ax.legend(handles=[ax.lines[0], camera_legend_line])
    plt.tight_layout()
    return fig

def draw_scale_heatmap(scale: torch.tensor, cam_locs: torch.tensor, lams: torch.tensor) -> plt.Figure:
    '''
    Displays the scale as a heatmap.
    0 is mapped to black, negative values to blue, positive values to red.
    '''
    if isinstance(scale, torch.Tensor):
        scale = scale.detach().cpu().numpy()

    vmin = np.min(scale)
    vmax = np.max(scale)
    vcenter = 0.0

    colors = [
        (0.0, "blue"),
        (0.5, "black"),
        (1.0, "red"),
    ]

    cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_black_red", colors)
    norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    fig, ax = plt.subplots(figsize=(scale.shape[1]*0.2 + 2, scale.shape[0]*0.2 + 2))
    im = ax.imshow(scale, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('PSF Size (pixels)', rotation=270, labelpad=15)
    cbar.set_ticks([vmin, 0, vmax])
    cbar.set_ticklabels([f'{vmin:.2f}', '0', f'{vmax:.2f}'])

    ax.set_xticks(np.arange(len(lams)))
    ax.set_xticklabels([f'{lam:.0f}' for lam in lams.cpu().numpy()], rotation=45)
    ax.set_yticks(np.arange(len(cam_locs)))
    ax.set_yticklabels([f'{loc:.1f}' for loc in cam_locs.cpu().numpy()])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Camera Location (mm)')
    ax.set_title('PSF Size for Each Wavelength and Camera Location')

    plt.tight_layout()
    return fig

def draw_filter_stack(filter_stack:torch.tensor, cam_locs:torch.tensor, lams:torch.tensor) -> plt.figure:
    '''
    This function displays the filter stack as agrid.
    '''
    if type(filter_stack) == torch.Tensor:
        filter_stack = filter_stack.detach().cpu().numpy()
    u, v, h, w = filter_stack.shape
    loc = [0, 0]
    fig, ax = plt.subplots(filter_stack.shape[0], filter_stack.shape[1], figsize=(v*2, u*2))
    for i in range(u):
        for j in range(v):
            ax[i,j].imshow(filter_stack[i,j], cmap='gray')
            if i == 0:
                ax[i,j].set_title(f'{lams[j].item():.0f} nm', fontsize=10)
            if j == 0:
                ax[i,j].set_ylabel(f'z={cam_locs[i].item():.1f} mm', fontsize=10)
            ax[i, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax[i, j].spines.values():
                spine.set_visible(False)
    plt.tight_layout()
    return fig