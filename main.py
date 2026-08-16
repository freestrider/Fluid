import torch
import vispy
from vispy.scene import visuals
import render
from time import time,sleep
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
from sklearn.metrics import r2_score

time_space_scale = 4
Dt = 0.02/time_space_scale
Dx = 1e0/time_space_scale
W,H = int(64*time_space_scale),int(64*time_space_scale)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
render_Dt = 0.1
viscosity = 2
transience = torch.tensor([1,1],device=device)*25
density =1
point_count = 2000
visual_scaling = 0.2/time_space_scale

cuboid_L_corner = (W*1//5,H*1//5)
cuboid_R_corner = (W*2//5,H*3//5)
stream_width = 6
particle_slower = 1




class fixed_object:
    def __init__(self,occupied_grid,SDF,center,area_func,normal_func=None):
        
        
        self.center = center 
        def sdf(position):
            rel_pos = position-self.center
            return SDF(rel_pos)
        self.SDF = sdf
        """if normal_func ==None:
            def foo(position):
                rel_position = position-center
                tangent = """

        def get_normal(position):
            rel_pos = position-self.center
        self.get_normal = get_normal


        



def get_k_grid(w, h, device):
   # Generates frequencies for a 2D grid
   kx = torch.fft.fftfreq(w, d=Dx).to(device) * 2 * torch.pi
   ky = torch.fft.fftfreq(h, d=Dx).to(device) * 2 * torch.pi
   kx_g, ky_g = torch.meshgrid(kx, ky, indexing='ij')
   return torch.stack((kx_g, ky_g), dim=-1)  # Shape: (W, H, 2)

k_grid = get_k_grid(W, H, device)
k_grid_mag_squared = torch.sum(k_grid*k_grid,dim=-1).unsqueeze(-1)

k_sq = torch.sum(k_grid**2, dim=-1, keepdim=True)
k_sq[k_sq == 0] = 1.0  # Avoid division by zero at DC component


def exponential_decay(x, a, b):
    """Exponential decay function: a * exp(-b * x)"""
    return a * np.exp(-b * x)


def exponential_growth(x, a, b):
    """Exponential growth function: a * exp(b * x)"""
    return a * np.exp(b * x)


def find_drag_peaks_and_troughs(drag_signal, prominence=None, distance=None):
    """
    Find peaks and troughs in drag signal (time domain, not frequency).
    
    Parameters:
    -----------
    drag_signal : array-like
        The drag force data over time
    prominence : float, optional
        Minimum prominence of peaks (helps filter noise)
    distance : int, optional
        Minimum distance between peaks in samples
    
    Returns:
    --------
    dict with keys:
        - 'peaks': indices of peaks
        - 'troughs': indices of troughs
        - 'peak_values': values at peaks
        - 'trough_values': values at troughs
    """
    drag_array = np.asarray(drag_signal)
    
    # Find peaks
    peaks, peak_props = find_peaks(drag_array, prominence=prominence, distance=distance)
    peak_values = drag_array[peaks]
    
    # Find troughs (peaks in inverted signal)
    troughs, trough_props = find_peaks(-drag_array, prominence=prominence, distance=distance)
    trough_values = drag_array[troughs]
    
    return {
        'peaks': peaks,
        'troughs': troughs,
        'peak_values': peak_values,
        'trough_values': trough_values,
        'peak_props': peak_props,
        'trough_props': trough_props
    }


def fit_exponential_to_peaks(drag_signal, peaks=None, prominence=None, distance=None, fit_type='decay'):
    """
    Fit exponential curve to drag peaks.
    
    Parameters:
    -----------
    drag_signal : array-like
        The drag force data
    peaks : array-like, optional
        Peak indices; if None, will find them
    prominence : float, optional
        Minimum prominence for peak detection
    distance : int, optional
        Minimum distance between peaks
    fit_type : str
        'decay' or 'growth' for exponential type
    
    Returns:
    --------
    dict with:
        - 'params': (a, b) fitted parameters
        - 'peaks': peak indices
        - 'peak_values': values at peaks
        - 'fitted_curve': fitted function
        - 'r_squared': R² goodness of fit
        - 'x_fitted': x values for fitted curve
        - 'y_fitted': y values for fitted curve
    """
    drag_array = np.asarray(drag_signal)
    
    if peaks is None:
        result = find_drag_peaks_and_troughs(drag_array, prominence=prominence, distance=distance)
        peaks = result['peaks']
        peak_values = result['peak_values']
    else:
        peak_values = drag_array[peaks]
    
    if len(peaks) < 2:
        print("Warning: Need at least 2 peaks to fit exponential")
        return None
    
    # Create x and y data for fitting
    x_data = np.arange(len(peaks))
    y_data = peak_values
    
    try:
        if fit_type == 'decay':
            # Initial guess for exponential decay
            popt, _ = curve_fit(exponential_decay, x_data, y_data, 
                              p0=[y_data[0], 0.1], maxfev=5000)
            fitted_func = exponential_decay
        else:
            # Initial guess for exponential growth
            popt, _ = curve_fit(exponential_growth, x_data, y_data,
                              p0=[y_data[0], 0.1], maxfev=5000)
            fitted_func = exponential_growth
        
        # Calculate R²
        y_pred = fitted_func(x_data, *popt)
        r_squared = r2_score(y_data, y_pred)
        
        # Generate fitted curve
        x_fitted = np.linspace(0, len(peaks) - 1, 200)
        y_fitted = fitted_func(x_fitted, *popt)
        
        return {
            'params': popt,
            'peaks': peaks,
            'peak_values': peak_values,
            'fitted_func': fitted_func,
            'r_squared': r_squared,
            'x_fitted': x_fitted,
            'y_fitted': y_fitted,
            'fit_type': fit_type
        }
    except Exception as e:
        print(f"Error fitting exponential: {e}")
        return None


def fit_exponential_to_troughs(drag_signal, troughs=None, prominence=None, distance=None, fit_type='decay'):
    """
    Fit exponential curve to drag troughs.
    
    Parameters:
    -----------
    drag_signal : array-like
        The drag force data
    troughs : array-like, optional
        Trough indices; if None, will find them
    prominence : float, optional
        Minimum prominence for peak detection
    distance : int, optional
        Minimum distance between troughs
    fit_type : str
        'decay' or 'growth' for exponential type
    
    Returns:
    --------
    dict with fitted results (same structure as fit_exponential_to_peaks)
    """
    drag_array = np.asarray(drag_signal)
    
    if troughs is None:
        result = find_drag_peaks_and_troughs(drag_array, prominence=prominence, distance=distance)
        troughs = result['troughs']
        trough_values = result['trough_values']
    else:
        trough_values = drag_array[troughs]
    
    if len(troughs) < 2:
        print("Warning: Need at least 2 troughs to fit exponential")
        return None
    
    # Create x and y data for fitting
    x_data = np.arange(len(troughs))
    y_data = trough_values
    
    try:
        if fit_type == 'decay':
            popt, _ = curve_fit(exponential_decay, x_data, y_data,
                              p0=[y_data[0], 0.1], maxfev=5000)
            fitted_func = exponential_decay
        else:
            popt, _ = curve_fit(exponential_growth, x_data, y_data,
                              p0=[y_data[0], 0.1], maxfev=5000)
            fitted_func = exponential_growth
        
        # Calculate R²
        y_pred = fitted_func(x_data, *popt)
        r_squared = r2_score(y_data, y_pred)
        
        # Generate fitted curve
        x_fitted = np.linspace(0, len(troughs) - 1, 200)
        y_fitted = fitted_func(x_fitted, *popt)
        
        return {
            'params': popt,
            'troughs': troughs,
            'trough_values': trough_values,
            'fitted_func': fitted_func,
            'r_squared': r_squared,
            'x_fitted': x_fitted,
            'y_fitted': y_fitted,
            'fit_type': fit_type
        }
    except Exception as e:
        print(f"Error fitting exponential: {e}")
        return None


def plot_drag_analysis(drag_signal, peak_fit=None, trough_fit=None, prominence=None, distance=None):
    """
    Plot drag signal with peaks, troughs, and fitted exponentials.
    
    Parameters:
    -----------
    drag_signal : array-like
        The drag force data
    peak_fit : dict, optional
        Result from fit_exponential_to_peaks()
    trough_fit : dict, optional
        Result from fit_exponential_to_troughs()
    prominence : float, optional
        Prominence for peak detection
    distance : int, optional
        Distance for peak detection
    """
    drag_array = np.asarray(drag_signal)
    time_axis = np.arange(len(drag_array))
    
    # Find peaks and troughs if not provided
    if peak_fit is None or trough_fit is None:
        result = find_drag_peaks_and_troughs(drag_array, prominence=prominence, distance=distance)
        peaks = result['peaks']
        troughs = result['troughs']
    
    if peak_fit is not None:
        peaks = peak_fit['peaks']
    if trough_fit is not None:
        troughs = trough_fit['troughs']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Time domain with peaks and troughs
    ax1.plot(time_axis, drag_array, 'b-', label='Drag Signal', linewidth=1.5)
    if peak_fit is not None:
        ax1.plot(peak_fit['peaks'], peak_fit['peak_values'], 'go', markersize=8, label='Peaks')
    if trough_fit is not None:
        ax1.plot(trough_fit['troughs'], trough_fit['trough_values'], 'ro', markersize=8, label='Troughs')
    else:
        ax1.plot(troughs, drag_array[troughs], 'ro', markersize=8, label='Troughs')
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Drag Force')
    ax1.set_title('Drag Signal with Peaks and Troughs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Exponential fits
    if peak_fit is not None:
        ax2.plot(peak_fit['x_fitted'], peak_fit['y_fitted'], 'g--', linewidth=2,
                label=f'Peak Fit: $y = a \cdot e^{{-bx}}$ (R² = {peak_fit["r_squared"]:.3f})')
        ax2.plot(np.arange(len(peak_fit['peak_values'])), peak_fit['peak_values'], 'go', markersize=8)
    
    if trough_fit is not None:
        ax2.plot(trough_fit['x_fitted'], trough_fit['y_fitted'], 'r--', linewidth=2,
                label=f'Trough Fit: $y = a \cdot e^{{-bx}}$ (R² = {trough_fit["r_squared"]:.3f})')
        ax2.plot(np.arange(len(trough_fit['trough_values'])), trough_fit['trough_values'], 'ro', markersize=8)
    
    ax2.set_xlabel('Peak/Trough Index')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Exponential Fits to Peaks and Troughs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_fitted_exponentials_only(peak_fit=None, trough_fit=None):
    """
    Create a detailed plot of fitted exponentials for peaks and troughs.
    
    Parameters:
    -----------
    peak_fit : dict, optional
        Result from fit_exponential_to_peaks()
    trough_fit : dict, optional
        Result from fit_exponential_to_troughs()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Peak exponential fit
    if peak_fit is not None:
        ax = axes[0]
        x_data = np.arange(len(peak_fit['peak_values']))
        ax.scatter(x_data, peak_fit['peak_values'], color='green', s=100, alpha=0.7, 
                  label='Peak Amplitudes', edgecolors='darkgreen', linewidth=2, zorder=3)
        ax.plot(peak_fit['x_fitted'], peak_fit['y_fitted'], 'g--', linewidth=3, 
               label=f"Fit: $y = {peak_fit['params'][0]:.4f} \cdot e^{{-{peak_fit['params'][1]:.4f} \cdot x}}$", zorder=2)
        ax.fill_between(peak_fit['x_fitted'], peak_fit['y_fitted'], alpha=0.2, color='green')
        ax.set_xlabel('Peak Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Peak Amplitude', fontsize=12, fontweight='bold')
        ax.set_title(f"Peak Exponential Fit (R² = {peak_fit['r_squared']:.4f})", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Trough exponential fit
    if trough_fit is not None:
        ax = axes[1]
        x_data = np.arange(len(trough_fit['trough_values']))
        ax.scatter(x_data, trough_fit['trough_values'], color='red', s=100, alpha=0.7,
                  label='Trough Amplitudes', edgecolors='darkred', linewidth=2, zorder=3)
        ax.plot(trough_fit['x_fitted'], trough_fit['y_fitted'], 'r--', linewidth=3,
               label=f"Fit: $y = {trough_fit['params'][0]:.4f} \cdot e^{{-{trough_fit['params'][1]:.4f} \cdot x}}$", zorder=2)
        ax.fill_between(trough_fit['x_fitted'], trough_fit['y_fitted'], alpha=0.2, color='red')
        ax.set_xlabel('Trough Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Trough Amplitude', fontsize=12, fontweight='bold')
        ax.set_title(f"Trough Exponential Fit (R² = {trough_fit['r_squared']:.4f})", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig, axes


if __name__ == "__main__":
    # Initialize the simulation grid
    grid = torch.zeros((W, H,2), device=device)
    # Generate true spatial coordinates for every cell center
    x_space = torch.arange(W, device=device, dtype=torch.float32)
    y_space = torch.arange(H, device=device, dtype=torch.float32)
    spatial_X, spatial_Y = torch.meshgrid(x_space, y_space, indexing='ij')
    spatial_grid = torch.stack((spatial_X, spatial_Y), dim=-1).to(device) # Shape: (W, H, 2)

    colours = torch.zeros((W,H,3),device=device)
    L_in_colours = torch.rand((H,3),device=device)
    B_in_colours = torch.rand((W,3),device=device)
    #L_in_colours = 0.5 + torch.sin(torch.linspace(0, 2 * np.pi, H, device=device)).unsqueeze(-1) * 0.5
    #B_in_colours = 0.5 + torch.cos(torch.linspace(0, 2 * np.pi, W, device=device)).unsqueeze(-1) * 0.5

    # Set initial conditions (e.g., a small perturbation in the center)
    grid[W//2, H//2] = 1.0
    points = torch.zeros((point_count,2),device=device)
    points = torch.rand_like(points)*torch.tensor([W-1,H-1],device=device)
    #points[:,1] = torch.arrange(0,point_count)*H/point_count
    def advect_(grid, Dt, extra_grids=[]):
        # 1. Backtrack using real spatial coordinates
        back_points = spatial_grid - grid * (Dt / Dx)
        
        # 2. Strictly clamp coordinates within safe interpolation boundaries
        torch.clamp_(back_points[..., 0], 0, W - 1 - 1e-5)
        torch.clamp_(back_points[..., 1], 0, H - 1 - 1e-5)
        
        int_indexes = torch.floor(back_points).to(torch.long)
        d = back_points - int_indexes
        dx = d[..., 0].unsqueeze(-1)
        dy = d[..., 1].unsqueeze(-1)
        
        x = int_indexes[..., 0]
        y = int_indexes[..., 1]
        
        # 3. Sample neighbors safely
        v00 = grid[x, y]
        v10 = grid[x + 1, y]
        v01 = grid[x, y + 1]
        v11 = grid[x + 1, y + 1]
        
        weights = ((1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy)

        # 4. Bilinear Interpolation
        velocities = (
            weights[0] * v00 +
            weights[1] * v10 +
            weights[2] * v01 +
            weights[3] * v11
        )

        updated_grids = []
        for gridi in extra_grids:
            g00 = gridi[x, y]
            g01 = gridi[x, y + 1]
            g10 = gridi[x + 1, y]
            g11 = gridi[x + 1, y + 1]

            new_grid = (
            weights[0] * g00 +
            weights[1] * g10 +
            weights[2] * g01 +
            weights[3] * g11
            )
            updated_grids.append(new_grid)

        return velocities, updated_grids # Return the freshly assigned velocity map and updated grids


    def update_points_(grid, points, Dt=Dt, Dx=Dx):
        torch.clamp_(points[:, 0], 0, W - 1 - 1e-5)
        torch.clamp_(points[:, 1], 0, H - 1 - 1e-5)
        
        int_indexes = torch.floor(points).to(torch.long)
        d = points - int_indexes
        dx = d[:, 0]
        dy = d[:, 1]
        x = int_indexes[:, 0]
        y = int_indexes[:, 1]
        
        # Corrected PyTorch indexing: grid[x, y] instead of grid[(x, y)]
        v00 = grid[x, y]
        v10 = grid[x + 1, y]
        v01 = grid[x, y + 1]
        v11 = grid[x + 1, y + 1]
        
        # Bilinear interpolation
        velocities = (
            ((1 - dx) * (1 - dy)).unsqueeze(-1) * v00 +
            (dx * (1 - dy)).unsqueeze(-1) * v10 +
            ((1 - dx) * dy).unsqueeze(-1) * v01 +
            (dx * dy).unsqueeze(-1) * v11
        )
        
        points += velocities * Dt * particle_slower  # Update positions based on interpolated velocities
        
        # Corrected boundary handling (clamp/wrap logic)
        x_over = points[:, 0] > W - 2
        x_under = points[:, 0] < 1
        y_over  = points[:, 1] > H - 2
        y_under = points[:, 1] < 1

        points[x_over, 0] = 1
        points[x_under, 0] = W-2
        points[y_over, 1] = 1
        points[y_under, 1] = H-2
        points[x_over | x_under, 1] = torch.rand_like(points[x_over | x_under, 1])*H
        points[y_over | y_under, 0] = torch.rand_like(points[y_over | y_under, 0])*W

    
    viscous_mult = torch.exp(-viscosity * k_grid_mag_squared * Dt/density)
    pre_vort = 0
    total_vort = 0
    vort_count=0
    drag_record = []
    OW_mask = torch.zeros_like(grid[..., 0], dtype=torch.bool, device=device)
    divergence_sum_history = []
    def update_grid(grid, Dt=Dt, Dx=Dx):
        global OW_mask
        # 1. Perform FFT explicitly on the spatial dimensions (0, 1)
        grid[0, :, 0] = transience[0]
        grid[-1, :, 0] = 0
        grid[0, :, 1] = transience[1]
        grid[-1, :, 1] = 0
        
        grid[cuboid_L_corner[0],cuboid_L_corner[1]:cuboid_R_corner[1]] = -grid[cuboid_L_corner[0]-1,cuboid_L_corner[1]:cuboid_R_corner[1]]
        grid[cuboid_R_corner[0],cuboid_L_corner[1]:cuboid_R_corner[1]] = -grid[cuboid_R_corner[0]+1,cuboid_L_corner[1]:cuboid_R_corner[1]]
        
        grid[cuboid_L_corner[0]:cuboid_R_corner[0],cuboid_L_corner[1]] = -grid[cuboid_L_corner[0]:cuboid_R_corner[0],cuboid_L_corner[1]-1]
        grid[cuboid_L_corner[0]:cuboid_R_corner[0],cuboid_R_corner[1]] = -grid[cuboid_L_corner[0]:cuboid_R_corner[0],cuboid_R_corner[1]+1]

        grid, updated_grids = advect_(grid, Dt,[colours])
        if updated_grids:
            colours[:] = updated_grids[0]

        grid[cuboid_L_corner[0]-1:cuboid_R_corner[0]+1, cuboid_L_corner[1]:cuboid_R_corner[1], 0] = 0
        grid[cuboid_L_corner[0]:cuboid_R_corner[0], cuboid_L_corner[1]-1:cuboid_R_corner[1]+1, 1] = 0

        grid_freq = torch.fft.fft2(grid, dim=(0, 1))
        
        grid_freq *= viscous_mult

        # 2. Calculate |k|^2 
        global k_sq
        
        # 3. Correct Helmholtz/Leray Projection
        divergence_freq = torch.sum(grid_freq * k_grid, dim=-1, keepdim=True)
        pressure_freq = divergence_freq / k_sq
        pressure = torch.fft.ifft2(pressure_freq, dim=(0, 1)).real
        divergence = torch.fft.ifft2(divergence_freq, dim=(0, 1)).real
        divergence_sum = torch.sum(torch.abs(divergence))
        divergence_sum_history.append(divergence_sum)

        #drag_x = torch.sum(divergence * grid[cuboid_L_corner[0],cuboid_L_corner[1]:cuboid_R_corner[1], 0]) - torch.sum(divergence * grid[cuboid_R_corner[0],cuboid_L_corner[1]:cuboid_R_corner[1], 0])/Dt*Dx
        #drag_y = torch.sum(divergence * grid[cuboid_L_corner[0]:cuboid_R_corner[0], cuboid_L_corner[1], 1]) - torch.sum(divergence * grid[cuboid_L_corner[0]:cuboid_R_corner[0], cuboid_R_corner[1], 1])/Dt*Dx

        grid_freq -= (divergence_freq / k_sq) * k_grid

        # 4. Advection in frequency domain (Note: ensure logic matches physics expectations)
        #grid_freq *= torch.exp(-1j * k_grid * Dt / (density * Dx))
          # Apply viscosity in frequency domain
        # Inverse FFT back to spatial domain
        
        # Extract frequency representations
        u_freq = grid_freq[..., 0]
        v_freq = grid_freq[..., 1]

        kx = k_grid[..., 0]
        ky = k_grid[..., 1]

        # Compute spatial derivatives by taking the Inverse FFT of (1j * k * freq)
        u_x = torch.fft.ifft2(1j * kx * u_freq).real
        u_y = torch.fft.ifft2(1j * ky * u_freq).real
        v_x = torch.fft.ifft2(1j * kx * v_freq).real

        # Calculate the Okubo-Weiss field
        W_param = 8 * (u_x**2 + u_y * v_x)
        # Create a binary map where 1 represents a vortex cell
        vortex_threshold = -transience[0]**2*2/Dx  # Adjust based on your flow velocity
        vortex_mask = W_param < vortex_threshold
        OW_mask = vortex_mask

        import scipy.ndimage as ndimage

        # Move mask to CPU and convert to a numpy boolean array
        mask_np = vortex_mask.cpu().numpy()

        # Label connected islands of pixels
        labeled_array, num_vortices = ndimage.label(mask_np)

        print(f"Current independent vortices on screen: {num_vortices}")

        global pre_vort, total_vort, vort_count
        vort_count += num_vortices
        if num_vortices > pre_vort:
            #print("New vortex detected!")
            total_vort += num_vortices - pre_vort
        pre_vort = num_vortices

        new_grid = torch.fft.ifft2(grid_freq, dim=(0, 1)).real
        
        # Enforce boundaries / obstacles
        #drag_x = -Dx**3 * torch.sum(new_grid[cuboid_L_corner[0]-1:cuboid_R_corner[0]+1, cuboid_L_corner[1]:cuboid_R_corner[1], 0] )
        #drag_y = -Dx**3 * torch.sum(new_grid[cuboid_L_corner[0]:cuboid_R_corner[0], cuboid_L_corner[1]-1:cuboid_R_corner[1]+1, 1] )
        #drag = (torch.sum(new_grid[0]-new_grid[-1],dim=(0,1))+torch.sum(new_grid[:,0]-new_grid[:,-1],dim=(0,1)))*transience
        #drag_x = (torch.sum((new_grid[0,:,0]**2-new_grid[-1,:,0]**2)) + torch.sum(new_grid[:,0,0]*new_grid[:,0,1]-new_grid[:,-1,0]*new_grid[:,-1,1]))*Dx
        #drag_y = (torch.sum((new_grid[:,0,1]**2-new_grid[:, -1, 1]**2)) + torch.sum(new_grid[0,:,1]*new_grid[0,:,0]-new_grid[-1,:,1]*new_grid[-1,:,0]))*Dx
        left_pressure = pressure[cuboid_L_corner[0]-1, cuboid_L_corner[1]:cuboid_R_corner[1], 0]
        right_pressure = pressure[cuboid_R_corner[0]+1, cuboid_L_corner[1]:cuboid_R_corner[1], 0]
        top_pressure = pressure[cuboid_L_corner[0]:cuboid_R_corner[0], cuboid_L_corner[1]-1, 0]
        bottom_pressure = pressure[cuboid_L_corner[0]:cuboid_R_corner[0], cuboid_R_corner[1]+1, 0]

        drag_x = torch.sum(left_pressure - right_pressure) * Dx
        drag_y = torch.sum(top_pressure - bottom_pressure) * Dx


        #print("drag:", drag_x.item(), drag_y.item())
        new_grid[cuboid_L_corner[0]-1:cuboid_R_corner[0]+1, cuboid_L_corner[1]:cuboid_R_corner[1], 0] = 0
        new_grid[cuboid_L_corner[0]:cuboid_R_corner[0], cuboid_L_corner[1]-1:cuboid_R_corner[1]+1, 1] = 0
        drag_record.append([drag_x,drag_y])


        # 5. Apply viscosity and transience
        new_grid[0, :, 0] = transience[0]
        new_grid[-1, :, 0] = 0
        new_grid[0, :, 1] = transience[1]
        new_grid[-1, :, 1] = 0

        mean = torch.mean(new_grid, dim=(0, 1))
        #new_grid = new_grid * viscosity + mean * (1 - viscosity)
        new_grid += transience - mean
        
        return new_grid
    #grid [:,20:30,0]=1
    #grid [:,30:40,0]=-1

    colour_canvas, colour_visual = render.render_colours(colours, window_size=(W*2, H*2))

    tick =1
    it = time()
    lines = None
    vector_canvas = None
    sim_time = 40
    for _ in range(int(floor(sim_time/Dt))):
        grid = update_grid(grid)
        colours[0] = L_in_colours
        colours[:,0] = B_in_colours
        update_points_(grid,points)
        if tick % (render_Dt // Dt) == 0:
            print(colours[1,1])
            
            print("mean update time:", (time()-it)*1000/(render_Dt // Dt), "ms")
            it2 = time()
            if lines is None:
                vector_canvas, lines = render.render_vector_grid_2D(grid.transpose(0,1)*visual_scaling, canvas=vector_canvas)
                markers = visuals.Markers(parent=lines.parent)
            else:
                render.update_vector_grid_2D(lines, (grid).transpose(0,1)*visual_scaling)
            markers.set_data((points).cpu().numpy(), face_color='cyan', size=8)
            
            vispy.app.process_events()
            # Update colours visualization
            render.update_colours(colour_visual, colours, colour_canvas)
                        
            print("render time:",(time()-it2)*1000,"ms")
            print(f"Total vortices detected: {total_vort}")
            print("mean vortices:",vort_count/tick)
            it = time()
        tick += 1

        
    # Plot drag_record
    if drag_record:
        drag_torch = torch.tensor(drag_record[int(1/Dt):], device=device)
        drag_freq = torch.fft.fft(drag_torch[...,0]-drag_torch[...,1], dim=0)
        drag_freq_peaks = torch.argsort(torch.abs(drag_freq), dim=0, descending=True)[:10].to(float)
        drag_freq_peaks /= len(drag_record) * sim_time  # Convert to frequency in Hz
        drag_freq_peaks = 1/drag_freq_peaks
        print("Frequency peaks:")
        print(drag_freq_peaks)
        
        # Clustering on frequency peaks
        drag_freq_peaks[torch.isinf(drag_freq_peaks)] = 0  # Replace inf with 0 for clustering
        drag_freq_peaks[torch.isnan(drag_freq_peaks)] = 0  # Replace inf with 0 for clustering

        peaks_np = drag_freq_peaks.cpu().numpy().reshape(-1, 1) if isinstance(drag_freq_peaks, torch.Tensor) else drag_freq_peaks.reshape(-1, 1)
        
        # Filter out zeros
        peaks_filtered = peaks_np[peaks_np.flatten() != 0]
        
        if len(peaks_filtered) > 0:
            # Remove anomalies using IQR method
            Q1 = np.percentile(peaks_filtered, 25)
            Q3 = np.percentile(peaks_filtered, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            peaks_no_anomalies = peaks_filtered[(peaks_filtered >= lower_bound) & (peaks_filtered <= upper_bound)]
            
            print(f"Total peaks: {len(peaks_np)}, After removing zeros: {len(peaks_filtered)}, After removing anomalies: {len(peaks_no_anomalies)}")
            print(f"IQR bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
            
            if len(peaks_no_anomalies) > 0:
                # Cluster the filtered peaks
                n_clusters = min(3, len(peaks_no_anomalies))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(peaks_no_anomalies.reshape(-1, 1))
                cluster_centers = kmeans.cluster_centers_
                
                print(f"\nClustering Results (n_clusters={n_clusters}):")
                for i in range(n_clusters):
                    cluster_peaks = peaks_no_anomalies[cluster_labels == i].flatten()
                    print(f"  Cluster {i}: center={cluster_centers[i][0]:.6f}, peaks={cluster_peaks}")
            else:
                print("No valid peaks after removing anomalies")
        else:
            print("All peaks were zeros")
        drag_x_data = np.array([d[0].item() if hasattr(d[0], 'item') else d[0] for d in drag_record])
        drag_y_data = np.array([d[1].item() if hasattr(d[1], 'item') else d[1] for d in drag_record])
        time_axis = np.arange(len(drag_record)) * Dt
        
        # Fourier transform of drag_x
        drag_x_centered = drag_y_data - np.mean(drag_y_data)
        N = len(drag_x_centered)
        if N > 1:
            fft_drag_x = np.fft.rfft(drag_x_centered)
            freq_axis = np.fft.rfftfreq(N, d=Dt)
            power_spectrum = np.abs(fft_drag_x)

            plt.figure(figsize=(10, 5))
            plt.plot(freq_axis, power_spectrum, '-o', markersize=4)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title('Fourier Transform of drag_x')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Filter data for time > 1
        mask = time_axis > 1
        time_filtered = time_axis[mask]
        drag_x_filtered = drag_x_data[mask]
        
        # Define fitting function: sine + cosine + constant
        def sine_cosine_const(t, sine_amp, cos_amp, frequency, offset):
            return sine_amp * np.sin(frequency * t) + cos_amp * np.cos(frequency * t) + offset
        
        # Fit the curve
        try:
            popt, pcov = curve_fit(sine_cosine_const, time_filtered, drag_x_filtered, 
                                   p0=[1, 1, 2*np.pi, 0], maxfev=5000)
            sine_amp, cos_amp, frequency, offset = popt
            print(f"\nFitted sine + cosine + constant (for t > 1):")
            print(f"  Sine amplitude: {sine_amp:.6f}")
            print(f"  Cosine amplitude: {cos_amp:.6f}")
            print(f"  Frequency: {frequency:.6f}")
            print(f"  Offset: {offset:.6f}")
            
            # Generate fitted curve
            drag_x_fitted = sine_cosine_const(time_filtered, *popt)
        except Exception as e:
            print(f"Curve fitting failed: {e}")
            drag_x_fitted = None
        
        # Plot drag and divergence with dual y-axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot drag on left y-axis
        ax1.plot(time_axis, drag_x_data, 'o-', label='drag_x', linewidth=2, markersize=4, color='tab:blue')
        ax1.plot(time_axis, drag_y_data, 's-', label='drag_y', linewidth=2, markersize=4, color='tab:orange')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Drag', fontsize=12, color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.axvline(x=1, color='gray', linestyle=':', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Plot divergence on right y-axis if available
        if divergence_sum_history:
            divergence_data = np.array([d.item()/W/H/Dx**2 if hasattr(d, 'item') else d for d in divergence_sum_history])
            divergence_time = np.arange(len(divergence_sum_history)) * Dt
            
            ax2 = ax1.twinx()
            ax2.plot(divergence_time, divergence_data, 'g-', linewidth=2, label='average pressure difference from mean', alpha=0.7)
            ax2.set_ylabel('average pressure difference from mean', fontsize=12, color='g')
            ax2.tick_params(axis='y', labelcolor='g')
        
        plt.title('Drag Force and Pressure Over Time', fontsize=14)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if divergence_sum_history:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            ax1.legend(lines1, labels1, loc='upper left')
        
        plt.tight_layout()
        plt.show()
        print("mean divergence drift:",np.sum(divergence_data)/len(divergence_sum_history))
        """
        # ============================================================
        # NEW: Analyze drag peaks and fit exponentials
        # ============================================================
        print("\n" + "="*60)
        print("DRAG PEAK AND TROUGH ANALYSIS")
        print("="*60)
        
        # Extract drag signal (use difference or magnitude)
        drag_magnitude = np.sqrt(drag_x_data**2 + drag_y_data**2)
        
        # Find peaks and troughs
        peak_trough_data = find_drag_peaks_and_troughs(
            drag_magnitude, 
            prominence=np.std(drag_magnitude) * 0.5,  # Adjust sensitivity
            distance=5  # Minimum samples between peaks
        )
        
        print(f"\nFound {len(peak_trough_data['peaks'])} peaks and {len(peak_trough_data['troughs'])} troughs")
        print(f"Peak values: min={peak_trough_data['peak_values'].min():.6f}, max={peak_trough_data['peak_values'].max():.6f}")
        print(f"Trough values: min={peak_trough_data['trough_values'].min():.6f}, max={peak_trough_data['trough_values'].max():.6f}")
        
        # Fit exponential to peaks
        peak_fit = fit_exponential_to_peaks(
            drag_magnitude,
            prominence=np.std(drag_magnitude) * 0.5,
            distance=5,
            fit_type='decay'  # Use 'growth' if peaks are increasing
        )
        
        if peak_fit is not None:
            print(f"\nPeak Fit Results (Exponential Decay: y = a * exp(-b*x)):")
            print(f"  Parameters: a={peak_fit['params'][0]:.6f}, b={peak_fit['params'][1]:.6f}")
            print(f"  R² score: {peak_fit['r_squared']:.6f}")
        
        # Fit exponential to troughs
        trough_fit = fit_exponential_to_troughs(
            drag_magnitude,
            prominence=np.std(drag_magnitude) * 0.5,
            distance=5,
            fit_type='decay'  # Use 'growth' if troughs are increasing
        )
        
        if trough_fit is not None:
            print(f"\nTrough Fit Results (Exponential Decay: y = a * exp(-b*x)):")
            print(f"  Parameters: a={trough_fit['params'][0]:.6f}, b={trough_fit['params'][1]:.6f}")
            print(f"  R² score: {trough_fit['r_squared']:.6f}")
        
        # Plot analysis
        fig, axes = plot_drag_analysis(
            drag_magnitude,
            peak_fit=peak_fit,
            trough_fit=trough_fit,
            prominence=np.std(drag_magnitude) * 0.5,
            distance=5
        )
        
        plt.show()
        
        # Plot fitted exponentials only (detailed view)
        fig_exp, axes_exp = plot_fitted_exponentials_only(
            peak_fit=peak_fit,
            trough_fit=trough_fit
        )
        
        plt.show()
        print("\n" + "="*60)"""
