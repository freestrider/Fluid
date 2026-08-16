import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from matplotlib.pylab import block
import torch
from standard import *
from vispy import scene, app
import numpy as np

VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
void main() {
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix * vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f); // Orange color
}
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



@timer
def render_volume_threshold(scalar_field, scalar_bound=0.5, canvas=None,color='viridis'):
    # 1. GPU Pre-processing (Normalization)
    if not isinstance(scalar_field, torch.Tensor):
        scalar_field = torch.from_numpy(np.asarray(scalar_field))
    
    field_clipped = torch.clamp(scalar_field - scalar_bound, min=0.0)
    f_max = field_clipped.max()
    norm_gpu = field_clipped / f_max if f_max > 0 else field_clipped
    
    # Transfer to CPU for VisPy
    data_np = norm_gpu.cpu().numpy().astype(np.float32)

    # 2. Handle Canvas and View Logic
    if canvas is None:
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True, bgcolor='black')
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
    else:
        # Retrieve existing view if it exists
        if len(canvas.central_widget.children) > 0:
            view = canvas.central_widget.children[0]
        else:
            view = canvas.central_widget.add_view()
            view.camera = 'turntable'

    # 3. Create Volume Visual
    # Added 'parent=view.scene' to ensure it attaches to the existing coordinate system
    volume = scene.visuals.Volume(data_np, parent=view.scene, method='mip',cmap=color)
    
    # 4. Camera Setup
    # Calculate center based on actual data shape
    shape = data_np.shape
    view.camera.center = (shape[2] / 2, shape[1] / 2, shape[0] / 2)
    view.camera.set_range() 
    
    return canvas, volume
@timer
def render_vector_grid(vector_field, spacing=2, length_scale=1.0, canvas=None, color=(0.2, 0.7, 1.0, 0.8),order=1):
    # 1. Downsample and Process Tensors (GPU-side)
    subsampled = vector_field[::spacing, ::spacing, :]
    w, h, _ = subsampled.shape
        
    
    def get_k_grid(w, h, device):
        # Generates frequencies for a 2D grid
        kx = torch.fft.fftfreq(w, d=1).to(device) * 2 * torch.pi
        ky = torch.fft.fftfreq(h, d=1).to(device) * 2 * torch.pi
        kx_g, ky_g = torch.meshgrid(kx, ky, indexing='ij')
        return torch.stack((kx_g, ky_g), dim=-1)  # Shape: (W, H, 2)

    k_grid = get_k_grid(w, h, device)

    x = k_grid[...,0]
    y = k_grid[...,1]
    p0 = torch.stack([x,y], dim=-1).reshape(-1, 3) * spacing
    p1 = p0 + (subsampled.reshape(-1, 3) * length_scale)
    
    # Interleave starts and ends for 'segments' mode
    line_coords = torch.stack([p0, p1], dim=1)
    vertices = line_coords.cpu().numpy().astype(np.float32)

    # 2. Handle Canvas and View Logic
    if canvas is None:
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 800), show=True, bgcolor='black')
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
    else:
        # Retrieve the existing view from the canvas
        # If the canvas was created with central_widget.add_view(), it's usually the first child
        if len(canvas.central_widget.children) > 0:
            view = canvas.central_widget.children[0]
        else:
            view = canvas.central_widget.add_view()
            view.camera = 'turntable'

    # 3. Add the Visual to the existing Scene
    lines = scene.visuals.Line(pos=vertices, connect='segments', 
                               color=color, parent=view.scene)
    lines.order = order

    # Re-adjust camera to fit both old and new data
    view.camera.set_range()
    
    return canvas, lines

@timer
def render_vector_grid_2D(vector_field, spacing=2, length_scale=1.0, canvas=None, color=(0.2, 0.7, 1.0, 0.8), order=1):
    """
    Render a 2D vector field as a grid of arrows/lines.
    
    Args:
        vector_field: 2D tensor of shape (height, width, 2) representing [x, y] components
        spacing: Downsample factor (render every nth vector)
        length_scale: Scale factor for arrow lengths
        canvas: Optional existing VisPy canvas
        color: RGBA color tuple
        order: Rendering order
    
    Returns:
        VisPy canvas with the rendered 2D vector grid
    """
    # 1. Downsample and Process Tensors (GPU-side)
    subsampled = vector_field[::spacing, ::spacing, :]
    w, h, _ = subsampled.shape
    
    # Create grid coordinates (y, x)
    x, y = torch.meshgrid(
        torch.arange(0, h, device=vector_field.device),
        torch.arange(0, w, device=vector_field.device),
        indexing='ij'
    )
    
    # Starting points (scale by spacing to match original coordinates)
    p0 = torch.stack([x, y], dim=-1).reshape(-1, 2) * spacing
    
    # Ending points (add scaled vector components)
    p1 = p0 + (subsampled.reshape(-1, 2) * length_scale)
    
    # Pad with z=0 for 3D rendering
    z_pad = torch.zeros((p0.shape[0], 1), device=p0.device)
    p0_3d = torch.cat([p0, z_pad], dim=1)
    p1_3d = torch.cat([p1, z_pad], dim=1)
    
    # Interleave starts and ends for 'segments' mode
    line_coords = torch.stack([p0_3d, p1_3d], dim=1)
    vertices = line_coords.cpu().numpy().astype(np.float32)

    # 2. Handle Canvas and View Logic
    if canvas is None:
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 800), show=True, bgcolor='black')
        view = canvas.central_widget.add_view()
        view.camera = 'panzoom'
    else:
        # Retrieve the existing view from the canvas
        if len(canvas.central_widget.children) > 0:
            view = canvas.central_widget.children[0]
        else:
            view = canvas.central_widget.add_view()
            view.camera = 'panzoom'

    # 3. Add the Visual to the existing Scene
    lines = scene.visuals.Line(pos=vertices, connect='segments', 
                               color=color, parent=view.scene)
    lines.order = order

    # Re-adjust camera to fit both old and new data
    view.camera.set_range()
    
    return canvas, lines

@timer
def update_vector_grid_2D(lines_visual, vector_field, spacing=2, length_scale=1.0):
    """
    Update existing 2D vector grid visual with new data.
    
    Args:
        lines_visual: The Line visual object to update
        vector_field: 2D tensor of shape (height, width, 2) with new data
        spacing: Downsample factor
        length_scale: Scale factor for arrow lengths
    """
    subsampled = vector_field[::spacing, ::spacing, :]
    h, w, _ = subsampled.shape
    
    y, x = torch.meshgrid(
        torch.arange(0, h, device=vector_field.device),
        torch.arange(0, w, device=vector_field.device),
        indexing='ij'
    )
    
    p0 = torch.stack([x, y], dim=-1).reshape(-1, 2) * spacing
    p1 = p0 + (subsampled.reshape(-1, 2) * length_scale)
    
    # Pad with z=0 for 3D rendering
    z_pad = torch.zeros((p0.shape[0], 1), device=p0.device)
    p0_3d = torch.cat([p0, z_pad], dim=1)
    p1_3d = torch.cat([p1, z_pad], dim=1)
    
    # Interleave starts and ends for 'segments' mode
    line_coords = torch.stack([p0_3d, p1_3d], dim=1).reshape(-1, 3)
    vertices = line_coords.cpu().numpy().astype(np.float32)
    
    lines_visual.set_data(pos=vertices, connect='segments')

@timer
def update_vector_grid(lines_visual, vector_field, spacing=2, length_scale=1.0):
    """
    Update existing 3D vector grid visual with new data.
    
    Args:
        lines_visual: The Line visual object to update
        vector_field: 3D tensor of shape (h, w, d, 3) with new data
        spacing: Downsample factor
        length_scale: Scale factor for arrow lengths
    """
    subsampled = vector_field[::spacing, ::spacing, ::spacing, :]
    h, w, d, _ = subsampled.shape
    
    z, y, x = torch.meshgrid(
        torch.arange(0, h, device=vector_field.device),
        torch.arange(0, w, device=vector_field.device),
        torch.arange(0, d, device=vector_field.device),
        indexing='ij'
    )
    
    p0 = torch.stack([z, y, x], dim=-1).reshape(-1, 3) * spacing
    p1 = p0 + (subsampled.reshape(-1, 3) * length_scale)
    
    line_coords = torch.stack([p0, p1], dim=1).reshape(-1, 3)
    vertices = line_coords.cpu().numpy().astype(np.float32)
    
    lines_visual.set_data(pos=vertices, connect='segments')

@timer
def render_scalar_grid(scalar_field, canvas=None, cmap='viridis',order=1,volume="flat"):
    """
    Render a tensor of scalars as a VisPy volume.
    
    Args:
        scalar_field: 3D tensor of scalar values
        canvas: Optional existing VisPy canvas
        cmap: Colormap name (default: 'viridis')
    
    Returns:
        VisPy canvas with the rendered volume
    """
    # 1. GPU Pre-processing (Normalization)
    if not isinstance(scalar_field, torch.Tensor):
        scalar_field = torch.from_numpy(np.asarray(scalar_field))
    
    # Normalize the scalar field to [0, 1] range
    f_min = scalar_field.min()
    f_max = scalar_field.max()
    norm_gpu = (scalar_field - f_min) / (f_max - f_min) if f_max > f_min else scalar_field - f_min
    
    # Transfer to CPU for VisPy
    data_np = norm_gpu.cpu().numpy().astype(np.float32)

    # 2. Handle Canvas and View Logic
    if canvas is None:
        canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True, bgcolor='black')
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
    else:
        # Retrieve existing view if it exists
        if len(canvas.central_widget.children) > 0:
            view = canvas.central_widget.children[0]
        else:
            view = canvas.central_widget.add_view()
            view.camera = 'turntable'

    # 3. Create visual for either 2D or 3D fields
    if data_np.ndim == 2:
        volume = scene.visuals.Image(data_np, parent=view.scene, cmap=cmap)
        view.camera = 'panzoom'
        shape = data_np.shape
        view.camera.set_range()
    else:
        volume = scene.visuals.Volume(data_np, parent=view.scene, method='mip', cmap=cmap)
        volume.order = order
        shape = data_np.shape
        view.camera.center = (shape[2] / 2, shape[1] / 2, shape[0] / 2)
        view.camera.set_range()

    return canvas, volume

@timer
def render_colours(colour_tensor, canvas=None, window_size=(800, 600)):
    """
    Render an RGB colour tensor in a separate VisPy window.
    
    Args:
        colour_tensor: Tensor of shape (H, W, 3) with RGB values in range [0, 1]
        canvas: Optional existing VisPy canvas to reuse
        window_size: Tuple of (width, height) for the window
    
    Returns:
        Tuple of (canvas, image_visual) for the rendered colours
    """
    # Convert to numpy if needed
    if isinstance(colour_tensor, torch.Tensor):
        # Clamp to [0, 1] and convert to numpy
        colour_np = torch.clamp(colour_tensor, 0, 1).cpu().numpy().astype(np.float32)
    else:
        colour_np = np.asarray(colour_tensor, dtype=np.float32)
    
    # Ensure it's in the right format (H, W, 3) for RGB
    if colour_np.shape[2] == 3:
        # RGB format is good
        pass
    elif colour_np.shape[2] == 4:
        # RGBA - use only RGB channels
        colour_np = colour_np[:, :, :3]
    
    # Create canvas if not provided
    if canvas is None:
        canvas = scene.SceneCanvas(
            keys='interactive', 
            size=window_size, 
            show=True, 
            bgcolor='black',
            title='Colour Tensor Visualization'
        )
        view = canvas.central_widget.add_view()
        view.camera = 'panzoom'
    else:
        # Retrieve existing view
        if len(canvas.central_widget.children) > 0:
            view = canvas.central_widget.children[0]
        else:
            view = canvas.central_widget.add_view()
            view.camera = 'panzoom'
    
    # Create image visual with the colour tensor
    image_visual = scene.visuals.Image(colour_np, parent=view.scene, interpolation='nearest')
    
    # Auto-range the camera
    view.camera.set_range()
    
    return canvas, image_visual

@timer
def update_colours(image_visual, colour_tensor, canvas=None):
    """
    Update an existing colour tensor visualization with new data.
    
    Args:
        image_visual: The Image visual object to update
        colour_tensor: New tensor of shape (H, W, 3) with RGB values
        canvas: Optional canvas to refresh
    """
    # Convert to numpy if needed
    if isinstance(colour_tensor, torch.Tensor):
        colour_np = torch.clamp(colour_tensor, 0, 1).cpu().numpy().astype(np.float32)
    else:
        colour_np = np.asarray(colour_tensor, dtype=np.float32)
    
    # Update the image data
    image_visual.set_data(colour_np)
    
    # Refresh canvas if provided
    if canvas is not None:
        canvas.update()
        app.process_events()

class window:
    def __init__(self, width=800, height=800):
        self.Simwidth = 40
        self.Simheight = 40
        self.width = width
        self.height = height
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not glfw.init():
            raise RuntimeError('Could not initialize GLFW')

        self.window = glfw.create_window(self.width, self.height, 'Fluid Sim', None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError('Could not create GLFW window')

        glfw.set_cursor_pos_callback(self.window, cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, mouse_button_callback)

        glfw.make_context_current(self.window)
        glOrtho(0, self.width, 0, self.height, -1, 1)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

    def frame(self, scalar_field=None, scalar_bound=0.0):
        if scalar_field is None:
            return

        render_volume_threshold(scalar_field, scalar_bound)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def cursor_pos_callback(window, xpos, ypos):
        global mouse_pos
        # Get window size to normalize coordinates
        win_w, win_h = glfw.get_window_size(window)
        # Map pixel (x, y) to simulation (width, height)
        # Note: OpenGL/GLFW Y-axis is often inverted compared to simulation
        mouse_pos[0] = (xpos / win_w) * self.Simwidth
        mouse_pos[1] = (1.0 - ypos / win_h) * self.Simheight

    def mouse_button_callback(window, button, action, mods):
        global mouse_pressed
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                mouse_pressed = True
            elif action == glfw.RELEASE:
                mouse_pressed = False
"""

class window:
    def __init__(self):
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True, bgcolor='black')
        self.view = canvas.central_widget.add_view()
        self.view.camera = 'turntable'
    @timer
    def render_volume_threshold(self,scalar_field, scalar_bound=0.5):
        # 1. GPU Pre-processing (Normalization)
        if not isinstance(scalar_field, torch.Tensor):
            scalar_field = torch.from_numpy(np.asarray(scalar_field))
        
        field_clipped = torch.clamp(scalar_field - scalar_bound, min=0.0)
        f_max = field_clipped.max()
        norm_gpu = field_clipped / f_max if f_max > 0 else field_clipped
        
        # Transfer to CPU for VisPy
        data_np = norm_gpu.cpu().numpy().astype(np.float32)

        # 2. Handle Canvas and View Logic
        
        # Retrieve existing view if it exists
        if len(canvas.central_widget.children) > 0:
            self.view = canvas.central_widget.children[0]
        else:
            self.view = canvas.central_widget.add_view()
            self.view.camera = 'turntable'

        # 3. Create Volume Visual
        # Added 'parent=view.scene' to ensure it attaches to the existing coordinate system
        volume = scene.visuals.Volume(data_np, parent=self.view.scene, method='mip')
        
        # 4. Camera Setup
        # Calculate center based on actual data shape
        shape = data_np.shape
        self.view.camera.center = (shape[2] / 2, shape[1] / 2, shape[0] / 2)
        self.view.camera.set_range() 

    @timer
    def render_vector_grid(self,vector_field, spacing=2, length_scale=1.0, color=(0.2, 0.7, 1.0, 0.8)):
        # 1. Downsample and Process Tensors (GPU-side)
        subsampled = vector_field[::spacing, ::spacing, ::spacing, :]
        h, w, d, _ = subsampled.shape
        
        z, y, x = torch.meshgrid(
            torch.arange(0, h * spacing, spacing, device=vector_field.device),
            torch.arange(0, w * spacing, spacing, device=vector_field.device),
            torch.arange(0, d * spacing, spacing, device=vector_field.device),
            indexing='ij'
        )
        
        p0 = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        p1 = p0 + (subsampled.reshape(-1, 3) * length_scale)
        
        # Interleave starts and ends for 'segments' mode
        line_coords = torch.stack([p0, p1], dim=1).reshape(-1, 3)
        vertices = line_coords.cpu().numpy().astype(np.float32)

        # 2. Handle Canvas and View Logic

        # Retrieve the existing view from the canvas
        # If the canvas was created with central_widget.add_view(), it's usually the first child
        if len(canvas.central_widget.children) > 0:
            self.view = canvas.central_widget.children[0]
        else:
            self.view = canvas.central_widget.add_view()
            self.view.camera = 'turntable'

        # 3. Add the Visual to the existing Scene
        lines = scene.visuals.Line(pos=vertices, connect='segments', 
                                color=color, parent=self.view.scene)

        # Re-adjust camera to fit both old and new data
        self.view.camera.set_range()
        
        return canvas
"""


# --- EXECUTION ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create a sphere-like gradient so it's not just random noise
    coords = torch.linspace(-1, 1, 16, device=device)
    x, y, z = torch.meshgrid(coords, coords, coords, indexing='ij')
    sphere = 1.0 - torch.sqrt(x**2 + y**2 + z**2)

    canvas = render_volume_threshold(sphere, scalar_bound=0.2)
    
    # IMPORTANT: This starts the actual windowing system
        # --- Example Usage ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create a "Swirl" field: (16x16x16x3)
    grid_range = torch.linspace(-1, 1, 16, device=device)
    z, y, x = torch.meshgrid(grid_range, grid_range, grid_range, indexing='ij')

    # Example math: vectors pointing in a circular pattern around the Z-axis
    u = -y
    v = x
    w = torch.zeros_like(x)
    field = torch.stack([u, v, w], dim=-1)

    canvas = render_vector_grid(field, spacing=2, length_scale=2.0,canvas=canvas)

    #app.process_events()
    app.run()

    import sys
    if sys.flags.interactive != 1:
        app.run()