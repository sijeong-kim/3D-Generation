# visualizer.py
"""
3D Gaussian Splatting Visualization and Rendering Module

This module provides comprehensive visualization capabilities for 3D Gaussian Splatting models,
enabling multi-viewpoint rendering, individual particle visualization, and advanced debugging
tools for analyzing rendering configurations and image quality.

Key Features:
- Multi-viewpoint rendering with customizable camera orbits and elevation control
- Individual particle visualization and combined particle views with flexible saving options
- Fixed viewpoint rendering with precise elevation and azimuth angle control
- Comprehensive debugging tools for rendering configuration analysis and optimization
- Automatic directory management and organized file structure
- Real-time rendering statistics and quality assessment
- Support for both single-particle and multi-particle scenarios

Rendering Capabilities:
- Horizontal orbit rendering with configurable number of viewpoints
- Fixed viewpoint rendering with custom camera positioning
- White background rendering for consistent visualization
- Automatic camera pose computation and parameter management
- Batch rendering with efficient tensor operations

Debugging and Analysis:
- Gaussian property analysis (position, opacity, scaling statistics)
- Visibility analysis and distance distribution assessment
- Camera configuration testing with multiple distance parameters
- Pixel-level image analysis and quality metrics
- Comprehensive logging and diagnostic output

Directory Structure:
The module creates organized directory structures for different visualization types:
- Multi-viewpoint renders: step_{step}_view_{num_views}_all_particles/
- Individual particle renders: step_{step}_view_{num_views}_particle_{id}/
- Fixed viewpoint renders: step_{step}_view_{elevation}_{horizontal}/
- Debug renders: debug_renders_step_{step}/

Author: [Your Name]
Date: [Date]
"""

import os
import numpy as np
import torch
from gs_renderer import Renderer, MiniCam
from cam_utils import OrbitCamera, orbit_camera
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class GaussianVisualizer:
    """
    A comprehensive visualizer for 3D Gaussian Splatting models with advanced rendering capabilities.
    
    This class provides methods to render and save images from multiple viewpoints,
    visualize individual particles, and debug rendering configurations. It supports
    both single-particle and multi-particle scenarios with flexible camera controls
    and comprehensive debugging tools.
    
    The visualizer creates organized directory structures for different visualization
    types and provides detailed analysis tools for optimizing rendering parameters
    and diagnosing rendering issues.
    
    Attributes:
        opt: Configuration object containing visualization parameters including:
             - visualize: Whether to enable visualization saving
             - outdir: Output directory for all visualizations
             - num_views: Number of viewpoints for multi-view rendering
             - num_particles: Number of particles to visualize
             - W, H: Image width and height
             - save_iid: Whether to save individual particle images
             - save_multi_viewpoints: Whether to save multi-viewpoint renders
             - save_multi_viewpoints_interval: Interval for saving multi-view renders
        renderers: List of renderer objects, one per particle. Each renderer should
                  have a render() method and gaussians attribute for debugging.
        cam: Camera configuration object with radius, fovy, fovx, near, far attributes.
        device: PyTorch device for tensor operations (automatically selected).
        save_dir: Directory for saving visualization outputs (created if visualize=True).
        multi_viewpoints: Pre-computed camera viewpoints for efficient multi-view rendering.
    """
    
    def __init__(self, opt=None, renderers=None, cam=None, debug=False):
        """
        Initialize the GaussianVisualizer with configuration and rendering components.
        
        This method sets up the visualizer with the provided configuration and validates
        required parameters. It initializes the output directory structure and pre-computes
        camera viewpoints for efficient rendering.
        
        Args:
            opt: Configuration object containing visualization parameters. Must include:
                 - visualize (bool): Whether to enable visualization saving
                 - outdir (str): Output directory for all visualizations
                 - num_views (int): Number of viewpoints for multi-view rendering
                 - num_particles (int): Number of particles to visualize
                 - W, H (int): Image width and height for rendering
                 - save_iid (bool): Whether to save individual particle images
                 - save_multi_viewpoints (bool): Whether to save multi-viewpoint renders
                 - save_multi_viewpoints_interval (int): Interval for saving multi-view renders
            renderers: List of renderer objects, one per particle. Each renderer should
                      have a render() method and gaussians attribute for debugging.
                      The renderers are used to generate images from different viewpoints.
            cam: Camera configuration object with the following attributes:
                 - radius (float): Camera distance from origin
                 - fovy, fovx (float): Vertical and horizontal field of view angles
                 - near, far (float): Near and far clipping planes
            debug (bool): Whether to enable debug mode for additional logging and analysis.
                         Default: False
        
        Raises:
            ValueError: If opt parameter is None or required configuration fields are missing.
        
        Note:
            - The device is automatically selected (CUDA if available, otherwise CPU)
            - The output directory is only created if visualization is enabled
            - Camera viewpoints are pre-computed for efficient multi-view rendering
            - All tensor operations are performed with torch.no_grad() for efficiency
        
        Example:
            >>> # Basic initialization
            >>> visualizer = GaussianVisualizer(opt, renderers, cam)
            >>> 
            >>> # With debug mode enabled
            >>> visualizer = GaussianVisualizer(opt, renderers, cam, debug=True)
        """
        # Validate required parameters
        if opt is None:
            raise ValueError("Configuration object 'opt' cannot be None")
        self.opt = opt
        
        # Initialize visualization directory if enabled
        if self.opt.visualize:
            self.save_dir = os.path.join(self.opt.outdir, 'visualizations')
            os.makedirs(self.save_dir, exist_ok=True)
            
        self.renderers = renderers
        self.cam = cam  # Supports both single camera and camera lists for backward compatibility
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute multi-viewpoint camera positions for efficient rendering
        self.multi_viewpoints = self.set_viewpoints(opt.num_views)
    
    @torch.no_grad()
    def set_viewpoints(self, num_views=8):
        """
        Generate a set of camera viewpoints for multi-view rendering.
        
        This method creates a horizontal orbit around the object at zero elevation,
        distributing viewpoints evenly across 360 degrees. The viewpoints are designed
        for comprehensive 3D scene analysis and consistent multi-view visualization.
        
        Methodology:
        1. Set all viewpoints to zero elevation (horizontal orbit)
        2. Distribute azimuth angles evenly from 0 to 360 degrees
        3. Return list of (elevation, azimuth) tuples for camera positioning
        
        Args:
            num_views (int): Number of viewpoints to generate. The viewpoints will be
                           distributed evenly around a 360-degree horizontal orbit.
                           Default: 8
        
        Returns:
            List[Tuple[float, float]]: List of camera viewpoints as (elevation, azimuth) tuples
                                      where:
                                      - elevation: Always 0.0 (horizontal orbit)
                                      - azimuth: Evenly distributed from 0 to 360 degrees
        
        Note:
            - All viewpoints are at zero elevation for consistent horizontal analysis
            - Azimuth angles are evenly distributed for comprehensive scene coverage
            - The first viewpoint is always at azimuth 0 (front view)
            - The last viewpoint is at azimuth 360 - 360/num_views
        
        Example:
            >>> viewpoints = visualizer.set_viewpoints(4)
            >>> print(viewpoints)
            [(0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, 270.0)]
            >>> 
            >>> viewpoints = visualizer.set_viewpoints(8)
            >>> print(viewpoints)
            [(0.0, 0.0), (0.0, 45.0), (0.0, 90.0), ..., (0.0, 315.0)]
        """
        elevations = [0.0] * num_views  # All viewpoints at zero elevation
        azimuths = np.linspace(0, 360, num_views, endpoint=False)  # Evenly distributed azimuth angles
        return [(elevations[i], azimuths[i]) for i in range(num_views)]

    @torch.no_grad()
    def save_rendered_images(self, step, images):
        """
        Save rendered images to the visualization directory with step-based naming.
        
        This method creates a 'rendered_images' subdirectory within the main visualization
        directory and saves the provided images with a step-based naming convention.
        Images are saved without normalization to preserve original pixel values.
        
        Args:
            step (int): Current training step for file naming. Used to create
                       step-based filenames for tracking training progress.
            images (torch.Tensor): Tensor of images to save with shape [N, C, H, W] where:
                                  - N: Number of images (batch size)
                                  - C: Number of channels (typically 3 for RGB)
                                  - H, W: Image height and width
        
        Note:
            - Images are saved without normalization to preserve original pixel values
            - The output directory is automatically created if it doesn't exist
            - Files are named as 'step_{step}.png' for easy tracking
            - All operations are performed with torch.no_grad() for efficiency
        
        Example:
            >>> # Save a batch of rendered images
            >>> images = torch.randn(4, 3, 256, 256)  # 4 RGB images, 256x256
            >>> visualizer.save_rendered_images(100, images)
            >>> # Creates: visualizations/rendered_images/step_100.png
        """
        # Ensure output directory exists
        rendered_images_dir = os.path.join(self.save_dir, 'rendered_images')
        if not os.path.exists(rendered_images_dir):
            os.makedirs(rendered_images_dir, exist_ok=True)
        
        # Save images with step-based naming
        output_path = os.path.join(rendered_images_dir, f'step_{step}.png')
        vutils.save_image(images, output_path, normalize=False)

    @torch.no_grad()
    def visualize_all_particles_in_multi_viewpoints(self, step, num_views=None, visualize=None, save_iid=None):
        """
        Render all particles from multiple viewpoints and optionally save individual images.
        
        This method creates a comprehensive multi-view visualization by rendering each particle
        from multiple camera positions arranged in a horizontal orbit. It supports both
        individual particle saving and combined particle views with flexible saving options.
        
        Methodology:
        1. Generate camera viewpoints for the specified number of views
        2. For each viewpoint, render all particles using the configured renderers
        3. Optionally save individual particle images and combined views
        4. Return multi-viewpoint image tensor for further processing
        
        Args:
            step (int): Current training step for file naming and interval checking.
                       Used to determine whether to save images based on save intervals.
            num_views (int, optional): Number of viewpoints to render. If None, uses
                                     opt.num_views from configuration. Default: None
            visualize (bool, optional): Whether to save images. If None, uses opt.visualize
                                       from configuration. Default: None
            save_iid (bool, optional): Whether to save individual particle images. If None,
                                     uses opt.save_iid from configuration. Default: None
        
        Returns:
            torch.Tensor: Multi-viewpoint images with shape [V, N, 3, H, W] where:
                         - V: Number of viewpoints
                         - N: Number of particles
                         - 3: RGB channels
                         - H, W: Image height and width
        
        Directory Structure (when saving enabled):
            visualizations/
            ├── step_{step}_view_{num_views}_all_particles/
            │   ├── view_000.png  # Combined view of all particles
            │   ├── view_001.png
            │   └── ...
            └── step_{step}_view_{num_views}_particle_{id}/
                ├── view_000.png  # Individual particle view
                ├── view_001.png
                └── ...
        
        Note:
            - Images are only saved when step % save_multi_viewpoints_interval == 0
            - Individual particle images are only saved when save_iid is True
            - Combined particle images are only saved when num_particles > 1
            - All rendering uses white background for consistent visualization
            - Camera poses are computed using orbit_camera() function
        
        Example:
            >>> # Render with default settings
            >>> images = visualizer.visualize_all_particles_in_multi_viewpoints(100)
            >>> print(f"Multi-view images shape: {images.shape}")  # [8, 4, 3, 256, 256]
            >>> 
            >>> # Render with custom settings
            >>> images = visualizer.visualize_all_particles_in_multi_viewpoints(
            ...     100, num_views=12, save_iid=True)
        """
        # Use default values if parameters are not provided
        if num_views is None:
            num_views = self.opt.num_views
        if save_iid is None:
            save_iid = self.opt.save_iid
        if visualize_multi_viewpoints is None:
            visualize_multi_viewpoints = self.opt.visualize_multi_viewpoints
            
        # Create camera object with current pose and parameters
        # used ref_size (256) instead of W, H (800) to decrease memory usage
        eval_W = getattr(self.opt, 'eval_W', self.opt.W) 
        eval_H = getattr(self.opt, 'eval_H', self.opt.H)
        eval_radius = getattr(self.opt, 'eval_radius', self.cam.radius)
            
        # Generate camera viewpoints for the specified number of views
        multi_viewpoints = self.set_viewpoints(num_views)
        
        # Initialize directory paths for saving (only if saving is enabled)
        multi_viewpoints_dir = None
        save_paths = None
        
        if visualize_multi_viewpoints and (step % self.opt.save_multi_viewpoints_interval == 0):
            # Create directory for combined particle views
            if self.opt.num_particles > 1:
                multi_viewpoints_dir = os.path.join(
                    self.save_dir, 
                    f'step_{step}_view_{num_views}_all_particles'
                )
                os.makedirs(multi_viewpoints_dir, exist_ok=True)
                
            # Create directories for individual particle views
            if save_iid:
                save_paths = []
                for particle_id in range(self.opt.num_particles):
                    particle_dir = os.path.join(
                        self.save_dir, 
                        f'step_{step}_view_{num_views}_particle_{particle_id}'
                    )
                    os.makedirs(particle_dir, exist_ok=True)
                    save_paths.append(particle_dir)
        
        # Render images from each viewpoint
        multi_viewpoint_images = []
        for i, (elevation, azimuth) in enumerate(multi_viewpoints):
            
            # Compute camera pose for current viewpoint
            pose = orbit_camera(elevation, azimuth, eval_radius)
            
            camera = MiniCam(
                pose,
                eval_W,
                eval_H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far
            )
            
            # Render each particle from current viewpoint
            particle_images = []
            for particle_id in range(self.opt.num_particles):
                # Render with white background for consistent visualization
                bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
                out = self.renderers[particle_id].render(camera, bg_color=bg_color)
                image = out["image"].unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
                particle_images.append(image)
                
                # Save individual particle image if enabled
                if visualize_multi_viewpoints and save_iid and (step % self.opt.save_multi_viewpoints_interval == 0):
                    output_path = os.path.join(save_paths[particle_id], f'view_{i:03d}.png')
                    vutils.save_image(image, output_path, normalize=False)
            
            # Combine all particle images for current viewpoint
            particle_images = torch.cat(particle_images, dim=0)  # [N, 3, H, W]
            multi_viewpoint_images.append(particle_images)
            
            # Save combined view of all particles if multiple particles exist
            if visualize_multi_viewpoints and (self.opt.num_particles > 1) and (step % self.opt.save_multi_viewpoints_interval == 0):
                output_path = os.path.join(multi_viewpoints_dir, f'view_{i:03d}.png')
                vutils.save_image(particle_images, output_path, normalize=False)
        
        # Stack all viewpoints into final tensor
        multi_viewpoint_images = torch.stack(multi_viewpoint_images, dim=0)  # [V, N, 3, H, W]
        
        return multi_viewpoint_images

    @torch.no_grad()
    def visualize_fixed_viewpoint(self, step, elevation, horizontal):
        """
        Render images from a specific fixed viewpoint with custom elevation and azimuth angles.
        
        This method provides precise control over camera positioning, allowing for
        detailed examination of the 3D scene from any desired angle. It renders all
        particles from the specified viewpoint and optionally saves individual and
        combined particle images.
        
        Methodology:
        1. Compute camera pose for the specified elevation and azimuth angles
        2. Create camera object with the computed pose and configured parameters
        3. Render each particle from the fixed viewpoint
        4. Optionally save individual particle images and combined view
        5. Return particle images tensor for further processing
        
        Args:
            step (int): Current training step for file naming. Used to create
                       step-based directory names for organizing outputs.
            elevation (float): Vertical angle in degrees. Range: [-90, 90]
                             - 90: Directly above the scene (top view)
                             - 0: Horizontal view (eye level)
                             - -90: Directly below the scene (bottom view)
            horizontal (float): Horizontal angle in degrees. Range: [0, 360]
                              - 0: Front view
                              - 90: Right view
                              - 180: Back view
                              - 270: Left view
        
        Returns:
            torch.Tensor: Particle images with shape [N, 3, H, W] where:
                         - N: Number of particles
                         - 3: RGB channels
                         - H, W: Image height and width
        
        Directory Structure (when visualization enabled):
            visualizations/
            └── step_{step}_view_{elevation}_{horizontal}/
                ├── particle_0.png  # Individual particle images
                ├── particle_1.png
                ├── ...
                └── all_particles.png  # Combined view of all particles
        
        Note:
            - Camera pose is computed using orbit_camera() function
            - All rendering uses white background for consistent visualization
            - Individual particle images are saved when visualization is enabled
            - Combined particle image is only saved when num_particles > 1
            - All operations are performed with torch.no_grad() for efficiency
        
        Example:
            >>> # Render from front view at eye level
            >>> images = visualizer.visualize_fixed_viewpoint(100, 0, 0)
            >>> print(f"Front view images shape: {images.shape}")  # [4, 3, 256, 256]
            >>> 
            >>> # Render from top-down view
            >>> images = visualizer.visualize_fixed_viewpoint(100, 90, 0)
            >>> print(f"Top view images shape: {images.shape}")  # [4, 3, 256, 256]
            >>> 
            >>> # Render from side view
            >>> images = visualizer.visualize_fixed_viewpoint(100, 0, 90)
            >>> print(f"Side view images shape: {images.shape}")  # [4, 3, 256, 256]
        """
        
        # Create output directory for this specific viewpoint
        if self.opt.visualize_fixed_viewpoint:
            viewpoint_dir = os.path.join(
                self.save_dir, 
                f'step_{step}_view_{elevation}_{horizontal}'
            )
            os.makedirs(viewpoint_dir, exist_ok=True)
        
        # Compute camera pose for specified angles
        pose = orbit_camera(elevation, horizontal, self.cam.radius)
        
        # Create camera object with computed pose
        camera = MiniCam(
            pose,
            self.opt.W,
            self.opt.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far
        )
        
        # Render each particle from the fixed viewpoint
        particle_images = []
        for particle_id in range(self.opt.num_particles):
            # Render with white background for consistent visualization
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            out = self.renderers[particle_id].render(camera, bg_color=bg_color)
            image = out["image"].unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
            particle_images.append(image)
        
            # Save individual particle image if visualization is enabled
            if self.opt.visualize_fixed_viewpoint:
                filename = f'particle_{particle_id}.png'
                output_path = os.path.join(viewpoint_dir, filename)
                vutils.save_image(image, output_path, normalize=False)
                
        # Combine all particle images
        particle_images = torch.cat(particle_images, dim=0)  # [N, 3, H, W]
        
        # Save combined view of all particles if visualization is enabled
        if self.opt.visualize_fixed_viewpoint:
            filename = f'all_particles.png'
            output_path = os.path.join(viewpoint_dir, filename)
            vutils.save_image(particle_images, output_path, normalize=False)
        
        return particle_images
    
    @torch.no_grad()
    def _debug_rendering_configurations(self, step):
        """
        Comprehensive debugging tool for analyzing rendering configurations and Gaussian properties.
        
        This method performs detailed analysis of the rendering setup, including Gaussian
        statistics, visibility analysis, distance distribution, and camera configuration
        testing. It helps diagnose rendering issues and optimize rendering parameters.
        
        Analysis Components:
        1. Gaussian Property Analysis:
           - Position statistics (mean, std, min, max)
           - Opacity statistics and visibility analysis
           - Scaling statistics for size assessment
           - Distance distribution from origin
        
        2. Camera Configuration Testing:
           - Multiple camera distances for optimal viewing
           - Pixel-level analysis of rendered images
           - Non-background pixel counting for visibility assessment
        
        3. Rendering Quality Assessment:
           - Test renders with different parameters
           - Statistical analysis of rendered images
           - Comparison of different camera configurations
        
        Args:
            step (int): Current training step for file naming and logging context.
                       Used to create debug directory and provide context for analysis.
        
        Output:
            Creates a debug directory with test renders and prints comprehensive
            statistics to help diagnose rendering issues.
        
        Debug Directory Structure:
            visualizations/
            └── debug_renders_step_{step}/
                ├── test_particle_0_radius_2.0.png
                ├── test_particle_0_radius_2.5.png
                ├── test_particle_1_radius_2.0.png
                └── ...
        
        Note:
            This method is intended for debugging purposes and should not be used
            in production code due to its verbose output and file generation.
            It provides comprehensive analysis but may be computationally expensive.
        
        Example:
            >>> # Run comprehensive debugging analysis
            >>> visualizer._debug_rendering_configurations(100)
            >>> # Creates debug directory with test renders and prints statistics
        """
        
        print(f"[DEBUG] Starting comprehensive visualization analysis for step {step}")
        print(f"[DEBUG] Number of particles: {self.opt.num_particles}")
        
        # Analyze Gaussian properties for each particle
        for particle_id in range(self.opt.num_particles):
            gaussians = self.renderers[particle_id].gaussians
            num_gaussians = gaussians.get_xyz.shape[0]
            xyz = gaussians.get_xyz
            opacity = gaussians.get_opacity
            scaling = gaussians.get_scaling
            
            print(f"[DEBUG] Particle {particle_id}: {num_gaussians} Gaussians")
            print(f"[DEBUG] Particle {particle_id}: XYZ mean={xyz.mean(dim=0)}, std={xyz.std(dim=0)}")
            print(f"[DEBUG] Particle {particle_id}: XYZ min={xyz.min(dim=0)[0]}, max={xyz.max(dim=0)[0]}")
            print(f"[DEBUG] Particle {particle_id}: Opacity mean={opacity.mean():.6f}, min={opacity.min():.6f}, max={opacity.max():.6f}")
            print(f"[DEBUG] Particle {particle_id}: Scaling mean={scaling.mean(dim=0)}, min={scaling.min(dim=0)[0]}, max={scaling.max(dim=0)[0]}")
            
            # Analyze visibility of Gaussians
            visible_gaussians = (opacity > 0.01).sum()
            print(f"[DEBUG] Particle {particle_id}: Visible Gaussians (opacity > 0.01): {visible_gaussians}")
            
            # Analyze distance distribution from origin
            distances = torch.norm(xyz, dim=1)
            print(f"[DEBUG] Particle {particle_id}: Distance from origin - mean={distances.mean():.3f}, min={distances.min():.3f}, max={distances.max():.3f}")
        
        print(f"[DEBUG] Camera radius: {self.cam.radius}")
        print(f"[DEBUG] Camera fovy: {self.cam.fovy}")
        
        # Test rendering with different camera configurations
        print(f"[DEBUG] Testing rendering configurations with multiple camera distances...")
        
        test_output_dir = os.path.join(self.save_dir, f'debug_renders_step_{step}')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Test different camera distances to find optimal viewing distance
        test_radii = [2.0, 2.5, 3.0, 3.5]
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)  # White background
        
        # Test first 2 particles to avoid excessive output
        for particle_id in range(min(2, self.opt.num_particles)):
            print(f"[DEBUG] Testing particle {particle_id}")
            
            for radius_idx, test_radius in enumerate(test_radii):
                # Create camera at test distance with front view
                pose = orbit_camera(0, 0, test_radius)  # Front view (elevation=0, azimuth=0)
                camera = MiniCam(
                    pose,
                    512, 512,  # Fixed resolution for consistent testing
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far
                )
                
                # Render test image
                out = self.renderers[particle_id].render(camera, bg_color=bg_color)
                image = out["image"]
                
                # Save test image
                filename = f'test_particle_{particle_id}_radius_{test_radius}.png'
                output_path = os.path.join(test_output_dir, filename)
                vutils.save_image(image.unsqueeze(0), output_path, normalize=False)
                
                # Analyze pixel statistics (count non-background pixels)
                non_bg_pixels = (image.sum(dim=0) < 2.9).sum().item()
                print(f"[DEBUG]   Radius {test_radius}, BG: white, non-bg pixels = {non_bg_pixels}")

    @torch.no_grad()
    def _debug_rendered_images(self, step, image, out, particle_id):
        """
        Analyze and log detailed statistics about rendered images and render outputs.
        
        This method provides pixel-level analysis of rendered images and additional
        render outputs like depth and alpha maps to help diagnose rendering issues
        and assess rendering quality.
        
        Analysis Components:
        1. Image Statistics:
           - Pixel value ranges (min, max, mean)
           - Non-background pixel count for visibility assessment
           - Number of unique pixel values for complexity analysis
           - Image dimensions and shape information
        
        2. Additional Render Outputs:
           - Depth map statistics (if available)
           - Alpha map statistics (if available)
           - Other render outputs for comprehensive analysis
        
        Args:
            step (int): Current training step for logging context. Used to provide
                       temporal context for the analysis and debugging.
            image (torch.Tensor): Rendered image tensor with shape [3, H, W] where:
                                 - 3: RGB channels
                                 - H, W: Image height and width
            out (dict): Render output dictionary containing additional information
                       such as depth maps, alpha maps, and other render outputs.
                       The exact contents depend on the renderer implementation.
            particle_id (int): ID of the particle being analyzed. Used to identify
                             which particle's rendering is being analyzed.
        
        Output:
            Prints comprehensive image statistics including:
            - Pixel value ranges (min, max, mean)
            - Non-background pixel count
            - Number of unique pixel values
            - Image dimensions
            - Depth map statistics (if available)
            - Alpha map statistics (if available)
        
        Note:
            This method is intended for debugging purposes and provides verbose
            output to help diagnose rendering quality issues. It should not be
            used in production code due to its computational overhead and verbose
            output. The analysis assumes white background (pixel sum < 2.9 for
            non-background pixels).
        
        Example:
            >>> # Analyze a rendered image
            >>> image = torch.randn(3, 256, 256)  # Simulated rendered image
            >>> out = {"depth": torch.randn(256, 256), "alpha": torch.randn(256, 256)}
            >>> visualizer._debug_rendered_images(100, image, out, 0)
            >>> # Prints comprehensive image statistics
        """
        # Analyze image pixel statistics
        image_stats = {
            'min': image.min().item(),
            'max': image.max().item(), 
            'mean': image.mean().item(),
            'non_white_pixels': (image.sum(dim=0) < 2.9).sum().item(),  # Count non-background pixels
            'unique_values': len(torch.unique(image.view(-1))),
            'shape': image.shape
        }
        print(f"[DEBUG] Step {step}, Particle {particle_id}: {image_stats}")
        
        # Analyze additional render outputs if available
        if 'depth' in out:
            depth = out['depth']
            print(f"[DEBUG] Step {step}, Particle {particle_id}: Depth - min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")
        
        if 'alpha' in out:
            alpha = out['alpha'] 
            print(f"[DEBUG] Step {step}, Particle {particle_id}: Alpha - min={alpha.min():.3f}, max={alpha.max():.3f}, mean={alpha.mean():.3f}")
        
        