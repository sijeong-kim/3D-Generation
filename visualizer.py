# visualizer.py
"""
Gaussian Splatting Visualization Module

This module provides comprehensive visualization capabilities for 3D Gaussian Splatting models.
It supports multi-viewpoint rendering, individual particle visualization, and debugging tools
for analyzing rendering configurations and image quality.

Key Features:
- Multi-viewpoint rendering with customizable camera orbits
- Individual particle visualization and combined particle views
- Fixed viewpoint rendering with specific elevation and azimuth angles
- Comprehensive debugging tools for rendering configuration analysis
- Automatic directory management and file organization

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
    A comprehensive visualizer for 3D Gaussian Splatting models.
    
    This class provides methods to render and save images from multiple viewpoints,
    visualize individual particles, and debug rendering configurations. It supports
    both single-particle and multi-particle scenarios with flexible camera controls.
    
    Attributes:
        opt: Configuration object containing visualization parameters
        renderers: List of renderer objects, one per particle
        cam: Camera configuration object
        device: PyTorch device for tensor operations
        save_dir: Directory for saving visualization outputs
        multi_viewpoints: Pre-computed camera viewpoints for multi-view rendering
    """
    
    def __init__(self, opt=None, renderers=None, cam=None, debug=False):
        """
        Initialize the GaussianVisualizer.
        
        Args:
            opt: Configuration object containing visualization parameters.
                 Must include: visualize, outdir, num_views, num_particles,
                 W, H, save_iid, save_multi_viewpoints_interval
            renderers: List of renderer objects, one per particle. Each renderer
                      should have a render() method and gaussians attribute.
            cam: Camera configuration object with radius, fovy, fovx, near, far attributes.
            debug: Whether to enable debug mode for additional logging.
        
        Raises:
            ValueError: If opt parameter is None.
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
        
        Creates a horizontal orbit around the object at zero elevation,
        distributing viewpoints evenly across 360 degrees.
        
        Args:
            num_views: Number of viewpoints to generate (default: 8).
        
        Returns:
            List of tuples: [(elevation, azimuth), ...] where elevation is 0.0
                           and azimuth ranges from 0 to 360 degrees.
        
        Example:
            >>> viewpoints = visualizer.set_viewpoints(4)
            >>> print(viewpoints)
            [(0.0, 0.0), (0.0, 90.0), (0.0, 180.0), (0.0, 270.0)]
        """
        elevations = [0.0] * num_views  # All viewpoints at zero elevation
        azimuths = np.linspace(0, 360, num_views)  # Evenly distributed azimuth angles
        return [(elevations[i], azimuths[i]) for i in range(num_views)]

    @torch.no_grad()
    def save_rendered_images(self, step, images):
        """
        Save rendered images to the visualization directory.
        
        Creates a 'rendered_images' subdirectory and saves the provided images
        with step-based naming convention.
        
        Args:
            step: Current training step for file naming.
            images: Tensor of images to save, shape [N, C, H, W].
        
        Note:
            Images are saved without normalization to preserve original pixel values.
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
        individual particle saving and combined particle views.
        
        Args:
            step: Current training step for file naming and interval checking.
            num_views: Number of viewpoints to render. If None, uses opt.num_views.
            visualize: Whether to save images. If None, uses opt.visualize.
            save_iid: Whether to save individual particle images. If None, uses opt.save_iid.
        
        Returns:
            torch.Tensor: Multi-viewpoint images with shape [V, N, 3, H, W] where:
                         V = number of viewpoints, N = number of particles,
                         3 = RGB channels, H = height, W = width.
        
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
        """
        # Use default values if parameters are not provided
        if num_views is None:
            num_views = self.opt.num_views
        if save_iid is None:
            save_iid = self.opt.save_iid
        if visualize is None:
            visualize = self.opt.visualize
            
        # Generate camera viewpoints for the specified number of views
        multi_viewpoints = self.set_viewpoints(num_views)
        
        # Initialize directory paths for saving (only if saving is enabled)
        multi_viewpoints_dir = None
        save_paths = None
        
        if visualize and (step % self.opt.save_multi_viewpoints_interval == 0):
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
            pose = orbit_camera(elevation, azimuth, self.cam.radius)
            
            # Create camera object with current pose and parameters
            camera = MiniCam(
                pose,
                self.opt.W,
                self.opt.H,
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
                if visualize and save_iid and (step % self.opt.save_multi_viewpoints_interval == 0):
                    output_path = os.path.join(save_paths[particle_id], f'view_{i:03d}.png')
                    vutils.save_image(image, output_path, normalize=False)
            
            # Combine all particle images for current viewpoint
            particle_images = torch.cat(particle_images, dim=0)  # [N, 3, H, W]
            multi_viewpoint_images.append(particle_images)
            
            # Save combined view of all particles if multiple particles exist
            if visualize and (self.opt.num_particles > 1) and (step % self.opt.save_multi_viewpoints_interval == 0):
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
        detailed examination of the 3D scene from any desired angle.
        
        Args:
            step: Current training step for file naming.
            elevation: Vertical angle in degrees. Range: [-90, 90]
                      - 90: Directly above the scene (top view)
                      - 0: Horizontal view (eye level)
                      - -90: Directly below the scene (bottom view)
            horizontal: Horizontal angle in degrees. Range: [0, 360]
                       - 0: Front view
                       - 90: Right view
                       - 180: Back view
                       - 270: Left view
        
        Returns:
            torch.Tensor: Particle images with shape [N, 3, H, W] where:
                         N = number of particles, 3 = RGB channels,
                         H = height, W = width.
        
        Directory Structure (when visualization enabled):
            visualizations/
            └── step_{step}_view_{elevation}_{horizontal}/
                ├── particle_0.png  # Individual particle images
                ├── particle_1.png
                ├── ...
                └── all_particles.png  # Combined view of all particles
        
        Example:
            >>> # Render from front view at eye level
            >>> images = visualizer.visualize_fixed_viewpoint(100, 0, 0)
            >>> 
            >>> # Render from top-down view
            >>> images = visualizer.visualize_fixed_viewpoint(100, 90, 0)
        """
        
        # Create output directory for this specific viewpoint
        if self.opt.visualize:
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
            if self.opt.visualize:
                filename = f'particle_{particle_id}.png'
                output_path = os.path.join(viewpoint_dir, filename)
                vutils.save_image(image, output_path, normalize=False)
                
        # Combine all particle images
        particle_images = torch.cat(particle_images, dim=0)  # [N, 3, H, W]
        
        # Save combined view of all particles if visualization is enabled
        if self.opt.visualize:
            filename = f'all_particles.png'
            output_path = os.path.join(viewpoint_dir, filename)
            vutils.save_image(particle_images, output_path, normalize=False)
        
        return particle_images
    
    @torch.no_grad()
    def _debug_rendering_configurations(self, step):
        """
        Comprehensive debugging tool for analyzing rendering configurations and Gaussian properties.
        
        This method performs detailed analysis of the rendering setup, including:
        - Gaussian statistics (position, opacity, scaling)
        - Visibility analysis (number of visible Gaussians)
        - Distance analysis from origin
        - Camera configuration testing with multiple distances
        - Pixel-level analysis of rendered images
        
        Args:
            step: Current training step for file naming and logging.
        
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
        render outputs like depth and alpha maps to help diagnose rendering issues.
        
        Args:
            step: Current training step for logging context.
            image: Rendered image tensor with shape [3, H, W].
            out: Render output dictionary containing additional information.
            particle_id: ID of the particle being analyzed.
        
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
            output to help diagnose rendering quality issues.
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
        
        