# visualizer.py
import os
import numpy as np
import torch
from gs_renderer import Renderer, MiniCam
from cam_utils import OrbitCamera, orbit_camera
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class GaussianVisualizer:
    def __init__(self, opt=None, renderers=None, cam=None, debug=False):
        # Validate opt parameter
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt
        
        if self.opt.visualize:
            self.save_dir = os.path.join(self.opt.outdir, 'visualizations')
            os.makedirs(self.save_dir, exist_ok=True)
            
        self.renderers = renderers
        # Handle both single camera and list of cameras for backward compatibility
        self.cam = cam
        
        self.device = torch.device("cuda")
        
        # viewpoints for multi-viewpoints
        self.multi_viewpoints = self.set_viewpoints(opt.num_views)
    
    @torch.no_grad()
    def set_viewpoints(self, num_views=8):
        vers = [0.0] * num_views
        hors = np.linspace(0, 360, num_views)
        return [(vers[i], hors[i]) for i in range(num_views)]

    @torch.no_grad()
    def save_rendered_images(self, step, images):
        # Create output directories for rendered images and visualizations
        if not os.path.exists(os.path.join(self.save_dir, 'rendered_images')):
            os.makedirs(os.path.join(self.save_dir, 'rendered_images'), exist_ok=True)
        
        vutils.save_image(
            images,
            os.path.join(self.save_dir, 'rendered_images', f'step_{step}.png'),
            normalize=False
        )

    # TODO: refactor this function to include legend and labels
    @torch.no_grad()
    def visualize_all_particles_in_multi_viewpoints(self, step, num_views=None, visualize=None,save_iid=None): # [V, N, 3, H, W]
        """
        Render images from specific viewpoints.
        
        Args:
            step: Current training step
            num_views: Number of viewpoints to render
            save_iid: Whether to save individual particle images
        
        Returns:
            multi_viewpoint_images: [V, N, 3, H, W]
        """
        # Create multi-viewpoints surrounding the object horizontally
        if num_views is None:
            num_views = self.opt.num_views # for evaluation use default number of view points
        multi_viewpoints = self.set_viewpoints(num_views) # for visualization, customizable number of view points
        
        if save_iid is None:
            save_iid = self.opt.save_iid
            
        if visualize is None:
            visualize = self.opt.visualize
            
        # Create output directory
        if visualize:
            if self.opt.num_particles > 1:
                multi_viewpoints_dir = os.path.join(self.save_dir, f'step_{step}_view_{num_views}_all_particles')
                os.makedirs(multi_viewpoints_dir, exist_ok=True)
            
            if save_iid:
                save_paths = []
                for particle_id in range(self.opt.num_particles):
                    path = os.path.join(self.save_dir, f'step_{step}_view_{num_views}_particle_{particle_id}')
                    os.makedirs(path, exist_ok=True)
                    save_paths.append(path)
        
        # Render images from each viewpoint
        multi_viewpoint_images = []
        for i, (ver, hor) in enumerate(multi_viewpoints):
            
            # Get camera pose using orbit_camera function
            pose = orbit_camera(ver, hor, self.cam.radius)
            
            # Create MiniCam with current camera state
            camera = MiniCam(
                pose,
                self.opt.W,
                self.opt.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far
            )
            
            # Render each particle from this viewpoint
            particle_images = []
            for particle_id in range(self.opt.num_particles):

                # Render with white background
                bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
                out = self.renderers[particle_id].render(camera, bg_color=bg_color)
                image = out["image"].unsqueeze(0)  # [1, 3, H, W]
                particle_images.append(image)
                
                # Save individual particle images
                # TODO: refactor this to include legend and labels
                if visualize and save_iid:
                    vutils.save_image(
                        image,
                        os.path.join(save_paths[particle_id], f'view_{i:03d}.png'),  # Use 3-digit padding for proper ordering
                        normalize=False
                    )
            
            # save all particles images in one image in parallel
            particle_images = torch.cat(particle_images, dim=0) # [N, 3, H, W]            
            multi_viewpoint_images.append(particle_images) # [V, N, 3, H, W]
            
            # Save combined image of all particles
            # TODO: refactor this to include legend and labels
            if visualize and self.opt.num_particles > 1:
                vutils.save_image(
                    particle_images, # [N, 3, H, W]
                        os.path.join(multi_viewpoints_dir, f'view_{i:03d}.png'),
                        normalize=False
                    )
        
        multi_viewpoint_images = torch.stack(multi_viewpoint_images, dim=0) # [V, N, 3, H, W]
        
        return multi_viewpoint_images

    @torch.no_grad()
    def visualize_fixed_viewpoint(self, step, elevation, horizontal):
        """
        Render images from a custom viewpoint with specific angles.
        
        Args:
            step: Current training step
            elevation: Vertical angle in degrees (-90 to 90, where 90 is top, -90 is bottom)
            horizontal: Horizontal angle in degrees (0 to 360, where 0 is front, 90 is right)
        """
        
        # Create output directory
        if self.opt.visualize:
            viewpoint_dir = os.path.join(self.save_dir, f'step_{step}_view_{elevation}_{horizontal}')
            os.makedirs(viewpoint_dir, exist_ok=True)
        
        # Get camera pose
        pose = orbit_camera(elevation, horizontal, self.cam.radius)
        
        # Create camera
        camera = MiniCam(
            pose,
            self.opt.W,
            self.opt.H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far
        )
        # Render each particle from this viewpoint
        particle_images = []
        for particle_id in range(self.opt.num_particles):
            
            # Render with white background
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            out = self.renderers[particle_id].render(camera, bg_color=bg_color)
            image = out["image"].unsqueeze(0)  # [1, 3, H, W]
            particle_images.append(image)
        
            # Save individual particle images
            if self.opt.visualize:
                filename = f'particle_{particle_id}.png'
                vutils.save_image(
                    image,
                    os.path.join(viewpoint_dir, filename),
                    normalize=False
                )
                
        particle_images = torch.cat(particle_images, dim=0) # [N, 3, H, W]
        
        # Save combined image of all particles
        if self.opt.visualize:
            filename = f'all_particles.png'
            vutils.save_image(
                particle_images,
                os.path.join(viewpoint_dir, filename),
                normalize=False
            )
        
        return particle_images
    
    @torch.no_grad()
    def _debug_rendering_configurations(self, step):
        """Test different rendering configurations to help debug visibility issues"""
        
        
        print(f"[DEBUG] Starting visualization for step {step}")
        print(f"[DEBUG] Number of particles: {self.opt.num_particles}")
        
        # Enhanced debugging - Print Gaussian stats for debugging
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
            
            # Check if any Gaussians are visible (opacity > threshold)
            visible_gaussians = (opacity > 0.01).sum()
            print(f"[DEBUG] Particle {particle_id}: Visible Gaussians (opacity > 0.01): {visible_gaussians}")
            
            # Check distance from origin
            distances = torch.norm(xyz, dim=1)
            print(f"[DEBUG] Particle {particle_id}: Distance from origin - mean={distances.mean():.3f}, min={distances.min():.3f}, max={distances.max():.3f}")
        
        print(f"[DEBUG] Camera radius: {self.cam.radius}")
        print(f"[DEBUG] Camera fovy: {self.cam.fovy}")
        
        # Test rendering with different configurations first
        print(f"[DEBUG] Testing rendering configurations...")
        
        test_output_dir = os.path.join(self.save_dir, f'debug_renders_step_{step}')
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Test different camera distances
        test_radii = [2.0, 2.5, 3.0, 3.5]
        # test_backgrounds = [
        #     torch.tensor([1.0, 1.0, 1.0], device=self.device),  # white
        #     torch.tensor([0.0, 0.0, 0.0], device=self.device),  # black
        #     torch.tensor([0.5, 0.5, 0.5], device=self.device),  # gray
        # ]
        # bg_names = ['white', 'black', 'gray']
        
        bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device) # white
        
        for particle_id in range(min(2, self.opt.num_particles)):  # Test first 2 particles only
            print(f"[DEBUG] Testing particle {particle_id}")
            
            for radius_idx, test_radius in enumerate(test_radii):
            # for bg_idx, bg_color in enumerate(test_backgrounds):
                # Create camera at different distances
                pose = orbit_camera(0, 0, test_radius)  # Front view
                camera = MiniCam(
                    pose,
                    512, 512,  # Fixed resolution for testing
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far
                )
                
                out = self.renderers[particle_id].render(camera, bg_color=bg_color)
                image = out["image"]
                
                # Save test image
                filename = f'test_particle_{particle_id}_radius_{test_radius}.png'
                vutils.save_image(
                    image.unsqueeze(0),
                    os.path.join(test_output_dir, filename),
                    normalize=False
                )
                
                # Log statistics
                # non_bg_pixels = 0
                # if bg_idx == 0:  # white background
                #     non_bg_pixels = (image.sum(dim=0) < 2.9).sum().item()
                # elif bg_idx == 1:  # black background
                #     non_bg_pixels = (image.sum(dim=0) > 0.1).sum().item()
                non_bg_pixels = (image.sum(dim=0) < 2.9).sum().item()
                print(f"[DEBUG]   Radius {test_radius}, BG: white, non-bg pixels = {non_bg_pixels}")

    @torch.no_grad()
    def _debug_rendered_images(self, step, image, out, particle_id):
        image_stats = {
            'min': image.min().item(),
            'max': image.max().item(), 
            'mean': image.mean().item(),
            'non_white_pixels': (image.sum(dim=1) < 2.9).sum().item(),  # Pixels that aren't white
            'unique_values': len(torch.unique(image.view(-1))),
            'shape': image.shape
        }
        print(f"[DEBUG] Step {step}, Particle {particle_id}: {image_stats}")
        
        # Additional render output analysis
        if 'depth' in out:
            depth = out['depth']
            print(f"[DEBUG] Step {step}, Particle {particle_id}: Depth - min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")
        if 'alpha' in out:
            alpha = out['alpha'] 
            print(f"[DEBUG] Step {step}, Particle {particle_id}: Alpha - min={alpha.min():.3f}, max={alpha.max():.3f}, mean={alpha.mean():.3f}")

        