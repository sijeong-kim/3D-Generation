# visualizer.py
import os
import numpy as np
import torch
from gs_renderer import Renderer, MiniCam
from cam_utils import OrbitCamera, orbit_camera
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class GaussianVisualizer:
    def __init__(self, save_dir=None, opt=None, renderers=None, cam=None, debug=False):
        # Validate opt parameter
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt
        
        if self.opt.visualize and not save_dir is None:
            self.save_dir = os.path.join(self.opt.outdir, 'visualizations')
            os.makedirs(self.save_dir, exist_ok=True)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderers = renderers
        # Handle both single camera and list of cameras for backward compatibility
        self.cam = cam
        
        # viewpoints for multi-viewpoints
        self.viewpoints = self.set_viewpoints(opt.num_viewpoints)
        # viewpoints for 360 degree rotation
        self.viewpoints_360 = self.set_viewpoints(120) # 120 viewpoints for 360 degree rotation
        
        
    def set_viewpoints(self, num_viewpoints=8):
        vers = [0.0] * num_viewpoints
        hors = np.linspace(0, 360, num_viewpoints)
        return [(vers[i], hors[i]) for i in range(num_viewpoints)]

    @torch.no_grad()
    def save_rendered_images(self, images, step):
        # Create output directories for rendered images and visualizations
        if not os.path.exists(os.path.join(self.save_dir, 'rendered_images')) and self.opt.visualize:
            os.makedirs(os.path.join(self.save_dir, 'rendered_images'), exist_ok=True)
        
        vutils.save_image(
            images,
            os.path.join(self.save_dir, 'rendered_images', f'step_{step}.png'),
            normalize=False
        )

        
    @torch.no_grad()
    def visualize_all_particles_360(self, step, save_individual_particles=False):
        
        
        if self.opt.visualize:
            save_all_particles_path = os.path.join(self.save_dir, f'gaussians_rendered_step_{step}_all_particles_360')
            os.makedirs(save_all_particles_path, exist_ok=True)
        
            if save_individual_particles:
                save_paths = []
                for particle_idx in range(self.opt.num_particles):
                    path = os.path.join(self.save_dir, f'gaussians_rendered_step_{step}_particle_{particle_idx}_360')
                    os.makedirs(path, exist_ok=True)
                    save_paths.append(path)
        
        for i, (ver, hor) in enumerate(self.viewpoints_360):
            
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
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            images = []
            
            for particle_idx in range(self.opt.num_particles):
                out = self.renderers[particle_idx].render(camera, bg_color=bg_color)
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                
                if save_individual_particles:
                    # Save individual particle images
                    vutils.save_image(
                        image,
                        os.path.join(save_paths[particle_idx], f'view_{i:03d}.png'),  # Use 3-digit padding for proper ordering
                        normalize=False
                    )
                
            # save all particles images in one image in parallel
            vutils.save_image(
                torch.cat(images, dim=0),
                os.path.join(save_all_particles_path, f'view_{i:03d}.png'),
                normalize=False
            )

            
    # TODO: refactor this function
    @torch.no_grad()
    def visualize_all_particles_multi_viewpoints(self, step, num_viewpoints=8): # [N, V, 3, H, W]
        """
        Render images from specific viewpoints.
        
        Args:
            step: Current training step
            num_viewpoints: Number of viewpoints to render
        """
        
        # Create output directory
        if self.opt.visualize and self.step % self.opt.vis_interval == 0:
            viewpoint_dir = os.path.join(self.save_dir, f'all_particles_viewpoints_step_{step}')
            os.makedirs(viewpoint_dir, exist_ok=True)

        
        viewpoint_images = []
        
        for i, (ver, hor) in enumerate(self.viewpoints):
            
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
            for particle_idx in range(self.opt.num_particles):

                # Render with white background
                bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
                out = self.renderers[particle_idx].render(camera, bg_color=bg_color)
                image = out["image"].unsqueeze(0)  # [1, 3, H, W]
                particle_images.append(image)
            
            # save all particles images in one image in parallel
            particle_images = torch.cat(particle_images, dim=0) # [N, 3, H, W]
            
            viewpoint_images.append(particle_images)
            
            # Save combined image of all particles
            if self.opt.num_particles > 1 and self.opt.visualize:
                filename = f'view_{i:03d}_all_particles.png'
                vutils.save_image(
                    particle_images, # [N, 3, H, W]
                        os.path.join(viewpoint_dir, filename),
                        normalize=False
                    )
        
        multi_viewpoint_images = torch.stack(viewpoint_images, dim=1) # [N, V, 3, H, W]
        
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
            viewpoint_dir = os.path.join(self.save_dir, f'fixed_viewpoints_step_{step}_elev{elevation}_hor{horizontal}')
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
        for particle_idx in range(self.opt.num_particles):
            
            # Render with white background
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            out = self.renderers[particle_idx].render(camera, bg_color=bg_color)
            image = out["image"].unsqueeze(0)  # [1, 3, H, W]
            particle_images.append(image)
        
            # Save individual particle images
            if self.opt.visualize:
                filename = f'particle_{particle_idx}.png'
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
        for particle_idx in range(self.opt.num_particles):
            gaussians = self.renderers[particle_idx].gaussians
            num_gaussians = gaussians.get_xyz.shape[0]
            xyz = gaussians.get_xyz
            opacity = gaussians.get_opacity
            scaling = gaussians.get_scaling
            
            print(f"[DEBUG] Particle {particle_idx}: {num_gaussians} Gaussians")
            print(f"[DEBUG] Particle {particle_idx}: XYZ mean={xyz.mean(dim=0)}, std={xyz.std(dim=0)}")
            print(f"[DEBUG] Particle {particle_idx}: XYZ min={xyz.min(dim=0)[0]}, max={xyz.max(dim=0)[0]}")
            print(f"[DEBUG] Particle {particle_idx}: Opacity mean={opacity.mean():.6f}, min={opacity.min():.6f}, max={opacity.max():.6f}")
            print(f"[DEBUG] Particle {particle_idx}: Scaling mean={scaling.mean(dim=0)}, min={scaling.min(dim=0)[0]}, max={scaling.max(dim=0)[0]}")
            
            # Check if any Gaussians are visible (opacity > threshold)
            visible_gaussians = (opacity > 0.01).sum()
            print(f"[DEBUG] Particle {particle_idx}: Visible Gaussians (opacity > 0.01): {visible_gaussians}")
            
            # Check distance from origin
            distances = torch.norm(xyz, dim=1)
            print(f"[DEBUG] Particle {particle_idx}: Distance from origin - mean={distances.mean():.3f}, min={distances.min():.3f}, max={distances.max():.3f}")
        
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
        
        for particle_idx in range(min(2, self.opt.num_particles)):  # Test first 2 particles only
            print(f"[DEBUG] Testing particle {particle_idx}")
            
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
                
                out = self.renderers[particle_idx].render(camera, bg_color=bg_color)
                image = out["image"]
                
                # Save test image
                filename = f'test_particle_{particle_idx}_radius_{test_radius}.png'
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
    def _debug_rendered_images(self, step, image, out, particle_idx):
        image_stats = {
            'min': image.min().item(),
            'max': image.max().item(), 
            'mean': image.mean().item(),
            'non_white_pixels': (image.sum(dim=1) < 2.9).sum().item(),  # Pixels that aren't white
            'unique_values': len(torch.unique(image.view(-1))),
            'shape': image.shape
        }
        print(f"[DEBUG] Step {step}, Particle {particle_idx}: {image_stats}")
        
        # Additional render output analysis
        if 'depth' in out:
            depth = out['depth']
            print(f"[DEBUG] Step {step}, Particle {particle_idx}: Depth - min={depth.min():.3f}, max={depth.max():.3f}, mean={depth.mean():.3f}")
        if 'alpha' in out:
            alpha = out['alpha'] 
            print(f"[DEBUG] Step {step}, Particle {particle_idx}: Alpha - min={alpha.min():.3f}, max={alpha.max():.3f}, mean={alpha.mean():.3f}")

        