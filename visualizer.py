# visualizer.py
import os
from tkinter import FALSE
import numpy as np
import torch
from gs_renderer import Renderer, MiniCam
from cam_utils import OrbitCamera, orbit_camera
import torchvision.utils as vutils
import matplotlib.pyplot as plt

class GaussianVisualizer:
    
    def __init__(self, opt=None, renderers=None, cam=None, debug=False):

        # Validate required parameters
        if opt is None:
            raise ValueError("Configuration object 'opt' cannot be None")
        self.opt = opt
        
        # Create copies of renderers for visualization to avoid affecting training
        self.renderers = []
        for renderer in renderers:
            # Create a deep copy of the renderer for visualization
            import copy
            renderer_copy = copy.deepcopy(renderer)
            self.renderers.append(renderer_copy)
        
        self.cam = cam
        
        # Evaluation parameters
        self.eval_W = getattr(self.opt, 'eval_W', self.opt.W) 
        self.eval_H = getattr(self.opt, 'eval_H', self.opt.H)
        self.eval_radius = getattr(self.opt, 'eval_radius', self.cam.radius) # TODO: remove this
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Grid layout parameters
        self.grid_rows = 2  # Default to 2 rows
        self.grid_cols = 4  # Default to 4 columns
        
        # Auto-adjust grid layout based on number of particles
        self.auto_adjust_grid = True
        
        
        ### Pre-compute camera positions for efficient rendering
        if self.opt.visualize_multi_viewpoints or self.opt.metrics:
            self.multi_viewpoints_cameras = self.get_multi_view_cameras(self.opt.num_views)
        else:
            self.multi_viewpoints_cameras = None
            
        if self.opt.visualize_fixed_viewpoint:
            self.fixed_viewpoint_camera = self.get_front_view_camera()
        else:
            self.fixed_viewpoint_camera = None
        
        
        ### Initialize visualization directories
        # Initialize visualization directory for all images
        self.save_dir = os.path.join(self.opt.outdir, 'visualizations')
        if self.opt.visualize:
            os.makedirs(self.save_dir, exist_ok=True)
            
        # Initialize visualization directory for rendered images
        if self.opt.save_rendered_images:
            self.rendered_images_dir = os.path.join(self.save_dir, "rendered_images")
            os.makedirs(self.rendered_images_dir, exist_ok=True)
        else:
            self.rendered_images_dir = None
            
        # Initialize visualization directory for multi-viewpoints
        if self.opt.visualize_multi_viewpoints or self.opt.video_snapshot:
            self.multi_viewpoints_dir = os.path.join(self.save_dir, "multi_viewpoints")
            os.makedirs(self.multi_viewpoints_dir, exist_ok=True)
        else:
            self.multi_viewpoints_dir = None
        
        # Initialize visualization directory for iid images
        # if self.opt.save_iid or :
        #     self.iid_dir = os.path.join(self.save_dir, "iid")
        #     os.makedirs(self.iid_dir, exist_ok=True)
        # else:
        #     self.iid_dir = None
            
        # Initialize visualization directory for fixed viewpoint
        if self.opt.visualize_fixed_viewpoint:
            self.fixed_viewpoint_dir = os.path.join(self.save_dir, "fixed_viewpoint")
            os.makedirs(self.fixed_viewpoint_dir, exist_ok=True)
        else:
            self.fixed_viewpoint_dir = None
            
        self.white_background_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
    
    def update_renderers(self, training_renderers):
        """
        Update visualizer's renderers with the latest training state.
        This ensures visualization shows the current training progress.
        Properly discards old copies to prevent memory leaks.
        """
        import copy
        import gc
        
        # Explicitly delete old renderers to free memory
        if hasattr(self, 'renderers') and self.renderers:
            for old_renderer in self.renderers:
                del old_renderer
            del self.renderers
        
        # Force garbage collection
        gc.collect()
        
        # Create new copies
        self.renderers = []
        for renderer in training_renderers:
            renderer_copy = copy.deepcopy(renderer)
            self.renderers.append(renderer_copy)
    
    def cleanup_renderers(self):
        """
        Explicitly clean up visualizer renderers to free memory.
        Call this after visualization is complete.
        """
        import gc
        
        if hasattr(self, 'renderers') and self.renderers:
            for renderer in self.renderers:
                del renderer
            del self.renderers
            self.renderers = []
        
        # Force garbage collection
        gc.collect()
    
    @torch.inference_mode()
    def arrange_particles_in_grid(self, particle_images):
        """
        Arrange particle images in a grid layout (2x4 by default, auto-adjusts for different numbers).
        
        Args:
            particle_images: List of tensors [N, 3, H, W] where N is number of particles
            
        Returns:
            Grid tensor [3, grid_rows*H, grid_cols*W]
        """
        if not particle_images:
            return None
            
        num_particles = len(particle_images)
        if num_particles == 0:
            return None
            
        # Get image dimensions from first particle
        _, channels, height, width = particle_images[0].shape
        
        # Calculate grid dimensions
        if self.auto_adjust_grid:
            # Auto-adjust grid to be as square as possible
            if num_particles <= 4:
                grid_rows = 2
                grid_cols = 2
            elif num_particles <= 6:
                grid_rows = 2
                grid_cols = 3
            elif num_particles <= 8:
                grid_rows = 2
                grid_cols = 4
            elif num_particles <= 9:
                grid_rows = 3
                grid_cols = 3
            elif num_particles <= 12:
                grid_rows = 3
                grid_cols = 4
            else:
                # For more than 12 particles, use a reasonable default
                grid_rows = 3
                grid_cols = 4
        else:
            # Use fixed grid dimensions
            grid_rows = min(self.grid_rows, num_particles)
            grid_cols = min(self.grid_cols, (num_particles + grid_rows - 1) // grid_rows)
        
        # Create grid tensor
        grid_height = grid_rows * height
        grid_width = grid_cols * width
        grid_tensor = torch.zeros(channels, grid_height, grid_width, device=particle_images[0].device)
        
        # Place each particle in the grid
        for i, particle_img in enumerate(particle_images):
            if i >= grid_rows * grid_cols:
                break
                
            row = i // grid_cols
            col = i % grid_cols
            
            start_h = row * height
            end_h = start_h + height
            start_w = col * width
            end_w = start_w + width
            
            # Place the particle image in the grid
            grid_tensor[:, start_h:end_h, start_w:end_w] = particle_img[0]  # Remove batch dimension
        
        return grid_tensor
    
    # Set viewpoints around object
    @torch.inference_mode()
    def set_viewpoints(self, num_views=8):
        elevations = [0.0] * num_views  # All viewpoints at zero elevation
        azimuths = np.linspace(0, 360, num_views, endpoint=False)  # Evenly distributed azimuth angles
        return [(elevations[i], azimuths[i]) for i in range(num_views)]

    # Save rendered images
    @torch.inference_mode()
    def save_rendered_images(self, step, images):
        """
        Save rendered images for visualization only. This method does NOT affect training.
        """
        if self.rendered_images_dir is not None:
            output_path = os.path.join(self.rendered_images_dir, f'step_{step}.png')
            # Ensure images are in [0,1] range before saving
            vutils.save_image(images, output_path, normalize=False)
        
        
    # Set specific viewpoint camera
    @torch.inference_mode()
    def get_fixed_view_camera(self, elevation, horizontal, radius):
        """
        Compute camera pose for specified angles
        """
        pose = orbit_camera(elevation, horizontal, radius)
        
        # Create camera object with computed pose
        camera = MiniCam(
            pose,
            self.eval_W,
            self.eval_H,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far
        )
        return camera
    
    # Set front view camera
    @torch.inference_mode()
    def get_front_view_camera(self):
        """
        Generate a camera viewpoint for front view rendering.
        """
        return self.get_fixed_view_camera(0.0, 0.0, self.eval_radius)

    # Set multi-viewpoints around object
    @torch.inference_mode()
    def set_multi_viewpoints_around_object(self, num_views=8):
        elevations = [0.0] * num_views  # All viewpoints at zero elevation
        azimuths = np.linspace(0, 360, num_views, endpoint=False)  # Evenly distributed azimuth angles
        return [(elevations[i], azimuths[i]) for i in range(num_views)]
    
    @torch.inference_mode()
    def get_multi_view_cameras(self, num_views=8):
        """
        Generate a set of camera viewpoints for multi-view rendering.
        """
        viewpoints = self.set_multi_viewpoints_around_object(num_views)
        cameras = []
        for elevation, azimuth in viewpoints:
            camera = self.get_fixed_view_camera(elevation, azimuth, self.eval_radius)
            cameras.append(camera)
            
        return cameras
    

    @torch.inference_mode()
    def visualize_all_particles_in_multi_viewpoints(self, step, num_views=None, visualize_multi_viewpoints=None, save_iid=None):
        # 📥 Use default values if parameters are not provided
        if num_views is None:
            num_views = self.opt.num_views
            multi_viewpoints_cameras = self.multi_viewpoints_cameras
        else:
            multi_viewpoints_cameras = self.get_multi_view_cameras(num_views)

        if visualize_multi_viewpoints is None:
            visualize_multi_viewpoints = self.opt.visualize_multi_viewpoints
        
        # 📥 Create directory for combined particle views
        multi_viewpoints_dir = None
        save_iid_paths = None
        
        if visualize_multi_viewpoints and (step==1 or step % self.opt.visualize_multi_viewpoints_interval == 0):
            # Create directory for combined particle views
            multi_viewpoints_dir = os.path.join(self.multi_viewpoints_dir, f'step_{step}_view_{num_views}_all_particles')
            os.makedirs(multi_viewpoints_dir, exist_ok=True)
                
            # Create directories for individual particle views
            if save_iid:
                save_iid_dir = os.path.join(self.multi_viewpoints_dir, f'step_{step}_view_{num_views}_iid_particles')
                os.makedirs(save_iid_dir, exist_ok=True)
                save_iid_paths = []
                for particle_id in range(self.opt.num_particles):
                    particle_dir = os.path.join(save_iid_dir, f'particle_{particle_id}')
                    os.makedirs(particle_dir, exist_ok=True)
                    save_iid_paths.append(particle_dir)
        
        # Render images from each viewpoint
        multi_viewpoint_images = []
        
        bg_color = self.white_background_color
        
        for i, camera in enumerate(multi_viewpoints_cameras):
            # Render each particle from current viewpoint
            particle_images = []
            
            
            for particle_id in range(self.opt.num_particles):
                # Render with white background for consistent visualization    
                out = self.renderers[particle_id].render(camera, bg_color=bg_color)
                image = out["image"].unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
                particle_images.append(image)
                
                # 📥 Save individual particle image if enabled
                if self.opt.video_snapshot or (step==1 or step % self.opt.visualize_multi_viewpoints_interval == 0):
                    if visualize_multi_viewpoints and save_iid and save_iid_paths is not None:
                        output_path = os.path.join(save_iid_paths[particle_id], f'view_{i:03d}.png')
                        # Ensure images are in [0,1] range before saving
                        vutils.save_image(image, output_path, normalize=False)
            
            # Combine all particle images for current viewpoint using grid layout
            particle_images_tensor = torch.cat(particle_images, dim=0)  # [N, 3, H, W] - keep for metrics
            multi_viewpoint_images.append(particle_images_tensor)
            
            # Create grid layout for visualization
            grid_image = self.arrange_particles_in_grid(particle_images)
            
            # 📥 Save combined view of all particles if multiple particles exist
            if self.opt.video_snapshot or (step==1 or step % self.opt.visualize_multi_viewpoints_interval == 0):
                if visualize_multi_viewpoints and multi_viewpoints_dir is not None:
                    output_path = os.path.join(multi_viewpoints_dir, f'view_{i:03d}.png')
                    # Ensure images are in [0,1] range before saving
                    vutils.save_image(grid_image, output_path, normalize=False)
            
        # Stack all viewpoints into final tensor
        multi_viewpoint_images = torch.stack(multi_viewpoint_images, dim=0)  # [V, N, 3, H, W]

        
        return multi_viewpoint_images

    @torch.inference_mode()
    def visualize_fixed_viewpoint(self, step, save_iid=None):
        if save_iid is None:
            save_iid = self.opt.save_iid
            
        # # Create output directory for this specific viewpoint
        # if self.fixed_viewpoint_dir is not None:
        #     viewpoint_dir = os.path.join(self.fixed_viewpoint_dir, f'step_{step}')
        #     os.makedirs(viewpoint_dir, exist_ok=True)
        
        camera = self.fixed_viewpoint_camera
        bg_color = self.white_background_color
        
        # Render each particle from the fixed viewpoint
        particle_images = []
        for particle_id in range(self.opt.num_particles):
            # Render with white background for consistent visualization
            
            out = self.renderers[particle_id].render(camera, bg_color=bg_color)
            image = out["image"].unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
            particle_images.append(image)
        
            # Save individual particle image if visualization is enabled
            if self.opt.visualize_fixed_viewpoint and self.fixed_viewpoint_dir is not None and save_iid:
                output_path = os.path.join(self.fixed_viewpoint_dir, f'step_{step}_particle_{particle_id}.png')
                # Ensure images are in [0,1] range before saving

                vutils.save_image(image, output_path, normalize=False)
                
        del image
        del out
        
        # Combine all particle images using grid layout
        particle_images_tensor = torch.cat(particle_images, dim=0)  # [N, 3, H, W] - keep for return
        grid_image = self.arrange_particles_in_grid(particle_images)
        
        # Save combined view of all particles if visualization is enabled
        if self.opt.visualize_fixed_viewpoint and self.fixed_viewpoint_dir is not None:
            output_path = os.path.join(self.fixed_viewpoint_dir, f'step_{step}_all_particles.png')
            # Ensure images are in [0,1] range before saving

            vutils.save_image(grid_image, output_path, normalize=False)
        
        del particle_images
        
        return particle_images_tensor
    