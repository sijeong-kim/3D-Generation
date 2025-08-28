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

        # Validate required parameters
        if opt is None:
            raise ValueError("Configuration object 'opt' cannot be None")
        self.opt = opt
        
        # Renderers
        self.renderers = renderers
        self.cam = cam
        
        # Evaluation parameters
        self.eval_W = getattr(self.opt, 'eval_W', self.opt.W) 
        self.eval_H = getattr(self.opt, 'eval_H', self.opt.H)
        self.eval_radius = getattr(self.opt, 'eval_radius', self.cam.radius) # TODO: remove this
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
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
        if self.opt.visualize_multi_viewpoints:
            self.multi_viewpoints_dir = os.path.join(self.save_dir, "multi_viewpoints")
            os.makedirs(self.multi_viewpoints_dir, exist_ok=True)
        else:
            self.multi_viewpoints_dir = None
        
        # Initialize visualization directory for iid images
        if self.opt.save_iid:
            self.iid_dir = os.path.join(self.save_dir, "iid")
            os.makedirs(self.iid_dir, exist_ok=True)
        else:
            self.iid_dir = None
            
        # Initialize visualization directory for fixed viewpoint
        if self.opt.visualize_fixed_viewpoint:
            self.fixed_viewpoint_dir = os.path.join(self.save_dir, "fixed_viewpoint")
            os.makedirs(self.fixed_viewpoint_dir, exist_ok=True)
        else:
            self.fixed_viewpoint_dir = None
    
    # Set viewpoints around object
    @torch.no_grad()
    def set_viewpoints(self, num_views=8):
        elevations = [0.0] * num_views  # All viewpoints at zero elevation
        azimuths = np.linspace(0, 360, num_views, endpoint=False)  # Evenly distributed azimuth angles
        return [(elevations[i], azimuths[i]) for i in range(num_views)]

    # Save rendered images
    @torch.no_grad()
    def save_rendered_images(self, step, images):
        
        if self.rendered_images_dir is not None:
            output_path = os.path.join(self.rendered_images_dir, f'step_{step}.png')
            vutils.save_image(images, output_path, normalize=False)
        
        
    # Set specific viewpoint camera
    @torch.no_grad()
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
    @torch.no_grad()
    def get_front_view_camera(self):
        """
        Generate a camera viewpoint for front view rendering.
        """
        return self.get_fixed_view_camera(0.0, 0.0, self.eval_radius)

    # Set multi-viewpoints around object
    @torch.no_grad()
    def set_multi_viewpoints_around_object(self, num_views=8):
        elevations = [0.0] * num_views  # All viewpoints at zero elevation
        azimuths = np.linspace(0, 360, num_views, endpoint=False)  # Evenly distributed azimuth angles
        return [(elevations[i], azimuths[i]) for i in range(num_views)]
    

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
    

    @torch.no_grad()
    def visualize_all_particles_in_multi_viewpoints(self, step, num_views=None, visualize_multi_viewpoints=None, save_iid=None):
        # 📥 Use default values if parameters are not provided
        if num_views is None:
            num_views = self.opt.num_views
        if save_iid is None:
            save_iid = self.opt.save_iid
        if visualize_multi_viewpoints is None:
            visualize_multi_viewpoints = self.opt.visualize_multi_viewpoints
        
        # 📥 Create directory for combined particle views
        multi_viewpoints_dir = None
        save_iid_paths = None
        
        if visualize_multi_viewpoints and (step % self.opt.visualize_multi_viewpoints_interval == 0):
            # Create directory for combined particle views
            if self.opt.num_particles > 1:
                multi_viewpoints_dir = os.path.join(self.multi_viewpoints_dir, f'step_{step}_view_{num_views}_all_particles')
                os.makedirs(multi_viewpoints_dir, exist_ok=True)
                
            # Create directories for individual particle views
            if save_iid:
                save_iid_paths = []
                for particle_id in range(self.opt.num_particles):
                    particle_dir = os.path.join(self.iid_dir, f'step_{step}_view_{num_views}_particle_{particle_id}')
                    os.makedirs(particle_dir, exist_ok=True)
                    save_iid_paths.append(particle_dir)
        
        
        # print(f"self.multi_viewpoints_cameras: {self.multi_viewpoints_cameras}")
        
        # Render images from each viewpoint
        multi_viewpoint_images = []
        for i, camera in enumerate(self.multi_viewpoints_cameras):
            # Render each particle from current viewpoint
            particle_images = []
            for particle_id in range(self.opt.num_particles):
                # Render with white background for consistent visualization
                bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
                out = self.renderers[particle_id].render(camera, bg_color=bg_color)
                image = out["image"].unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
                particle_images.append(image)
                
                # 📥 Save individual particle image if enabled
                if visualize_multi_viewpoints and save_iid and (step % self.opt.visualize_multi_viewpoints_interval == 0) and save_iid_paths is not None:
                    output_path = os.path.join(save_iid_paths[particle_id], f'view_{i:03d}.png')
                    vutils.save_image(image, output_path, normalize=False)
            
            # Combine all particle images for current viewpoint
            particle_images = torch.cat(particle_images, dim=0)  # [N, 3, H, W]
            multi_viewpoint_images.append(particle_images)
            
            # 📥 Save combined view of all particles if multiple particles exist
            if visualize_multi_viewpoints and (self.opt.num_particles > 1) and (step % self.opt.visualize_multi_viewpoints_interval == 0) and multi_viewpoints_dir is not None:
                output_path = os.path.join(multi_viewpoints_dir, f'view_{i:03d}.png')
                vutils.save_image(particle_images, output_path, normalize=False)
        
        # Stack all viewpoints into final tensor
        multi_viewpoint_images = torch.stack(multi_viewpoint_images, dim=0)  # [V, N, 3, H, W]
        
        return multi_viewpoint_images

    @torch.no_grad()
    def visualize_fixed_viewpoint(self, step):
        # Create output directory for this specific viewpoint
        if self.fixed_viewpoint_dir is not None:
            viewpoint_dir = os.path.join(self.fixed_viewpoint_dir, f'step_{step}')
            os.makedirs(viewpoint_dir, exist_ok=True)
        
        camera = self.fixed_viewpoint_camera
        
        # Render each particle from the fixed viewpoint
        particle_images = []
        for particle_id in range(self.opt.num_particles):
            # Render with white background for consistent visualization
            bg_color = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            out = self.renderers[particle_id].render(camera, bg_color=bg_color)
            image = out["image"].unsqueeze(0)  # Add batch dimension: [1, 3, H, W]
            particle_images.append(image)
        
            # Save individual particle image if visualization is enabled
            if self.opt.visualize_fixed_viewpoint and self.fixed_viewpoint_dir is not None:
                filename = f'step_{step}_particle_{particle_id}.png'
                output_path = os.path.join(viewpoint_dir, filename)
                vutils.save_image(image, output_path, normalize=False)
                
        # Combine all particle images
        particle_images = torch.cat(particle_images, dim=0)  # [N, 3, H, W]
        
        # Save combined view of all particles if visualization is enabled
        if self.opt.visualize_fixed_viewpoint and self.fixed_viewpoint_dir is not None:
            filename = f'all_particles.png'
            output_path = os.path.join(viewpoint_dir, filename)
            vutils.save_image(particle_images, output_path, normalize=False)
        
        return particle_images
    