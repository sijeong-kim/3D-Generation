# main.py
import numpy as np
import dearpygui.dearpygui as dpg

import torch

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

import random

import subprocess

from visualizer import GaussianVisualizer
from feature_extractor import DINOv2FeatureExtractor
from loss_utils import rbf_kernel_and_grad
from metrics import MetricsCalculator


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    # For better determinism (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_nvidia_smi_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    used, total = result.stdout.strip().split(", ")
    return int(used), int(total)

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None

        self.renderers = [Renderer(sh_degree=self.opt.sh_degree) for _ in range(self.opt.num_particles)]
            
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # # override if provide a checkpoint
        # if self.opt.load is not None: # False
        #     for i in range(self.opt.num_particles):
        #         self.renderers[i].initialize(f"{self.opt.load}_particle_{i}")  # load_path = "path/to/checkpoint"
        # else: # True
        # initialize gaussians to a blob
        base_seed = self.opt.seed if self.opt.seed is not None else 42
        for i in range(self.opt.num_particles):
            # Set different seed for each particle during initialization
            init_seed = base_seed + i * 1000
            torch.manual_seed(init_seed)
            np.random.seed(init_seed)
            random.seed(init_seed)
            
            self.renderers[i].initialize(num_pts=self.opt.num_pts)
            
        # Initialize visualizer and metrics calculator
        # Create visualizer if needed for visualization OR multi-view evaluation
        if self.opt.visualize or self.opt.multi_view_evaluation:
            self.visualizer = GaussianVisualizer(save_dir=self.opt.outdir, opt=self.opt, renderers=self.renderers, cam=self.cam)
        else:
            self.visualizer = None
        
        if self.opt.metrics:
            self.metrics_calculator = MetricsCalculator(save_dir=self.opt.outdir, opt=self.opt, prompt=self.prompt)
        else:
            self.metrics_calculator = None
            
            
    def __del__(self):
        if self.gui:
            dpg.destroy_context()
            
    # Training preparation (O)
    def prepare_train(self):

        self.step = 0
        
        self.optimizers = []
        
        for i in range(self.opt.num_particles):
            # setup training
            self.renderers[i].gaussians.training_setup(self.opt)
            # do not do progressive sh-level
            self.renderers[i].gaussians.active_sh_degree = self.renderers[i].gaussians.max_sh_degree
            
            self.optimizers.append(self.renderers[i].gaussians.optimizer)

        # StableDiffusion
        from sd_utils import StableDiffusion
        self.guidance_sd = StableDiffusion(self.device)
        
        # Feature extractor
        self.feature_extractor = DINOv2FeatureExtractor(self.opt.feature_extractor_model_name).eval().to(self.device)
        
        # prepare embeddings
        with torch.no_grad():
            self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

    def set_camera_pose(self, step_ratio, particle_idx):
        
        ### novel view (manual batch)
        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

        ### a random camera pose per training step ###
        # render random camera pose
        ver = np.random.randint(min_ver, max_ver)
        hor = np.random.randint(-180, 180)
        radius = 0

        pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
        
        # Use single camera object
        cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far) # render_resolution = 128 -> 256 -> 512
        
        return cur_cam
        
    # Training for only one step, which iterate for 500 times (O)
    def render_step_per_particle(self, step_ratio, particle_idx): 
        cur_cam = self.set_camera_pose(step_ratio, particle_idx)
        
        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
        
        # render image for one view
        out = self.renderers[particle_idx].render(cur_cam, bg_color=bg_color) 

        image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
        
        return image, out
        
    def densification_step_per_particle(self, out, particle_idx):
        
        # densify and prune
        if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
            viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
            self.renderers[particle_idx].gaussians.max_radii2D[visibility_filter] = torch.max(self.renderers[particle_idx].gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            assert viewspace_point_tensor.grad is not None, "viewspace_point_tensor.grad is None"
            
            # gradient is available at this point, so we can add densification stats
            self.renderers[particle_idx].gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.step % self.opt.densification_interval == 0:
                self.renderers[particle_idx].gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
            
            if self.step % self.opt.opacity_reset_interval == 0:
                self.renderers[particle_idx].gaussians.reset_opacity()
        
    
    def train(self, iters=500): # iters = 500
        
        self.prepare_train()
        
        # Get base seed for particle-specific seeding
        base_seed = self.opt.seed if self.opt.seed is not None else 42
        
        # Training...
        for i in range(iters):
            
            total_loss = 0
            
            # start timer
            with torch.no_grad():
                if self.opt.metrics and self.metrics_calculator is not None:
                    starter = torch.cuda.Event(enable_timing=True)
                    ender = torch.cuda.Event(enable_timing=True)
                    starter.record()

            ### initialize optimization step ###
            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters) # step_ratio = 0.002

            images = [] # [N, 3, H, W]
            outs = []
            # render images
            for j in range(self.opt.num_particles):
                # Set different seed for each particle to ensure diversity
                particle_seed = base_seed + j * 1000 + self.step  # Unique seed per particle per step
                torch.manual_seed(particle_seed)
                np.random.seed(particle_seed)
                random.seed(particle_seed)
                
                # zero grad
                self.optimizers[j].zero_grad()
                # update lr
                self.renderers[j].gaussians.update_learning_rate(self.step) 
                # render one image
                image, out = self.render_step_per_particle(step_ratio=step_ratio, particle_idx=j)
                images.append(image) # [1, 3, H, W]
                # Store the output for densification after backward pass
                outs.append(out)

            images = torch.cat(images, dim=0) # [N, 3, H, W] (images are normalized)
            
            # Always extract features for metrics calculation
            features = self.feature_extractor(images) # [N, D]
            
            # compute guidance loss
            # attraction_loss = torch.tensor(0.0, device=self.device)
            # repulsion_loss = torch.tensor(0.0, device=self.device)
            
            if self.opt.repulsion_enabled:
                
                # score matching regulizers (sds_loss) - computed without gradients
                if self.opt.repulsion_gradient_type == 'rlsd':  
                    sds_loss, sigma_t = self.guidance_sd.train_step_for_rlsd_repulsion(images, step_ratio=step_ratio if self.opt.anneal_timestep else None) # [1]
                    
                    # 2. RBF kernel gradient wrt features
                    rbf_kernel, rbf_kernel_log_grad = rbf_kernel_and_grad(features, tau=self.opt.repulsion_tau, gradient_type=self.opt.repulsion_gradient_type) # [N, N], [N, D_feature] 
                    
                    # 3. Attraction loss (SDS) + 1/D_latent
                    attraction_loss = sds_loss # [1]
                    
                    # 4. Repulsion loss (Repulsion)
                    repulsion_loss = sigma_t * (rbf_kernel_log_grad * features).sum(dim=1).mean() # [N, D_feature] * [N, D_feature] -> [N] -> [1]
                    # repulsion_loss = sigma_t * torch.einsum('nd,nd->n', rbf_kernel_log_grad, features).mean()
                    
                elif self.opt.repulsion_gradient_type == 'svgd':
                    score_gradients, sigma_t = self.guidance_sd.train_step_for_svgd_repulsion(images, step_ratio=step_ratio if self.opt.anneal_timestep else None) # [N, D_latent], [1]
                    
                    # 2. RBF kernel computation
                    rbf_kernel, rbf_kernel_grad = rbf_kernel_and_grad(features, tau=self.opt.repulsion_tau, gradient_type=self.opt.repulsion_gradient_type) # [N, N], [N, D_feature] 
                    
                    # 3. Attraction loss: 1/N * sum_j k(xj, xi) * ∇_xj log p(xj)
                    # rbf_kernel: [N, N] where rbf_kernel[i,j] = k(xj, xi)
                    # score_gradients: [N] where score_gradients[j] = ∇_xj log p(xj)
                    # Multiply kernel matrix with score gradients: [N, N] * [N] -> [N] (sum over j for each i)
                    # 1/D_latent
                    attraction_per_particle = (rbf_kernel * score_gradients.unsqueeze(0)).sum(dim=1) # [N]
                    attraction_loss = attraction_per_particle.mean() # [1]
                    
                    # # 3. Attraction loss: 1/N * sum_j k(xj, xi) * ∇_xj log p(xj)
                    # weighted_grad = rbf_kernel @ score_gradients  # [N, D_latent]
                    # attraction_loss = weighted_grad.norm(dim=1).mean()  # optional scalar for loss tracking
                    
                    # 4. Repulsion loss (same as before)
                    repulsion_loss = sigma_t * (rbf_kernel_grad * features).sum(dim=1).mean() # [N, D_feature] * [N, D_feature] -> [N] -> [1]
                    # [Question] should we divide by num_particles (mean)? in SVGD paper, they did divide by num_particles (mean)                              
        
                scaled_repulsion_loss = self.opt.lambda_repulsion * repulsion_loss
                total_loss = self.opt.lambda_sds * attraction_loss - scaled_repulsion_loss

            else: # baseline
                total_loss = self.opt.lambda_sds * self.guidance_sd.train_step_for_baseline(images, step_ratio=step_ratio if self.opt.anneal_timestep else None) # [1] -> [1]

            # Backward pass
            total_loss.backward()
            
            # Save rendered images (every visualize_interval steps)
            with torch.no_grad():
                if self.opt.visualize and self.visualizer is not None and self.step % self.opt.vis_interval == 0:
                    self.visualizer.save_rendered_images_in_one_step(images, step=self.step)

            # Densification step after backward pass (gradients are now available)
            for j in range(self.opt.num_particles):
                self.densification_step_per_particle(outs[j], particle_idx=j)
                
            # Step all optimizers
            for j in range(self.opt.num_particles):
                self.optimizers[j].step()
            
            ### end of optimization step ###
            with torch.no_grad():
                if self.opt.metrics and self.metrics_calculator is not None:
                    
                    # efficiency
                    ender.record()
                    torch.cuda.synchronize()
                    t = starter.elapsed_time(ender)
                    
                    # memory usage
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
                    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
                    
                    # Always calculate basic metrics for early stopping
                    attraction_loss_val = attraction_loss.item() if self.opt.repulsion_enabled else 0
                    repulsion_loss_val = repulsion_loss.item() if self.opt.repulsion_enabled else 0
                    scaled_repulsion_loss_val = repulsion_loss_val * self.opt.lambda_repulsion if self.opt.repulsion_enabled else 0
                    total_loss_val = total_loss.item()
                    
                    # diversity
                    if self.opt.multi_view_evaluation: # clip similarity-based best-view selection
                        # TODO: render multi-viewpoint images (let's use 4 viewpoints for now)
                        multi_viewpoint_images = self.visualizer.visualize_all_particles_multi_viewpoints(step=self.step, viewpoint_names=['front', 'back', 'left', 'right']) # [N, V, 3, H, W]
                        representative_images, clip_similarities = self.metrics_calculator.select_best_views_by_clip_fidelity(multi_viewpoint_images) # [N, 3, H, W], [N]
                        # OPTION: visualize representative_images
                        fidelity = clip_similarities.mean().item() # [1]
                        features = self.feature_extractor(representative_images) # [N, D]
                        diversity = self.metrics_calculator.compute_rlsd_diversity(features) # [1]
                    else:
                        # TODO: single viewpoint images rendering
                        fidelity = self.metrics_calculator.compute_clip_fidelity(images)
                        # TODO: extract features from single viewpoint images
                        diversity = self.metrics_calculator.compute_rlsd_diversity(features) # [1]
                        
                    # log
                    self.metrics_calculator.log_metrics(step=self.step, diversity=diversity, fidelity=fidelity, attraction_loss=attraction_loss_val, repulsion_loss=repulsion_loss_val, scaled_repulsion_loss = scaled_repulsion_loss_val, total_loss=total_loss_val, time=t, memory_allocated_mb=memory_allocated, max_memory_allocated_mb=max_memory_allocated)
    
        # visualize
        if self.opt.visualize and self.visualizer is not None and self.step % self.opt.vis_interval == 0 and self.step >= 500:
            self.visualizer.visualize_all_particles_360(step=self.step)
            # Render specific viewpoints at the end of training
            # self.visualizer.render_specific_viewpoints(step=self.step, viewpoint_names=['front', 'back', 'left', 'right'])
            
        # do a last prune
        for j in range(self.opt.num_particles):
            self.renderers[j].gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
            
        # if self.opt.save_model:
        #     save_model = SaveModel(self.opt, self.renderers, self.cam)
        #     save_model.save_model(mode='geo', texture_size=self.opt.save_model_texture_size)
        #     save_model.save_model(mode='geo+tex', texture_size=self.opt.save_model_texture_size)


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    # Set seed for reproducibility
    if opt.seed is not None:    
        seed_everything(opt.seed)
        print(f"[INFO] Set random seed to {opt.seed}")
        
    gui = GUI(opt)
    gui.train(opt.iters)
