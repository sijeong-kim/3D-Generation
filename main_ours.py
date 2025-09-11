import os
import cv2
import time
import gc
import tqdm
import numpy as np
import random
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
from guidance.sd_utils import StableDiffusion

from visualizer import GaussianVisualizer
from metrics import MetricsCalculator
from feature_extractor import DINOv2MultiLayerFeatureExtractor


from kernels import rbf_kernel_and_grad, cosine_kernel_and_grad

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        # self.seed = "random"
        self.seed = opt.seed

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        
        # Set CUDA device if specified in options
        if hasattr(opt, 'gpu_id') and opt.gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_id)
            self.device = torch.device(f"cuda:{opt.gpu_id}")
            print(f"[INFO] Using GPU {opt.gpu_id}: {torch.cuda.get_device_name(opt.gpu_id)}")
        
        
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        # self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.renderers = []
        for i in range(self.opt.num_particles):
            self.renderers.append(Renderer(sh_degree=self.opt.sh_degree))
            

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

        self.seeds = []
        # override if provide a checkpoint
        for i in range(self.opt.num_particles):
            # Set different seed for each particle during initialization
            init_seed = self.seed + i * self.opt.iters
            self.seeds.append(init_seed)
            self.seed_everything(init_seed)
            
            if self.opt.load is not None:
                self.renderers[i].initialize(self.opt.load) # TODO: load from different checkpoints for each particle
            else:
                # initialize gaussians to a blob
                self.renderers[i].initialize(num_pts=self.opt.num_pts)
                
        # visualizer
        if self.opt.visualize or self.opt.metrics:
            self.visualizer = GaussianVisualizer(opt=self.opt, renderers=self.renderers, cam=self.cam)
        else:
            self.visualizer = None
            
    def __del__(self):
        pass
    
    def cleanup_gpu_resources(self):
        """Clean up GPU resources and free memory."""
        print("[INFO] Cleaning up GPU resources...")
        
        # Clean up renderers
        if hasattr(self, 'renderers'):
            for renderer in self.renderers:
                if hasattr(renderer, 'gaussians'):
                    del renderer.gaussians
                del renderer
            self.renderers.clear()
        
        # Clean up visualizer
        if hasattr(self, 'visualizer'):
            del self.visualizer
            self.visualizer = None
        
        # Clean up feature extractor
        if hasattr(self, 'feature_extractor'):
            del self.feature_extractor
            self.feature_extractor = None
        
        # Clean up guidance models
        if hasattr(self, 'guidance_sd'):
            del self.guidance_sd
            self.guidance_sd = None
        
        # Clean up metrics calculator
        if hasattr(self, 'metrics_calculator'):
            del self.metrics_calculator
            self.metrics_calculator = None
        
        # Clean up optimizers
        if hasattr(self, 'optimizers'):
            for optimizer in self.optimizers:
                del optimizer
            self.optimizers.clear()
        
        # Force garbage collection and CUDA cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        print("[INFO] GPU resources cleaned up successfully.")
    
    def set_gpu_device(self, gpu_id: int):
        """Set the CUDA device for this experiment."""
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
            print(f"[INFO] Switched to GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            return True
        else:
            print(f"[WARNING] GPU {gpu_id} not available, using CPU")
            self.device = torch.device("cpu")
            return False

    def seed_everything(self, seed):
        try:
            seed = int(seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def prepare_train(self):

        self.step = 0
        self.optimizers = []
        for i in range(self.opt.num_particles):
            self.renderers[i].gaussians.training_setup(self.opt)
            self.renderers[i].gaussians.active_sh_degree = self.renderers[i].gaussians.max_sh_degree
            self.optimizers.append(self.renderers[i].gaussians.optimizer)

        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        
        if self.opt.repulsion_type in ['rlsd', 'svgd']:
            # feature extractor
            print(f"[INFO] Using DINOv2 features from '{self.opt.feature_layer}' layer")
            self.feature_extractor = DINOv2MultiLayerFeatureExtractor(
                model_name=self.opt.feature_extractor_model_name, 
                device=self.device
            )
            
            # Freeze feature extractor weights
            self.feature_extractor.model.eval()
            for param in self.feature_extractor.model.parameters():
                param.requires_grad = False    
            
            
            # Make sure the feature layer is an integer and convert str to int if necessary
            n_layers = len(self.feature_extractor.model.encoder.layer)
            fl = self.opt.feature_layer
            if isinstance(fl, str):
                fl = fl.lower()
                if fl == 'early':
                    self.opt.feature_layer = max(0, int(0.25 * n_layers) - 1) # 12 / 4 - 1 = 2 -> 2
                elif fl == 'mid':
                    self.opt.feature_layer = max(0, int(0.50 * n_layers) - 1) # 12 / 2 - 1 = 5 -> 5
                elif fl in ['last', 'final', 'late']:
                    self.opt.feature_layer = n_layers - 1 # 12 - 1 = 11 -> 11
                else:
                    raise ValueError(f"Unknown feature_layer '{self.opt.feature_layer}' — use early|mid|last or an integer")
                
        # metrics
        if self.opt.metrics:
            self.metrics_calculator = MetricsCalculator(opt=self.opt, prompt=self.prompt, device=self.device.type)
        else:
            self.metrics_calculator = None

        print(f"[INFO] loading SD...")
        self.guidance_sd = StableDiffusion(self.device)
        
        # Freeze all model weights
        for module in [self.guidance_sd.vae, self.guidance_sd.text_encoder, self.guidance_sd.unet]:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
                
        print(f"[INFO] loaded SD!")

        # prepare embeddings
        with torch.no_grad():
            self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

    def train_step(self):
    # 1. Forward pass
        self.step += 1
        
        with torch.no_grad():
            if self.step==1 or self.step % self.opt.efficiency_interval == 0 and self.opt.metrics and self.metrics_calculator is not None:
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                starter.record()
                

        step_ratio = min(1, self.step / self.opt.iters)

        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        attraction_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        repulsion_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        ### novel view (manual batch)
        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        images = []
        outputs = []
        # poses = []
        # vers, hors, radii = [], [], []
        # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
        min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)
        
        # render random view
        
        self.seed_everything(self.seed + self.step)
        
        ver = np.random.randint(min_ver, max_ver)
        hor = np.random.randint(-180, 180)
        radius = 0
        
        pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)        
        cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")

        for j in range(self.opt.num_particles):
            
            # set seed for each particle + iteration step for different viewpoints each iter
            # self.seed_everything(self.seeds[j] + self.step)
            # update lr
            self.renderers[j].gaussians.update_learning_rate(self.step)

            # vers.append(ver)
            # hors.append(hor)
            # radii.append(radius)
            # poses.append(pose)


            out = self.renderers[j].render(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
            images.append(image)
            
            # Store output for each particle
            outputs.append(out)
                
        images = torch.cat(images, dim=0) # [N, 3, H, W]
        # poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device) # TODO: Check if this is correct
        
        repulsion_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        # repulsion loss (feature space)
        if self.opt.repulsion_type in ['rlsd', 'svgd']:
            features = self.feature_extractor.extract_cls_from_layer(self.opt.feature_layer, images).to(self.device).to(torch.float32)   # [N, D_feature] 
            
            # 4. Kernel computation (feature space)
            if self.opt.kernel_type == 'rbf':
                kernel, kernel_grad = rbf_kernel_and_grad(
                    features, 
                    repulsion_type=self.opt.repulsion_type, 
                    beta=self.opt.rbf_beta,
                )  # kernel:[N,N], kernel_grad:[N, D_feature]
            elif self.opt.kernel_type == 'cosine':
                kernel, kernel_grad = cosine_kernel_and_grad(
                    features, 
                    repulsion_type=self.opt.repulsion_type,
                    beta=self.opt.cosine_beta,
                    eps_shift=self.opt.cosine_eps_shift,
                )  # kernel:[N,N], kernel_grad:[N, D_feature]
            else:
                raise ValueError(f"Invalid kernel type: {self.opt.kernel_type}")
            
            kernel = kernel.detach().to(torch.float32)
            kernel_grad = kernel_grad.detach().to(torch.float32)
            
            repulsion_loss = (kernel_grad * features).sum(dim=1) # [N, D_feature] * [N, D_feature] -> [N]
            repulsion_loss = repulsion_loss.mean() # [N] -> [1]

        # attraction loss (latent space)
        attraction_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
    
        score_gradients, latents = self.guidance_sd.train_step_gradient(
            images, step_ratio=step_ratio if self.opt.anneal_timestep else None, 
            guidance_scale=self.opt.guidance_scale,
            force_same_t=self.opt.force_same_t,
            force_same_noise=self.opt.force_same_noise,
        )  # score_gradients: [N, 4, 64, 64], latents: [N, 4, 64, 64]
        
        score_gradients = score_gradients.to(torch.float32)
        latents = latents.to(torch.float32)
        
        if self.opt.repulsion_type == 'svgd':
            v = torch.einsum('ij,jchw->ichw', kernel.detach(), score_gradients)  # [N,4,64,64]  
            target = (latents - v).detach()
        elif self.opt.repulsion_type in ['rlsd', 'wo']:
            target = (latents - score_gradients).detach()
        else:
            raise ValueError(f"Invalid repulsion type: {self.opt.repulsion_type}")
            
        attraction_loss = 0.5 * F.mse_loss(latents, target, reduction='none').view(self.opt.num_particles, -1).sum(dim=1)  # [N]
        attraction_loss = attraction_loss.mean()
            
        # elif self.opt.repulsion_type in ['rlsd', 'wo']:
        #     for j in range(self.opt.num_particles):
        #         sds_loss = self.guidance_sd.train_step(images[j:j+1], step_ratio=step_ratio if self.opt.anneal_timestep else None)
        #         attraction_loss = attraction_loss + sds_loss
            

        ### optimize step ### 
        
        scaled_attraction_loss = self.opt.lambda_sd * attraction_loss
        scaled_repulsion_loss = self.opt.lambda_repulsion * repulsion_loss
        total_loss = scaled_attraction_loss + scaled_repulsion_loss
        
        # 2. Backward pass (Compute gradients)
        total_loss.backward()
        
        # 3. Optimize step (Update parameters)
        for j in range(self.opt.num_particles):
            self.optimizers[j].step()

        # densify and prune (after backward pass so gradients are available)
        for j in range(self.opt.num_particles):
            if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = outputs[j]["viewspace_points"], outputs[j]["visibility_filter"], outputs[j]["radii"]
                self.renderers[j].gaussians.max_radii2D[visibility_filter] = torch.max(self.renderers[j].gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.renderers[j].gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                if self.step % self.opt.densification_interval == 0:
                        self.renderers[j].gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                
                if self.step % self.opt.opacity_reset_interval == 0:
                        self.renderers[j].gaussians.reset_opacity()
                        
        
        # 4. Zero gradients (Prepare for next iteration)
        for j in range(self.opt.num_particles):
            self.optimizers[j].zero_grad()
        

        #########################################################
        # Log metrics and visualize
        #########################################################
        with torch.no_grad():
            if self.opt.metrics and self.metrics_calculator is not None:
                # time
                if self.step==1 or self.step % self.opt.efficiency_interval == 0:
                    ender.record()
                    torch.cuda.synchronize()
                    t = starter.elapsed_time(ender)
                    
                    # memory usage
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
                    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
                    
                    # log
                    self.metrics_calculator.log_efficiency(
                        step=self.step,
                        efficiency= {
                            "time": t,
                            "memory_allocated_mb": memory_allocated,
                            "max_memory_allocated_mb": max_memory_allocated,
                        },
                    )
                    
                # losses
                if self.step==1 or self.step % self.opt.losses_interval == 0:
                    attraction_loss_val = attraction_loss.item()
                    repulsion_loss_val = repulsion_loss.item()
                    scaled_attraction_loss_val = self.opt.lambda_sd * attraction_loss_val
                    scaled_repulsion_loss_val = self.opt.lambda_repulsion * repulsion_loss_val
                    total_loss_val = total_loss.item()
                    
                    # log
                    self.metrics_calculator.log_losses(
                        step=self.step,
                        losses= {
                            "attraction_loss": attraction_loss_val,
                            "repulsion_loss": repulsion_loss_val,
                            "scaled_attraction_loss": scaled_attraction_loss_val,
                            "scaled_repulsion_loss": scaled_repulsion_loss_val,
                            "total_loss": total_loss_val,
                            "scaled_repulsion_loss_ratio": abs(scaled_repulsion_loss_val / scaled_attraction_loss_val) * 100,
                        },
                    )
            
                # quantitative metrics
                if self.step==1 or self.step % self.opt.quantitative_metrics_interval == 0 and self.visualizer is not None:
                    # Update visualizer with current training state
                    self.visualizer.update_renderers(self.renderers)
                    multi_view_images = self.visualizer.visualize_all_particles_in_multi_viewpoints(self.step, visualize_multi_viewpoints=self.opt.visualize_multi_viewpoints, save_iid=self.opt.save_iid)  # [V, N, 3, H, W]
                    # Clean up visualizer renderers to free memory
                    self.visualizer.cleanup_renderers()
                    # fidelity
                    fidelity_mean, fidelity_std = self.metrics_calculator.compute_clip_fidelity_in_multi_viewpoints_stats(multi_view_images)
                    # compute inter-particle diversity 
                    inter_particle_diversity_mean, inter_particle_diversity_std = self.metrics_calculator.compute_inter_particle_diversity_in_multi_viewpoints_stats(multi_view_images)
                    # Compute cross-view consistency
                    cross_view_consistency_mean, cross_view_consistency_std = self.metrics_calculator.compute_cross_view_consistency_stats(multi_view_images)
                    if self.opt.enable_lpips:
                        # LPIPS (inter-sample and cross-view consistency)
                        lpips_inter_mean, lpips_inter_std, lpips_consistency_mean, lpips_consistency_std = self.metrics_calculator.compute_lpips_inter_and_consistency(multi_view_images)
                        
                    # log
                    self.metrics_calculator.log_quantitative_metrics(
                        step=self.step,
                        metrics= {
                            "fidelity_mean": fidelity_mean,
                            "fidelity_std": fidelity_std,
                            "inter_particle_diversity_mean": inter_particle_diversity_mean,
                            "inter_particle_diversity_std": inter_particle_diversity_std,
                            "cross_view_consistency_mean": cross_view_consistency_mean,
                            "cross_view_consistency_std": cross_view_consistency_std,
                            # LPIPS
                            "lpips_inter_mean": lpips_inter_mean if self.opt.enable_lpips else None,
                            "lpips_inter_std": lpips_inter_std if self.opt.enable_lpips else None,
                            "lpips_consistency_mean": lpips_consistency_mean if self.opt.enable_lpips else None,
                            "lpips_consistency_std": lpips_consistency_std if self.opt.enable_lpips else None,
                        }
                    )
                    
            # visualize
            if self.opt.visualize and self.visualizer is not None:
                # save rendered images (save at the end of each interval)
                if self.opt.save_rendered_images and (self.step==1 or self.step % self.opt.save_rendered_images_interval == 0):
                    self.visualizer.update_renderers(self.renderers)
                    self.visualizer.save_rendered_images(self.step, images)
                    self.visualizer.cleanup_renderers()
                    
                if self.opt.visualize_fixed_viewpoint and (self.step==1 or self.step % self.opt.visualize_fixed_viewpoint_interval == 0):
                    self.visualizer.update_renderers(self.renderers)
                    self.visualizer.visualize_fixed_viewpoint(self.step)
                    self.visualizer.cleanup_renderers()
                
                # Clean up visualizer renderers to free memory
                
                
                
        # Periodic GPU memory cleanup
        if self.step % self.opt.efficiency_interval == 0:
            try:
                del images
            except NameError:
                pass
            try:
                del outputs
            except NameError:
                pass
            try:
                del score_gradients
            except NameError:
                pass
            try:
                del latents
            except NameError:
                pass
            try:
                del features
            except NameError:
                pass
            try:
                del kernel
            except NameError:
                pass
            try:
                del kernel_grad
            except NameError:
                pass
            try:
                del scaled_attraction_loss
            except NameError:
                pass
            try:
                del scaled_repulsion_loss
            except NameError:
                pass
            try:
                del total_loss
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, particle_id=0):
        path = os.path.join(self.opt.outdir, f'saved_models')
        os.makedirs(path, exist_ok=True)
        
        if mode == 'geo':
            path = os.path.join(path, f'particle_{particle_id}_mesh.ply')
            mesh = self.renderers[particle_id].gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)
            # Cleanup heavy objects and CUDA cache
            try:
                del mesh
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elif mode == 'geo+tex':
            path = os.path.join(path, f'particle_{particle_id}_mesh.' + self.opt.mesh_format)
            mesh = self.renderers[particle_id].gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] particle {particle_id} unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                
                cur_out = self.renderers[particle_id].render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    
                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(self.device)

                v_cam = torch.matmul(F.pad(mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(glctx, v_clip, mesh.f, (render_resolution, render_resolution))

                depth, _ = dr.interpolate(-v_cam[..., [2]], rast, mesh.f) # [1, H, W, 1]
                depth = depth.squeeze(0) # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(mesh.vt.unsqueeze(0), rast, mesh.ft)  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn)
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()
                
                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h, w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )
                
                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[tuple(search_coords[indices[:, 0]].T)]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

            # Cleanup heavy objects and CUDA cache
            try:
                del albedo
            except NameError:
                pass
            try:
                del cnt
            except NameError:
                pass
            try:
                del mesh
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            path = os.path.join(path, f'particle_{particle_id}_model.ply')
            self.renderers[particle_id].gaussians.save_ply(path)

            print(f"[INFO] particle {particle_id} save model to {path}.")
            # Cleanup CUDA cache after saving
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            for j in range(self.opt.num_particles):
                self.renderers[j].gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
    
        # Multi-viewpoints for 30 fps video (save at the end of training)
        if self.opt.video_snapshot:
            # Update visualizer with final training state
            self.visualizer.update_renderers(self.renderers)
            self.visualizer.visualize_all_particles_in_multi_viewpoints(self.step, num_views=120, save_iid=True) # 360 / 120 for 30 fps
            # Clean up visualizer renderers to free memory
            self.visualizer.cleanup_renderers()

        # save model
        if self.opt.save_model:
            for j in range(self.opt.num_particles):
                self.save_model(mode='model', particle_id=j)
                self.save_model(mode='geo+tex', particle_id=j)   

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    gui.train(opt.iters)
