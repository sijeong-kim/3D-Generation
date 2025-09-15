# main_ours.py
from omegaconf import OmegaConf
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

        self.seed = opt.seed

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Set CUDA device if specified in options
        if hasattr(opt, 'gpu_id') and opt.gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(opt.gpu_id)
            self.device = torch.device(f"cuda:{opt.gpu_id}")
            print(f"[INFO] Using GPU {opt.gpu_id}: {torch.cuda.get_device_name(opt.gpu_id)}")
        
        self.bg_remover = None

        self.guidance_sd = None

        self.enable_sd = False

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

        # self.seeds = []
        # # override if provide a checkpoint
        # for i in range(self.opt.num_particles):
        #     # Set different seed for each particle during initialization
        #     init_seed = self.seed + i * 1000000 
        #     self.seeds.append(init_seed)
        #     self.seed_everything(init_seed)
            
        #     if self.opt.load is not None:
        #         self.renderers[i].initialize(self.opt.load) # TODO: load from different checkpoints for each particle
        #     else:
        #         # initialize gaussians to a blob
        #         self.renderers[i].initialize(num_pts=self.opt.num_pts)
        self.seeds = []
        for i in range(self.opt.num_particles):
            init_seed = self.seed + i * 1000000
            self.seeds.append(init_seed)
            self.seed_everything(init_seed)
            if self.opt.load is not None:
                self.renderers[i].initialize(self.opt.load)
            else:
                self.renderers[i].initialize(num_pts=self.opt.num_pts)

        # 🔹 RNGs: 전역 시드 리셋 없이 사용할 전용 제너레이터들
        self.rng_np = np.random.default_rng(int(self.seed))

        if torch.cuda.is_available():
            self.sd_rng = torch.Generator(device=self.device)
        else:
            self.sd_rng = torch.Generator()
        self.sd_rng.manual_seed(int(self.seed) + 12345)

        self.sd_rngs = []
        for s in self.seeds:
            g = torch.Generator(device=self.device) if torch.cuda.is_available() else torch.Generator()
            g.manual_seed(int(s) + 12345)
            self.sd_rngs.append(g)

        # visualizer
        if self.opt.visualize or self.opt.metrics:
            self.visualizer = GaussianVisualizer(opt=self.opt, renderers=self.renderers, cam=self.cam)
        else:
            self.visualizer = None
            
        # tsne
        self.features_outdir = os.path.join(self.opt.outdir, "features")
        os.makedirs(self.features_outdir, exist_ok=True)
        
        self._io_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        # __init__
        self.fast_mode = bool(getattr(opt, "fast_mode", True))
        if self.fast_mode:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    # seed_everything: 시드만!
    def seed_everything(self, seed):
        try: seed = int(seed)
        except: seed = np.random.randint(0, 1_000_000)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        
    def __del__(self):
        pass
    
    
    def _dump_run_meta(self, outdir):
        
        gpu_idx = torch.cuda.current_device() if torch.cuda.is_available() else None
        gpu_name = torch.cuda.get_device_name(gpu_idx) if gpu_idx is not None else None
        
        meta = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "gpu_name": gpu_name,
            "seed": int(self.seed),
            "opt": OmegaConf.to_container(self.opt, resolve=True),
        }
        import json, os
        with open(os.path.join(outdir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
                
    def _periodic_cuda_trim(self):
        gc.collect()
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved()
            allocated = torch.cuda.memory_allocated()
            if reserved - allocated > 1 * 1024**3:  # 1GB 이상 놀고 있으면
                torch.cuda.empty_cache()

            
    def teardown(self):
        # 렌더러/가우시안
        if hasattr(self, 'renderers'):
            for r in self.renderers:
                if hasattr(r, 'gaussians'):
                    del r.gaussians
                del r
            self.renderers = []

        # 시각화/피처/가이던스/메트릭/옵티마이저
        for name in ['visualizer','feature_extractor','guidance_sd','metrics_calculator']:
            if hasattr(self, name):
                delattr(self, name)
        if hasattr(self, 'optimizers'):
            for opt in self.optimizers:
                del opt
            self.optimizers = []
            
        if hasattr(self, '_io_stream'):
            del self._io_stream

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # def seed_everything(self, seed):
    #     try:
    #         seed = int(seed)
    #     except:
    #         seed = np.random.randint(0, 1000000)

    #     os.environ["PYTHONHASHSEED"] = str(seed)
    #     random.seed(seed)                       # ← 추가
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)        # ← 추가(멀티 GPU 대비)

    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.use_deterministic_algorithms(True)
    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #     torch.backends.cuda.matmul.allow_tf32 = False
    #     torch.backends.cudnn.allow_tf32 = False
    # def seed_everything(self, seed):
    #     try:
    #         seed = int(seed)
    #     except:
    #         seed = np.random.randint(0, 1_000_000)

    #     os.environ["PYTHONHASHSEED"] = str(seed)
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed)

    #     # 🔑 fast_mode에 따라 분기
    #     if getattr(self, "fast_mode", False):
    #         # 빠른 모드: 결정성 강제 X, TF32/benchmark 허용
    #         os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    #         torch.backends.cudnn.deterministic = False
    #         torch.backends.cudnn.benchmark = True
    #         torch.use_deterministic_algorithms(False)
    #         torch.backends.cuda.matmul.allow_tf32 = True
    #         torch.backends.cudnn.allow_tf32 = True
    #         if hasattr(torch, "set_float32_matmul_precision"):
    #             torch.set_float32_matmul_precision("high")
    #     else:
    #         # 재현성 모드
    #         os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    #         torch.use_deterministic_algorithms(True)
    #         torch.backends.cuda.matmul.allow_tf32 = False
    #         torch.backends.cudnn.allow_tf32 = False


    def prepare_train(self):

        self.step = 0
        self.optimizers = []
        for i in range(self.opt.num_particles):
            self.renderers[i].gaussians.training_setup(self.opt)
            self.renderers[i].gaussians.active_sh_degree = self.renderers[i].gaussians.max_sh_degree
            self.optimizers.append(self.renderers[i].gaussians.optimizer)

        # feature extractor
        if self.opt.repulsion_type != 'wo' or self.opt.save_features:
            # Use multi-layer extractor for specific layer extraction
            # Supports: 'early' (25% depth), 'mid' (50% depth), 'last' (final layer)
            
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
        self.step += 1

        wall_start = time.perf_counter()
        measure = bool(self.opt.metrics and self.metrics_calculator is not None)
        if measure and torch.cuda.is_available():
            cuda_evt_start = torch.cuda.Event(enable_timing=True)
            cuda_evt_start.record()
        else:
            cuda_evt_start = None

        # ---- Forward ----
        step_ratio = min(1, self.step / self.opt.schedule_iters)
        total_loss = 0

        render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
        images, outputs = [], []

        min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
        max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)

        # ✅ 전용 RNG 사용 (전역 시드 리셋 금지)
        ver = int(self.rng_np.integers(min_ver, max_ver))
        hor = int(self.rng_np.integers(-180, 180))
        radius = 0

        pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
        cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

        bg_white = self.rng_np.random() > self.opt.invert_bg_prob
        bg_color = torch.tensor([1, 1, 1] if bg_white else [0, 0, 0], dtype=torch.float32, device=self.device)

        for j in range(self.opt.num_particles):
            self.renderers[j].gaussians.update_learning_rate(self.step)
            out = self.renderers[j].render(cur_cam, bg_color=bg_color)

            image = out["image"].unsqueeze(0)
            images.append(image)
            outputs.append({
                "viewspace_points": out["viewspace_points"],
                "visibility_filter": out["visibility_filter"],
                "radii": out["radii"],
            })
            del out

        images = torch.cat(images, dim=0)  # [N,3,H,W]
        t_after_render = time.perf_counter()

        # ✅ 미리 None 선언 (wo일 때 클린업 안전)
        features = None
        kernel = None
        kernel_grad = None

        # Repulsion loss (feature space)
        repulsion_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.opt.repulsion_type in ['rlsd', 'svgd']:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(self.device.type == "cuda")):
                features = self.feature_extractor.extract_cls_from_layer(self.opt.feature_layer, images)
            features = features.to(torch.float32)

            if self.opt.kernel_type == 'rbf':
                kernel, kernel_grad = rbf_kernel_and_grad(features, repulsion_type=self.opt.repulsion_type, beta=self.opt.rbf_beta)
            elif self.opt.kernel_type == 'cosine':
                kernel, kernel_grad = cosine_kernel_and_grad(features, repulsion_type=self.opt.repulsion_type, beta=self.opt.cosine_beta, eps_shift=self.opt.cosine_eps_shift)
            else:
                raise ValueError(f"Invalid kernel type: {self.opt.kernel_type}")

            kernel = kernel.detach().to(torch.float32)
            kernel_grad = kernel_grad.detach().to(torch.float32)

            repulsion_loss = (kernel_grad * features).sum(dim=1).mean()

        # Attraction (latent space w/ SD)
        if self.opt.force_same_noise:
            gen = self.sd_rng
            score_gradients, latents = self.guidance_sd.train_step_gradient(
                images, step_ratio=step_ratio if self.opt.anneal_timestep else None,
                guidance_scale=self.opt.guidance_scale,
                force_same_t=self.opt.force_same_t,
                force_same_noise=True,
                generator=gen,
            )
        else:
            score_list, lat_list = [], []
            for j in range(self.opt.num_particles):
                gen_j = self.sd_rngs[j]
                sg_j, lat_j = self.guidance_sd.train_step_gradient(
                    images[j:j+1], step_ratio=step_ratio if self.opt.anneal_timestep else None,
                    guidance_scale=self.opt.guidance_scale,
                    force_same_t=self.opt.force_same_t,
                    force_same_noise=False,
                    generator=gen_j,
                )
                score_list.append(sg_j)
                lat_list.append(lat_j)
            score_gradients = torch.cat(score_list, dim=0)
            latents = torch.cat(lat_list, dim=0)

        # ✅ 수치 안정성: float32로 계산
        score_gradients = score_gradients.float()
        latents = latents.float()

        t_after_sd = time.perf_counter()

        if self.opt.repulsion_type == 'svgd':
            v = torch.einsum('ij,jchw->ichw', kernel.detach(), score_gradients)
            target = (latents - v).detach()
        elif self.opt.repulsion_type in ['rlsd', 'wo']:
            target = (latents - score_gradients).detach()
        else:
            raise ValueError(f"Invalid repulsion type: {self.opt.repulsion_type}")

        attraction_loss = 0.5 * F.mse_loss(latents, target, reduction='none').view(self.opt.num_particles, -1).sum(dim=1).mean()

        scaled_attraction_loss = self.opt.lambda_sd * attraction_loss
        scaled_repulsion_loss = self.opt.lambda_repulsion * repulsion_loss
        total_loss = scaled_attraction_loss + scaled_repulsion_loss

        if not torch.isfinite(total_loss):
            print(f"[WARN] non-finite total_loss at step {self.step}: "
                f"attract={scaled_attraction_loss.item():.4e}, "
                f"repulse={scaled_repulsion_loss.item():.4e}")
            for j in range(self.opt.num_particles):
                self.optimizers[j].zero_grad(set_to_none=True)
            return

        total_loss.backward()
        for j in range(self.opt.num_particles):
            self.optimizers[j].step()
        t_after_update = time.perf_counter()

        # densify & prune
        for j in range(self.opt.num_particles):
            if (self.step >= self.opt.density_start_iter) and (self.step <= self.opt.density_end_iter):
                viewspace_point_tensor = outputs[j]["viewspace_points"]
                visibility_filter = outputs[j]["visibility_filter"]
                radii = outputs[j]["radii"]
                self.renderers[j].gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.renderers[j].gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                self.renderers[j].gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.step % self.opt.densification_interval == 0:
                    self.renderers[j].gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)

                if getattr(self.opt, 'opacity_reset_interval', 0) and self.opt.opacity_reset_interval > 0:
                    if self.step % self.opt.opacity_reset_interval == 0:
                        self.renderers[j].gaussians.reset_opacity()

        t_after_densify = time.perf_counter()

        for j in range(self.opt.num_particles):
            self.optimizers[j].zero_grad()

        #########################################################
        # Log metrics and visualize
        #########################################################
        with torch.no_grad():
            if self.opt.metrics and (self.metrics_calculator is not None):
                # time
                if (self.step==1 or self.step % self.opt.efficiency_interval == 0):

                    # GPU step elapsed
                    gpu_step_ms = None
                    if torch.cuda.is_available():
                        cuda_evt_end = torch.cuda.Event(enable_timing=True)
                        cuda_evt_end.record()
                        torch.cuda.synchronize()
                        gpu_step_ms = cuda_evt_start.elapsed_time(cuda_evt_end)  # ms

                        memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                        max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    else:
                        memory_allocated = None
                        max_memory_allocated = None

                    # Wall-clock & section times (ms)
                    render_ms   = (t_after_render   - wall_start)      * 1000.0
                    sd_ms       = (t_after_sd       - t_after_render)  * 1000.0
                    backward_update_ms = (t_after_update - t_after_sd)      * 1000.0
                    densify_ms  = (t_after_densify  - t_after_update)* 1000.0
                    step_wall_ms= (t_after_densify  - wall_start)      * 1000.0

                    # Throughput (pixels/sec) — 참고용
                    pixels = int(self.opt.num_particles * render_resolution * render_resolution)
                    px_per_s = float(pixels) / max(step_wall_ms / 1000.0, 1e-9)

                    self.metrics_calculator.log_efficiency(
                        step=self.step,
                        efficiency={
                            "step_wall_ms": step_wall_ms,
                            "step_gpu_ms": gpu_step_ms,
                            "render_ms": render_ms,
                            "sd_guidance_ms": sd_ms,
                            "backward_update_ms": backward_update_ms,
                            "densify_ms": densify_ms,
                            "pixels": pixels,
                            "px_per_s": px_per_s,
                            "memory_allocated_mb": memory_allocated,
                            "max_memory_allocated_mb": max_memory_allocated,
                        },
                    )
                    
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
            
                # losses
                if (self.step==1 or self.step % self.opt.losses_interval == 0):
                    attraction_loss_val = attraction_loss.item()
                    repulsion_loss_val = repulsion_loss.item()
                    scaled_attraction_loss_val = self.opt.lambda_sd * attraction_loss_val
                    scaled_repulsion_loss_val = self.opt.lambda_repulsion * repulsion_loss_val
                    total_loss_val = total_loss.item()
                    ratio_pct = 100.0 * scaled_repulsion_loss_val / max(abs(scaled_attraction_loss_val), 1e-8)


                    # log
                    self.metrics_calculator.log_losses(
                        step=self.step,
                        losses= {
                            "attraction_loss": attraction_loss_val,
                            "repulsion_loss": repulsion_loss_val,
                            "scaled_attraction_loss": scaled_attraction_loss_val,
                            "scaled_repulsion_loss": scaled_repulsion_loss_val,
                            "total_loss": total_loss_val,
                            "scaled_repulsion_loss_ratio": ratio_pct,
                            "lambda_repulsion": float(self.opt.lambda_repulsion),
                        },
                    )
            
                # quantitative metrics
                if (self.step==1 or self.step % self.opt.quantitative_metrics_interval == 0) and (self.visualizer is not None):
                    # Update visualizer with current training state
                    self.visualizer.update_renderers(self.renderers)
                    multi_view_images = self.visualizer.visualize_all_particles_in_multi_viewpoints(self.step, visualize_multi_viewpoints=self.opt.visualize_multi_viewpoints, save_iid=self.opt.save_iid)  # [V, N, 3, H, W]
                    
                    if self.opt.save_features and (self.step==1 or self.step % self.opt.save_features_interval == 0):
                        self.save_features(multi_view_images, self.step)

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
                            # fidelity
                            "fidelity_mean": fidelity_mean,
                            "fidelity_std": fidelity_std,
                            # inter-particle diversity
                            "inter_particle_diversity_mean": inter_particle_diversity_mean,
                            "inter_particle_diversity_std": inter_particle_diversity_std,
                            # cross-view consistency
                            "cross_view_consistency_mean": cross_view_consistency_mean,
                            "cross_view_consistency_std": cross_view_consistency_std,
                            # LPIPS (inter-sample and cross-view consistency)
                            "lpips_inter_mean": lpips_inter_mean if self.opt.enable_lpips else None,
                            "lpips_inter_std": lpips_inter_std if self.opt.enable_lpips else None,
                            "lpips_consistency_mean": lpips_consistency_mean if self.opt.enable_lpips else None,
                            "lpips_consistency_std": lpips_consistency_std if self.opt.enable_lpips else None,
                        }
                    )
                    
                

            # visualize
            if self.opt.visualize and (self.visualizer is not None):
                # save rendered images (save at the end of each interval)
                if self.opt.save_rendered_images and (self.step==1 or self.step % self.opt.save_rendered_images_interval == 0):
                    self.visualizer.update_renderers(self.renderers)
                    self.visualizer.save_rendered_images(self.step, images)
                    self.visualizer.cleanup_renderers()
                    
                if self.opt.visualize_fixed_viewpoint and (self.step==1 or self.step % self.opt.visualize_fixed_viewpoint_interval == 0):
                    self.visualizer.update_renderers(self.renderers)
                    self.visualizer.visualize_fixed_viewpoint(self.step)
                    self.visualizer.cleanup_renderers()
                
            # # save model at the best step
            # if self.step == self.opt.best_step:
            #     print(f"[INFO] best step reached: {self.step}")
            #     # Multi-viewpoints for 30 fps video (save at the end of training)
            #     if self.opt.video_snapshot:
            #         # Update visualizer with final training state
            #         self.visualizer.update_renderers(self.renderers)
            #         self.visualizer.visualize_all_particles_in_multi_viewpoints(self.step, num_views=120, save_iid=True) # 360 / 120 for 30 fps
            #         # Clean up visualizer renderers to free memory
            #         self.visualizer.cleanup_renderers()

            #     # save model
            #     if self.opt.save_model:
            #         for j in range(self.opt.num_particles):
            #             self.save_model(mode='model', particle_id=j, step=self.step)
            #             self.save_model(mode='geo+tex', particle_id=j, step=self.step)
                
        # Periodic GPU memory cleanup
        if self.step % self.opt.efficiency_interval == 0:
            images = outputs = score_gradients = latents = features = kernel = kernel_grad = None
            self._periodic_cuda_trim()


    
    @torch.no_grad()
    def _extract_particle_features_from_multi_view(
        self,
        images_VN_3HW: torch.Tensor,   # [V, N, 3, H, W]
        layer_idx: int,
        use_amp: bool = True,
        chunk_size: int = 64,
        l2_after_each_view: bool = True,
    ):
        """
        반환:
        particle_feats [N, D]  : 뷰 평균된 파티클 대표 특징(최종 L2 정규화)
        view_feats     [V, N, D] : 각 뷰별 특징(각각 L2 정규화되어 반환)
        """
        V, N, C, H, W = images_VN_3HW.shape
        X = images_VN_3HW.reshape(V * N, C, H, W).contiguous().to(self.device, non_blocking=True)

        feats_list = []
        i = 0
        while i < V * N: # 8 * 8 = 64
            j = min(i + chunk_size, V * N)
            x = X[i:j]
            with torch.no_grad():
                if use_amp:
                    from contextlib import nullcontext

                    amp_ctx = (
                        torch.amp.autocast("cuda", dtype=torch.float16, enabled=(self.device.type == "cuda"))
                        if torch.cuda.is_available()
                        else nullcontext()
                    )
                    with amp_ctx:
                        f = self.feature_extractor.extract_cls_from_layer(layer_idx, x)  # [B, D]
                else:
                    f = self.feature_extractor.extract_cls_from_layer(layer_idx, x)
            if l2_after_each_view:
                f = F.normalize(f, dim=-1)
            feats_list.append(f)
            i = j

        feats = torch.cat(feats_list, dim=0)        # [V*N, D]
        feats = feats.view(V, N, -1)                # [V, N, D]
        # 뷰 평균 → 파티클 대표 특징
        particle_feats = feats.mean(dim=0)          # [N, D]
        particle_feats = F.normalize(particle_feats, dim=-1)
        view_feats = F.normalize(feats, dim=-1) if not l2_after_each_view else feats  # [V, N, D]
        return particle_feats, view_feats

                
    @torch.no_grad()
    def save_features(self, images, step=None):  # images: [V, N, 3, H, W]
        step = int(self.step if step is None else step)
        # 멀티뷰 → 파티클 대표 특징 뽑기
        particle_feats, view_feats = self._extract_particle_features_from_multi_view(
            images_VN_3HW=images,
            layer_idx=self.opt.feature_layer,
            use_amp=True,
            chunk_size=int(getattr(self.opt, "feat_chunk", 64)),
            l2_after_each_view=True,
        )  # [N, D], [V, N, D]

        # FP16으로 축소 후 CPU로 비블로킹 복사
        pf16 = particle_feats.to(torch.float16).contiguous()
        vf16 = view_feats.to(torch.float16).contiguous()

        if torch.cuda.is_available():
            if self._io_stream is None:
                self._io_stream = torch.cuda.Stream()
            with torch.cuda.stream(self._io_stream):
                pf16_cpu = pf16.to("cpu", non_blocking=True)
                vf16_cpu = vf16.to("cpu", non_blocking=True)
            torch.cuda.current_stream().wait_stream(self._io_stream)
        else:
            pf16_cpu = pf16
            vf16_cpu = vf16

        out_path = os.path.join(self.features_outdir, f"step_{step:06d}.pt")
        torch.save(
            {
                "step": step,
                "V": int(images.shape[0]),
                "N": int(images.shape[1]),
                "particle_ids": torch.arange(self.opt.num_particles, dtype=torch.int32),
                "particle_feats_fp16": pf16_cpu,  # [N, D]
                "view_feats_fp16": vf16_cpu,      # [V, N, D]
                "model_name": self.opt.feature_extractor_model_name,
                "layer_idx": int(self.opt.feature_layer),
            },
            out_path
        )
        print(f"[INFO] saved features: {out_path}")

    @torch.no_grad()
    def save_model(self, mode='geo', texture_size=1024, particle_id=0, step=None):
        path = os.path.join(self.opt.outdir, f'saved_models')
        os.makedirs(path, exist_ok=True)
        
        if mode == 'geo':
            path = os.path.join(path, f'step_{step}_particle_{particle_id}_mesh.ply')
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
            path = os.path.join(path, f'step_{step}_particle_{particle_id}_mesh.' + self.opt.mesh_format)
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

            # if not self.opt.force_cuda_rast and (not self.opt.gui or os.name == 'nt'):
            #     glctx = dr.RasterizeGLContext()
            # else:
            #     glctx = dr.RasterizeCudaContext()
                
            use_gl = not getattr(self.opt, "force_cuda_rast", False) and (not getattr(self.opt, "gui", False) or os.name == 'nt')
            glctx = dr.RasterizeGLContext() if use_gl else dr.RasterizeCudaContext()


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
            path = os.path.join(path, f'step_{step}_particle_{particle_id}_model.ply')
            self.renderers[particle_id].gaussians.save_ply(path)

            print(f"[INFO] particle {particle_id} save model to {path}.")
            # Cleanup CUDA cache after saving
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # no gui mode
    def train(self, iters=500):
        
        os.makedirs(self.opt.outdir, exist_ok=True)
        self._dump_run_meta(self.opt.outdir)
        
        try:
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
                self.visualizer.visualize_all_particles_in_multi_viewpoints(self.step, num_views=120, visualize_multi_viewpoints=True, save_iid=True) # 360 / 120 for 30 fps
                # Clean up visualizer renderers to free memory
                self.visualizer.cleanup_renderers()

            # save model
            if self.opt.save_model:
                for j in range(self.opt.num_particles):
                    self.save_model(mode='model', particle_id=j, step=self.step)
                    self.save_model(mode='geo+tex', particle_id=j, step=self.step)   

        finally:
            self.teardown()

if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    gui.train(opt.iters)
