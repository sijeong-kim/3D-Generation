# metrics.py
import torch
import torch.nn.functional as F
import open_clip
import os
from typing import Tuple

from feature_extractor import DINOv2MultiLayerFeatureExtractor

class MetricsCalculator:
    def __init__(
        self, 
        opt: dict = None, 
        prompt: str = "a photo of a hamburger",
        multi_view_type: str = None, 
        feature_extractor: DINOv2MultiLayerFeatureExtractor = None,
    ):
        # Validate opt parameter
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt
        
        self.save_dir = os.path.join(self.opt.outdir, 'metrics')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.prompt = prompt
        
        self.multi_view_type = multi_view_type

        self.feature_extractor = feature_extractor

        # OpenCLIP settings - using laion/CLIP-ViT-bigG-14-laion2B-39B-b160k1
        self.features_normalized = True
        self.model_name = "ViT-bigG-14"
        self.pretrained = "laion2b_s39b_b160k"
        self.device = "cuda"

        # Load OpenCLIP
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, device=self.device)
        tokenizer = open_clip.get_tokenizer(self.model_name)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # Initialize CSV files
        if opt is not None and hasattr(opt, 'metrics') and opt.metrics:
            self.metrics_csv_path = os.path.join(self.save_dir, "quantitative_metrics.csv")
            self.metrics_csv_file = open(self.metrics_csv_path, "w")
            self.metrics_csv_file.write(f"step,inter_particle_diversity_mean,inter_particle_diversity_std,intra_particle_diversity_mean,intra_particle_diversity_std,fidelity_mean,fidelity_std\n")
        
            self.losses_csv_path = os.path.join(self.save_dir, "losses.csv")
            self.losses_csv_file = open(self.losses_csv_path, "w")
            self.losses_csv_file.write("step,attraction_loss,repulsion_loss,scaled_attraction_loss,scaled_repulsion_loss,total_loss\n")
            
            self.efficiency_csv_path = os.path.join(self.save_dir, "efficiency.csv")
            self.efficiency_csv_file = open(self.efficiency_csv_path, "w")
            self.efficiency_csv_file.write("step,time,memory_allocated_mb,max_memory_allocated_mb\n")
        else:
            self.metrics_csv_file = None
            self.losses_csv_file = None
            self.efficiency_csv_file = None
        
    @torch.no_grad()
    def log_quantitative_metrics(self, step: int, inter_particle_diversity_mean: float, inter_particle_diversity_std: float, intra_particle_diversity_mean: float, intra_particle_diversity_std: float, fidelity_mean: float, fidelity_std: float):
        if self.metrics_csv_file is not None:
            # Always output columns per header; use empty if not provided
            def fmt(x):
                return "" if x is None else f"{x}"
            self.metrics_csv_file.write(
                f"{step},{fmt(inter_particle_diversity_mean)},{fmt(inter_particle_diversity_std)},{fmt(intra_particle_diversity_mean)},{fmt(intra_particle_diversity_std)},{fmt(fidelity_mean)},{fmt(fidelity_std)}\n"
            )
            self.metrics_csv_file.flush()
            
    @torch.no_grad()
    def log_losses(self, step: int, attraction_loss: float, repulsion_loss: float, scaled_attraction_loss: float, scaled_repulsion_loss: float, total_loss: float):
        if self.losses_csv_file is not None:
            self.losses_csv_file.write(f"{step},{attraction_loss},{repulsion_loss},{scaled_attraction_loss},{scaled_repulsion_loss},{total_loss}\n")
            self.losses_csv_file.flush()
            
    @torch.no_grad()
    def log_efficiency(self, step: int, time: float, memory_allocated_mb: float = None, max_memory_allocated_mb: float = None):
        if self.efficiency_csv_file is not None:
            self.efficiency_csv_file.write(f"{step},{time},{memory_allocated_mb},{max_memory_allocated_mb}\n") # time in ms, memory in MB
            self.efficiency_csv_file.flush()
            
    @torch.no_grad()
    def log_metrics(self, step: int, inter_particle_diversity_mean: float, inter_particle_diversity_std: float, intra_particle_diversity_mean: float, intra_particle_diversity_std: float, fidelity_mean: float, fidelity_std: float, attraction_loss: float, repulsion_loss: float, scaled_attraction_loss: float, scaled_repulsion_loss: float, total_loss: float, time: float, memory_allocated_mb: float = None, max_memory_allocated_mb: float = None):
        if self.opt is not None and self.save_dir is not None:
            if hasattr(self.opt, 'quantitative_metrics_interval') and step % self.opt.quantitative_metrics_interval == 0:
                self.log_quantitative_metrics(step=step, inter_particle_diversity_mean=inter_particle_diversity_mean, inter_particle_diversity_std=inter_particle_diversity_std, intra_particle_diversity_mean=intra_particle_diversity_mean, intra_particle_diversity_std=intra_particle_diversity_std, fidelity_mean=fidelity_mean, fidelity_std=fidelity_std)
            if hasattr(self.opt, 'losses_interval') and step % self.opt.losses_interval == 0:
                self.log_losses(step=step, attraction_loss=attraction_loss, repulsion_loss=repulsion_loss, scaled_attraction_loss=scaled_attraction_loss, scaled_repulsion_loss=scaled_repulsion_loss, total_loss=total_loss)
            if hasattr(self.opt, 'efficiency_interval') and step % self.opt.efficiency_interval == 0:
                self.log_efficiency(step=step, time=time, memory_allocated_mb=memory_allocated_mb, max_memory_allocated_mb=max_memory_allocated_mb)

    @torch.no_grad()
    def _preprocess_images_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images using torchvision transforms (all on GPU, no PIL).
        Input images: [N, 3, H, W] in [0, 1] float32
        Output: [N, 3, 224, 224] normalized for CLIP
        """
        # Resize to 224x224 (CLIP input size)
        images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize using CLIP stats (depends on model)
        # CLIP ViT-B/32 (OpenCLIP) uses these stats
        mean = torch.tensor([0.4815, 0.4578, 0.4082], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2686, 0.2613, 0.2758], device=images.device).view(1, 3, 1, 1)

        images_normalized = (images_resized - mean) / std
        return images_normalized
    
    
    # @torch.no_grad()
    # def compute_rlsd_diversity(self, features: torch.Tensor) -> float: 
    #     """
    #     Computes RLSD-based diversity as 1 - average cosine similarity.

    #     Args:
    #         features: [N, D] tensor of feature vectors (e.g., DINO embeddings)

    #     Returns:
    #         float: diversity score (higher = more diverse)
    #     """
        
    #     N = features.shape[0]
    #     if N < 2:
    #         return 0.0

    #     # Normalize features for cosine similarity
    #     if not self.features_normalized:
    #         features_norm = F.normalize(features, dim=1)
    #     else:
    #         features_norm = features

    #     # Compute cosine similarity matrix
    #     sim_matrix = torch.matmul(features_norm, features_norm.T) # [V, N, N]

    #     # Remove diagonal (self-similarity)
    #     mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
    #     sims = sim_matrix[mask].view(N, -1)

    #     # Average similarity and convert to diversity
    #     sim_mean = sims.mean().item()
    #     diversity = 1.0 - sim_mean
    #     return max(0.0, min(1.0, diversity))  # Clamp to [0,1]
    
    
    # def compute_rlsd_diversity_in_multi_viewpoints_mean(self, multi_view_images: torch.Tensor, step: int) -> float:
    #     """
    #     Computes inter-particle RLSD diversity averaged over views.

    #     For each view v, builds an N×N cosine-similarity matrix between particles
    #     and averages the off-diagonal entries; then averages across views. Returns
    #     1 − mean_similarity.

    #     Args:
    #         multi_view_images: [V, N, 3, H, W]

    #     Returns:
    #         float: diversity score (higher = more diverse)
    #     """
    #     # [V, N, 3, H, W] -> [V*N, 3, H, W]
    #     multi_view_images = multi_view_images.view(-1, 3, multi_view_images.shape[-2], multi_view_images.shape[-1])
    #     multi_view_features = self.feature_extractor.extract_cls_from_layer(self.opt.feature_layer, multi_view_images).to(self.device)  # [V*N, D]
    #     # [V*N, D_feature] -> [V, N, D_feature]
    #     multi_view_features = multi_view_features.view(self.opt.num_views, self.opt.num_particles, -1)
        
    #     # save features

    #     if self.opt.save_rep_features and step % self.opt.save_rep_features_interval == 0:
    #         save_dir = os.path.join(self.opt.outdir, f"features")
    #         os.makedirs(save_dir, exist_ok=True)
    #         torch.save(multi_view_features, os.path.join(save_dir, f"step_{step}_layer_{self.opt.feature_layer}.pt"))
            
    #     # Compute RLSD over views: for each view v, compute [N,N] cosine sim, then average over views
    #     V = multi_view_features.shape[0]
    #     N = multi_view_features.shape[1]
    #     if N < 2:
    #         return 0.0

    #     # Normalize features along D if needed
    #     if not self.features_normalized:
    #         multi_view_features = F.normalize(multi_view_features, dim=2)

    #     # Batched similarity: [V, N, D] x [V, D, N] -> [V, N, N]
    #     sim_matrix = torch.bmm(multi_view_features, multi_view_features.transpose(1, 2))

    #     # Remove diagonal per-view
    #     diag_mask = ~torch.eye(N, dtype=torch.bool, device=multi_view_features.device).unsqueeze(0).expand(V, -1, -1)
    #     sims = sim_matrix[diag_mask]

    #     # Average similarity across all off-diagonal pairs and views
    #     sim_mean = sims.mean().item()
    #     diversity = 1.0 - sim_mean
    #     return max(0.0, min(1.0, diversity))  # Clamp to [0,1]

    @torch.no_grad()
    def compute_inter_particle_diversity_in_multi_viewpoints_stats(self, multi_view_images: torch.Tensor, step: int):
        """
        Inter-particle RLSD diversity statistics across views.

        Returns mean and std of per-view diversity, where per-view diversity is
        defined as 1 - mean off-diagonal cosine similarity between particles.

        Args:
            multi_view_images: [V, N, 3, H, W]

        Returns:
            Tuple[float, float]: (mean_diversity, std_diversity)
        """
        multi_view_images = multi_view_images.view(-1, 3, multi_view_images.shape[-2], multi_view_images.shape[-1])
        multi_view_features = self.feature_extractor.extract_cls_from_layer(self.opt.feature_layer, multi_view_images).to(self.device)  # [V*N, D]
        multi_view_features = multi_view_features.view(self.opt.num_views, self.opt.num_particles, -1)  # [V, N, D]

        V = multi_view_features.shape[0]
        N = multi_view_features.shape[1]
        if N < 2:
            return 0.0, 0.0

        if not self.features_normalized:
            multi_view_features = F.normalize(multi_view_features, dim=2)

        sim_matrix = torch.bmm(multi_view_features, multi_view_features.transpose(1, 2))  # [V, N, N]
        diag_mask = ~torch.eye(N, dtype=torch.bool, device=multi_view_features.device).unsqueeze(0).expand(V, -1, -1)
        sims = sim_matrix[diag_mask].view(V, -1)  # [V, N*(N-1)]
        per_view_sim_mean = sims.mean(dim=1)  # [V]
        per_view_div = 1.0 - per_view_sim_mean  # [V]
        mean_div = per_view_div.mean().item()
        std_div = per_view_div.std(unbiased=False).item() if V > 1 else 0.0
        return max(0.0, min(1.0, mean_div)), max(0.0, min(1.0, std_div))

    @torch.no_grad()
    def compute_intra_particle_diversity_in_multi_viewpoints_stats(self, multi_view_images: torch.Tensor, step: int):
        """
        Computes intra-particle diversity across viewpoints.

        For each particle i, computes the V×V cosine-similarity matrix across its
        views and averages the off-diagonal entries; converts to diversity as
        1 − mean_similarity per particle, and finally averages across particles.

        Args:
            multi_view_images: [V, N, 3, H, W]

        Returns:
            Tuple[float, torch.Tensor]:
                - mean_diversity: scalar float averaged over particles
                - per_particle_diversity: [N] tensor
        """
        # [V, N, 3, H, W] -> [V*N, 3, H, W]
        multi_view_images = multi_view_images.view(-1, 3, multi_view_images.shape[-2], multi_view_images.shape[-1])
        features = self.feature_extractor.extract_cls_from_layer(self.opt.feature_layer, multi_view_images).to(self.device)  # [V*N, D]
        # [V*N, D] -> [V, N, D]
        features = features.view(self.opt.num_views, self.opt.num_particles, -1)

        V = features.shape[0]
        N = features.shape[1]
        if V < 2 or N == 0:
            return 0.0, torch.zeros(N, device=self.device)

        # Normalize along feature dim if needed
        if not self.features_normalized:
            features = F.normalize(features, dim=2)

        # Rearrange per particle: [N, V, D]
        features_nv = features.permute(1, 0, 2).contiguous()

        # Per particle view-similarity: [N, V, D] x [N, D, V] -> [N, V, V]
        sims_nv = torch.bmm(features_nv, features_nv.transpose(1, 2))

        # Remove diagonal per particle
        diag_mask_v = ~torch.eye(V, dtype=torch.bool, device=features.device).unsqueeze(0).expand(N, -1, -1)
        sims_off_diag = sims_nv[diag_mask_v].view(N, -1)  # [N, V*(V-1)]

        per_particle_sim_mean = sims_off_diag.mean(dim=1)  # [N]
        per_particle_diversity = 1.0 - per_particle_sim_mean  # [N]
        mean_diversity = per_particle_diversity.mean().item()

        return max(0.0, min(1.0, mean_diversity)), per_particle_diversity



    # @torch.no_grad()
    # def compute_clip_fidelity(self, images: torch.Tensor) -> float:
    #     """
    #     Compute CLIP-based fidelity score between generated images and prompt.
    #     Args:
    #         images: [N, 3, H, W] in [0, 1]
    #     Returns:
    #         float: Mean cosine similarity between CLIP image and text embeddings
    #     """
    #     # Ensure images are on the correct device
    #     images = images.to(self.device)
        
    #     # OpenCLIP method
    #     images_clip = self._preprocess_images_tensor(images)  # [N, 3, 224, 224]

    #     # Tokenize and expand prompt
    #     tokens = self.tokenizer(self.prompt).to(self.device)
    #     tokens = tokens.expand(images.shape[0], -1)

    #     # Encode
    #     image_features = self.model.encode_image(images_clip)
    #     text_features = self.model.encode_text(tokens)

    #     # Normalize & similarity
    #     image_features = F.normalize(image_features, dim=1)
    #     text_features = F.normalize(text_features, dim=1)
    #     similarities = (image_features * text_features).sum(dim=1)

    #     # Return mean similarity
    #     return similarities.mean().item()
    
    # @torch.no_grad()
    # def compute_clip_fidelity_in_multi_viewpoints_backup(self, images: torch.Tensor, multi_view_type: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Compute CLIP fidelity across multiple viewpoints using different aggregation strategies.

    #     Args:
    #         images: [N, V, 3, H, W] in [0, 1] - N particles, V views per particle
    #         multi_view_type: Strategy for aggregating views ('best_views', 'average_views', 'cross_attention_views')

    #     Returns:
    #         best_images: [N, 3, H, W] - Selected images for each particle
    #         best_similarities: [N] - Similarity scores for selected images
    #     """
    #     N, V, C, H, W = images.shape
    #     device = images.device

    #     # Flatten views for batch CLIP processing
    #     images_flat = images.view(N * V, C, H, W)

    #     # Preprocess images for CLIP
    #     images_clip = self._preprocess_images_tensor(images_flat)  # [N*V, 3, 224, 224]

    #     # Encode image features
    #     image_features = self.model.encode_image(images_clip)
    #     image_features = F.normalize(image_features, dim=1)

    #     # Encode repeated prompt
    #     tokens = self.tokenizer(self.prompt).to(device)
    #     tokens = tokens.expand(N * V, -1)
    #     text_features = self.model.encode_text(tokens)
    #     text_features = F.normalize(text_features, dim=1)

    #     # Compute similarity
    #     sims = (image_features * text_features).sum(dim=1)  # [N*V]
   
    #     sims = sims.view(N, V)  # [N, V]


    #     # Use instance default if not specified
    #     if multi_view_type is None:
    #         multi_view_type = self.multi_view_type
            
    #     if multi_view_type == "best_views":
    #         # Find best similarity and corresponding view index
    #         best_sim_vals, best_view_indices = sims.max(dim=1)  # [N]
    #     elif multi_view_type == "average_views":
    #         # Compute average similarity across all views
    #         best_sim_vals = sims.mean(dim=1)  # [N]
    #         # For average case, we still need to select one representative image
    #         # Choose the view closest to the average similarity for each particle
    #         _, best_view_indices = torch.min(torch.abs(sims - best_sim_vals.unsqueeze(1)), dim=1)
    #     else:
    #         raise ValueError(f"Invalid multi-view type: {multi_view_type}. Valid options: 'best_views', 'average_views', 'cross_attention_views'")

    #     # Select best images (V>=1)
    #     best_images = images[torch.arange(N, device=device), best_view_indices] # [N, 3, H, W]

    #     return best_images, best_sim_vals


    
    # @torch.no_grad()
    # def compute_clip_fidelity_in_multi_viewpoints(self, images: torch.Tensor, multi_view_type: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Compute CLIP fidelity across multiple viewpoints using different aggregation strategies.

    #     Args:
    #         images: [N, V, 3, H, W] in [0, 1] - N particles, V views per particle
    #         multi_view_type: Strategy for aggregating views ('best_views', 'average_views', 'cross_attention_views')

    #     Returns:
    #         similarities: [N] - Similarity scores for selected images
    #     """
    #     N, V, C, H, W = images.shape
    #     device = images.device

    #     # Flatten views for batch CLIP processing
    #     images_flat = images.view(N * V, C, H, W)

    #     # Preprocess images for CLIP
    #     images_clip = self._preprocess_images_tensor(images_flat)  # [N*V, 3, 224, 224]

    #     # Encode image features
    #     image_features = self.model.encode_image(images_clip)
    #     image_features = F.normalize(image_features, dim=1)

    #     # Encode repeated prompt
    #     tokens = self.tokenizer(self.prompt).to(device)
    #     tokens = tokens.expand(N * V, -1)
    #     text_features = self.model.encode_text(tokens)
    #     text_features = F.normalize(text_features, dim=1)

    #     # Compute similarity
    #     sims = (image_features * text_features).sum(dim=1)  # [N*V]
   
    #     sims = sims.view(N, V)  # [N, V]


    #     # Use instance default if not specified
    #     if multi_view_type is None:
    #         multi_view_type = self.multi_view_type
            
    #     if multi_view_type == "best_views":
    #         # Find best similarity and corresponding view index
    #         sim_vals, view_indices = sims.max(dim=1)  # [N]
    #     elif multi_view_type == "average_views":
    #         # Compute average similarity across all views
    #         sim_vals = sims.mean(dim=1)  # [N]
    #     else:
    #         raise ValueError(f"Invalid multi-view type: {multi_view_type}. Valid options: 'best_views', 'average_views'")

    #     fidelity = sim_vals.mean().item()
    #     fidelity_std = sim_vals.std(unbiased=False).item() if sim_vals.numel() > 1 else 0.0


    #     return fidelity, fidelity_std # [N]

if __name__ == "__main__":
    import argparse
    
    # Try to import optional dependencies for full test
    try:
        from guidance.sd_utils import seed_everything      
        has_full_deps = True
    except ImportError as e:
        print(f"Warning: Some dependencies not available ({e}). Running simplified test.")
        has_full_deps = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="a photo of a hamburger")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic images instead of generated ones")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for saving results")
    opt = parser.parse_args()

    if has_full_deps:
        seed_everything(opt.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N = 4  # Number of images
    V = 3  # Number of views per image for multi-view test
    resolution = 512
