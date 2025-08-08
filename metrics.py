# metrics.py
import torch
import torch.nn.functional as F
import open_clip
import os
from typing import Tuple

class MetricsCalculator:
    def __init__(self, opt: dict = None, prompt: str = "a photo of a hamburger", multi_view_type: str = None):
        # Validate opt parameter
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt
        
        self.save_dir = os.path.join(self.opt.outdir, 'metrics')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.prompt = prompt
        
        if self.opt.multi_view_type is not None:
            self.multi_view_type = multi_view_type
        else:
            self.multi_view_type = self.opt.multi_view_type


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
            self.metrics_csv_file.write(f"step,diversity,fidelity\n")
        
            self.losses_csv_path = os.path.join(self.save_dir, "losses.csv")
            self.losses_csv_file = open(self.losses_csv_path, "w")
            self.losses_csv_file.write("step,attraction_loss,repulsion_loss,scaled_repulsion_loss,total_loss\n")
            
            self.efficiency_csv_path = os.path.join(self.save_dir, "efficiency.csv")
            self.efficiency_csv_file = open(self.efficiency_csv_path, "w")
            self.efficiency_csv_file.write("step,time,memory_allocated_mb,max_memory_allocated_mb\n")
        else:
            self.metrics_csv_file = None
            self.losses_csv_file = None
            self.efficiency_csv_file = None
        
    @torch.no_grad()
    def log_quantitative_metrics(self, step: int, diversity: float, fidelity: float):
        if self.metrics_csv_file is not None:
            self.metrics_csv_file.write(f"{step},{diversity},{fidelity}\n")
            self.metrics_csv_file.flush()
            
    @torch.no_grad()
    def log_losses(self, step: int, attraction_loss: float, repulsion_loss: float, scaled_repulsion_loss: float, total_loss: float):
        if self.losses_csv_file is not None:
            self.losses_csv_file.write(f"{step},{attraction_loss},{repulsion_loss},{scaled_repulsion_loss},{total_loss}\n")
            self.losses_csv_file.flush()
            
    @torch.no_grad()
    def log_efficiency(self, step: int, time: float, memory_allocated_mb: float = None, max_memory_allocated_mb: float = None):
        if self.efficiency_csv_file is not None:
            self.efficiency_csv_file.write(f"{step},{time},{memory_allocated_mb},{max_memory_allocated_mb}\n") # time in ms, memory in MB
            self.efficiency_csv_file.flush()
            
    @torch.no_grad()
    def log_metrics(self, step: int, diversity: float, fidelity: float, attraction_loss: float, repulsion_loss: float, scaled_repulsion_loss: float, total_loss: float, time: float, memory_allocated_mb: float = None, max_memory_allocated_mb: float = None):
        if self.opt is not None and self.save_dir is not None:
            if hasattr(self.opt, 'quantitative_metrics_interval') and step % self.opt.quantitative_metrics_interval == 0:
                self.log_quantitative_metrics(step=step, diversity=diversity, fidelity=fidelity)
            if hasattr(self.opt, 'losses_interval') and step % self.opt.losses_interval == 0:
                self.log_losses(step=step, attraction_loss=attraction_loss, repulsion_loss=repulsion_loss, scaled_repulsion_loss=scaled_repulsion_loss, total_loss=total_loss)
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
    
    
    @torch.no_grad()
    def compute_rlsd_diversity(self, features: torch.Tensor) -> float:
        """
        Computes RLSD-based diversity as 1 - average cosine similarity.

        Args:
            features: [N, D] tensor of feature vectors (e.g., DINO embeddings)

        Returns:
            float: diversity score (higher = more diverse)
        """
        N = features.shape[0]
        if N < 2:
            return 0.0

        # Normalize features for cosine similarity
        if not self.features_normalized:
            features_norm = F.normalize(features, dim=1)
        else:
            features_norm = features

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(features_norm, features_norm.T)

        # Remove diagonal (self-similarity)
        mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        sims = sim_matrix[mask].view(N, -1)

        # Average similarity and convert to diversity
        sim_mean = sims.mean().item()
        diversity = 1.0 - sim_mean
        return max(0.0, min(1.0, diversity))  # Clamp to [0,1]
    
    @torch.no_grad()
    def compute_clip_fidelity(self, images: torch.Tensor) -> float:
        """
        Compute CLIP-based fidelity score between generated images and prompt.
        Args:
            images: [N, 3, H, W] in [0, 1]
        Returns:
            float: Mean cosine similarity between CLIP image and text embeddings
        """
        # Ensure images are on the correct device
        images = images.to(self.device)
        
        # OpenCLIP method
        images_clip = self._preprocess_images_tensor(images)  # [N, 3, 224, 224]

        # Tokenize and expand prompt
        tokens = self.tokenizer(self.prompt).to(self.device)
        tokens = tokens.expand(images.shape[0], -1)

        # Encode
        image_features = self.model.encode_image(images_clip)
        text_features = self.model.encode_text(tokens)

        # Normalize & similarity
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        similarities = (image_features * text_features).sum(dim=1)

        # Return mean similarity
        return similarities.mean().item()
    
    @torch.no_grad()
    def compute_clip_fidelity_in_multi_viewpoints(self, images: torch.Tensor, multi_view_type: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute CLIP fidelity across multiple viewpoints using different aggregation strategies.

        Args:
            images: [N, V, 3, H, W] in [0, 1] - N particles, V views per particle
            multi_view_type: Strategy for aggregating views ('best_views', 'average_views', 'cross_attention_views')

        Returns:
            best_images: [N, 3, H, W] - Selected images for each particle
            best_similarities: [N] - Similarity scores for selected images
        """
        N, V, C, H, W = images.shape
        device = images.device

        # Flatten views for batch CLIP processing
        images_flat = images.view(N * V, C, H, W)

        # Encode image features
        image_features = self.model.encode_image(images_flat)
        image_features = F.normalize(image_features, dim=1)

        # Encode repeated prompt
        tokens = self.tokenizer(self.prompt).to(device)
        tokens = tokens.expand(N * V, -1)
        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=1)

        # Compute similarity
        sims = (image_features * text_features).sum(dim=1)  # [N*V]
   
        sims = sims.view(N, V)  # [N, V]


        # Use instance default if not specified
        if multi_view_type is None:
            multi_view_type = self.multi_view_type
            
        if multi_view_type == "best_views":
            # Find best similarity and corresponding view index
            best_sim_vals, best_view_indices = sims.max(dim=1)  # [N]
            
            return best_sim_vals, best_view_indices
        
        elif multi_view_type == "average_views":
            # Compute average similarity across all views
            mean_sim_vals = sims.mean(dim=1)  # [N]
            # For average case, we still need to select one representative image
            # Choose the view closest to the average similarity for each particle
            _, mean_view_indices = torch.min(torch.abs(sims - mean_sim_vals.unsqueeze(1)), dim=1)
            
            return mean_sim_vals, mean_view_indices
            
        elif multi_view_type == "cross_attention_views":
            # Compute cross-attention similarity between image and text
            cross_attn = self.model.get_cross_attention(images_flat, tokens)
            best_sim_vals = cross_attn.mean(dim=1)  # [N]
            # For cross-attention case, we still need to select one representative image
            # Choose the view closest to the average similarity for each particle
            _, best_view_indices = torch.min(torch.abs(sims - best_sim_vals.unsqueeze(1)), dim=1)
            
            return best_sim_vals, best_view_indices
        
        else:
            raise ValueError(f"Invalid multi-view type: {multi_view_type}. Valid options: 'best_views', 'average_views', 'cross_attention_views'")


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
