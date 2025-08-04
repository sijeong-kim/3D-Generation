# metrics.py
import torch
import torch.nn.functional as F
import open_clip
import os
from typing import Tuple

class MetricsCalculator:
    def __init__(self, opt: dict = None, prompt: str = "a photo of a cat"):
        # Validate opt parameter
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt
        
        self.save_dir = os.path.join(self.opt.outdir, 'metrics')
        os.makedirs(self.save_dir, exist_ok=True)
            
        self.prompt = prompt

        # CLIP settings
        self.features_normalized = True
        self.model_name = "ViT-B-32"
        self.pretrained = "laion2b_s34b_b79k"
        self.device = "cuda"

        # Load CLIP
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, device=self.device)
        tokenizer = open_clip.get_tokenizer(self.model_name)

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        # Initialize CSV files
        if opt is not None and hasattr(opt, 'metrics') and opt.metrics:
            self.metrics_csv_path = os.path.join(self.save_dir, "quantitative_metrics.csv")
            self.metrics_csv_file = open(self.metrics_csv_path, "w")
            self.metrics_csv_file.write("step,diversity,fidelity\n")
        
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
            images: [N, V, 3, H, W] in [0, 1]
        Returns:
            float: Mean cosine similarity between CLIP image and text embeddings
        """
        # Ensure images are on the correct device
        images = images.to(self.device)
        
        # Preprocess images for CLIP (resizing + normalization on GPU)
        images_clip = self._preprocess_images_tensor(images)  # [N, V, 3, 224, 224]

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

        return similarities.mean().item()
    
    @torch.no_grad()
    def select_best_views_by_clip_fidelity(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized: Select best view per particle based on CLIP similarity.

        Args:
            images: [N, V, 3, H, W] in [0, 1]

        Returns:
            best_images: [N, 3, H, W]
            best_similarities: [N]
        """
        N, V, C, H, W = images.shape
        device = images.device

        # Flatten views for batch CLIP processing
        images_flat = images.view(N * V, C, H, W)
        images_clip = self._preprocess_images_tensor(images_flat)  # [N*V, 3, 224, 224]

        # Encode image features
        image_features = self.model.encode_image(images_clip)
        image_features = F.normalize(image_features, dim=1)

        # Encode repeated prompt
        tokens = self.tokenizer(self.prompt).to(device)
        tokens = tokens.expand(N * V, -1)
        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=1)

        # Compute similarity
        sims = (image_features * text_features).sum(dim=1)  # [N*V]
        sims = sims.view(N, V)  # [N, V]

        # Find best similarity and corresponding view index
        best_sim_vals, best_view_indices = sims.max(dim=1)  # [N]

        # Select best images (V>=1)
        best_images = images[torch.arange(N, device=device), best_view_indices] # [N, 3, H, W]

        return best_images, best_sim_vals


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    from feature_extractor import DINOv2FeatureExtractor
    from sd_utils import StableDiffusion
    from sd_utils import seed_everything
    import torchvision.utils as vutils
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="a small saguaro cactus planted in a clay pot")
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device("cuda")

    N = 4
    resolution = 512
    prompts = [opt.prompt] * N
    sd = StableDiffusion(device=device, fp16=False)
    
    # Generate images one by one to avoid batch size mismatch
    images_list = []
    for prompt in prompts: 
        img = sd.prompt_to_img([prompt], negative_prompts="", height=resolution, width=resolution) # numpy array [1, H, W, 3], [0, 255]
        # Convert numpy array to tensor and normalize to [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        # Convert from [1, H, W, 3] to [1, 3, H, W]
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        images_list.append(img_tensor)
    
    images = torch.cat(images_list, dim=0) # [N, 3, H, W]

    # Visualize the generated images
    print(f"Generated {N} images with prompt: '{opt.prompt}'")
    print(f"Images shape: {images.shape}")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Save individual images
    for i, img in enumerate(images):
        vutils.save_image(img, f"outputs/generated_image_{i+1}.png")
    
    # Create and save a grid of all images
    grid = vutils.make_grid(images, nrow=2, padding=2, normalize=False)
    vutils.save_image(grid, f"outputs/generated_images_grid.png")
    
    # Display the grid using matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Generated Images for: '{opt.prompt}'")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("outputs/generated_images_plot.png", dpi=150, bbox_inches='tight')
    plt.show()

    feature_extractor = DINOv2FeatureExtractor(device=device)
    features = feature_extractor(images) # [N, 768]
    
    metrics = MetricsCalculator(save_dir="outputs", opt=None, prompt=opt.prompt)
    diversity = metrics.compute_rlsd_diversity(features)
    fidelity = metrics.compute_clip_fidelity(images) 
    
    print(f"CLIP Fidelity: {fidelity:.4f}")
    print(f"RLSD Diversity: {diversity:.4f}")
    
    # Analysis
    print(f"\n=== METRICS ANALYSIS ===")
    print(f"Prompt: '{opt.prompt}'")
    print(f"Number of images: {N}")
    print(f"CLIP Fidelity: {fidelity:.4f} (higher = better alignment with prompt)")
    print(f"RLSD Diversity: {diversity:.4f} (higher = more diverse images)")
    
    if fidelity > 0.25:
        print("✓ Good fidelity - images match the prompt well")
    else:
        print("⚠ Low fidelity - images may not match the prompt well")
        
    if diversity > 0.3:
        print("✓ Good diversity - images are sufficiently different")
    else:
        print("⚠ Low diversity - images are very similar to each other")
        
    print(f"\nImages saved to:")
    print(f"  - Individual images: outputs/generated_image_*.png")
    print(f"  - Grid view: outputs/generated_images_grid.png")
    print(f"  - Plot: outputs/generated_images_plot.png")
