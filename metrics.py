# metrics.py
import torch
import torch.nn.functional as F
import open_clip
import os
from typing import Tuple

# Import transformers conditionally
try:
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class MetricsCalculator:
    def __init__(self, opt: dict = None, prompt: str = "a photo of a cat", clip_metrics_type: str = None):
        # Validate opt parameter
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt
        
        self.save_dir = os.path.join(self.opt.outdir, 'metrics')
        os.makedirs(self.save_dir, exist_ok=True)
            
        self.prompt = prompt
        
        if clip_metrics_type is not None:
            self.clip_metrics_type = clip_metrics_type
        else:
            self.clip_metrics_type = self.opt.clip_metrics_type


        if self.clip_metrics_type == "openclip":
            # OpenCLIP settings
            self.features_normalized = True
            self.model_name = "ViT-B-32"
            self.pretrained = "laion2b_s34b_b79k"
            self.device = "cuda"

            # Load OpenCLIP
            model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, device=self.device)
            tokenizer = open_clip.get_tokenizer(self.model_name)

            self.model = model
            self.preprocess = preprocess
            self.tokenizer = tokenizer
        elif self.clip_metrics_type == "hf_clip":
            # HF CLIP settings
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library is required for hf_clip but not installed. Install with: pip install transformers")
            
            self.features_normalized = True
            self.model_name = "openai/clip-vit-large-patch14"
            self.device = "cuda"

            # Load HF CLIP
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

            # For HF CLIP, we don't pre-tokenize since we'll do it per batch
            self.tokenizer = None
            self.preprocess = None
        else:
            raise ValueError(f"Invalid clip metrics type: {self.clip_metrics_type}")
        
        # Initialize CSV files
        if opt is not None and hasattr(opt, 'metrics') and opt.metrics:
            self.metrics_csv_path = os.path.join(self.save_dir, "quantitative_metrics.csv")
            self.metrics_csv_file = open(self.metrics_csv_path, "w")
            self.metrics_csv_file.write(f"step,diversity,fidelity({self.clip_metrics_type})\n")
        
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
        
        if self.clip_metrics_type == "openclip":
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

        elif self.clip_metrics_type == "hf_clip":
            # HuggingFace CLIP method - direct tensor processing
            # Process images directly as tensors (much faster than PIL conversion)
            
            # Resize images to 224x224 and convert to [0, 255] range
            images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            images_uint8 = (images_resized * 255).clamp(0, 255).to(torch.uint8)
            
            # Process text
            text_inputs = self.processor.tokenizer(
                [self.prompt] * images.shape[0],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Process images directly as tensors
            vision_inputs = self.processor.image_processor(
                images_uint8.cpu(),  # Pass tensor directly for efficiency
                return_tensors="pt"
            )
            
            # Move vision inputs to device
            pixel_values = vision_inputs['pixel_values'].to(self.device)
            
            # Get embeddings
            image_features = self.model.get_image_features(pixel_values)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize & similarity  
            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
            similarities = (image_features * text_features).sum(dim=1)
        
        else:
            raise ValueError(f"Invalid clip metrics type: {self.clip_metrics_type}")

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

        if self.clip_metrics_type == "openclip":
            # OpenCLIP method
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
            
        elif self.clip_metrics_type == "hf_clip":
            # HuggingFace CLIP method - direct tensor processing
            # Process images directly as tensors (much faster than PIL conversion)
            
            # Resize images to 224x224 and convert to [0, 255] range
            images_resized = F.interpolate(images_flat, size=(224, 224), mode='bilinear', align_corners=False)
            images_uint8 = (images_resized * 255).clamp(0, 255).to(torch.uint8)
            
            # Process text
            text_inputs = self.processor.tokenizer(
                [self.prompt] * (N * V),
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            
            # Process images directly as tensors
            vision_inputs = self.processor.image_processor(
                images_uint8,  # Pass tensor directly for efficient processing
                return_tensors="pt"
            )
            
            # Move vision inputs to device
            pixel_values = vision_inputs['pixel_values'].to(device)
            
            # Get embeddings
            image_features = self.model.get_image_features(pixel_values)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize & similarity  
            image_features = F.normalize(image_features, dim=1)
            text_features = F.normalize(text_features, dim=1)
            sims = (image_features * text_features).sum(dim=1)  # [N*V]
            
        else:
            raise ValueError(f"Invalid clip metrics type: {self.clip_metrics_type}")

        sims = sims.view(N, V)  # [N, V]

        # Find best similarity and corresponding view index
        best_sim_vals, best_view_indices = sims.max(dim=1)  # [N]

        # Select best images (V>=1)
        best_images = images[torch.arange(N, device=device), best_view_indices] # [N, 3, H, W]

        return best_images, best_sim_vals


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Try to import optional dependencies for full test
    try:
        from feature_extractor import DINOv2FeatureExtractor
        from guidance.sd_utils import StableDiffusion, seed_everything
        import torchvision.utils as vutils
        has_full_deps = True
    except ImportError as e:
        print(f"Warning: Some dependencies not available ({e}). Running simplified test.")
        has_full_deps = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="a small saguaro cactus planted in a clay pot")
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

    # Generate test images
    if opt.use_synthetic or not has_full_deps:
        print("Creating synthetic test images...")
        # Create synthetic images with different patterns
        images = torch.zeros(N, 3, resolution, resolution, device=device)
        for i in range(N):
            # Create different colored patterns for each image
            if i == 0:  # Red gradient
                images[i, 0] = torch.linspace(0, 1, resolution, device=device).unsqueeze(0).repeat(resolution, 1)
            elif i == 1:  # Green checkerboard
                x, y = torch.meshgrid(torch.arange(resolution, device=device), torch.arange(resolution, device=device), indexing='ij')
                images[i, 1] = ((x // 32) + (y // 32)) % 2
            elif i == 2:  # Blue circle
                center = resolution // 2
                y, x = torch.meshgrid(torch.arange(resolution, device=device), torch.arange(resolution, device=device), indexing='ij')
                dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
                images[i, 2] = (dist < resolution // 4).float()
            else:  # Mixed pattern
                images[i] = torch.rand(3, resolution, resolution, device=device)
        
        # Create multi-view images (same images repeated across views)
        multi_view_images = images.unsqueeze(1).repeat(1, V, 1, 1, 1)  # [N, V, 3, H, W]
        
    else:
        print("Generating images using Stable Diffusion...")
        sd = StableDiffusion(device=device, fp16=False)
        
        # Generate images one by one to avoid batch size mismatch
        images_list = []
        prompts = [opt.prompt] * N
        for prompt in prompts: 
            img = sd.prompt_to_img([prompt], negative_prompts="", height=resolution, width=resolution)
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            images_list.append(img_tensor)
        
        images = torch.cat(images_list, dim=0).to(device)  # [N, 3, H, W]
        
        # Create slight variations for multi-view test
        multi_view_images = torch.zeros(N, V, 3, resolution, resolution, device=device)
        for i in range(N):
            for v in range(V):
                # Add slight noise/rotation for different views
                noise = torch.randn_like(images[i]) * 0.05
                multi_view_images[i, v] = torch.clamp(images[i] + noise, 0, 1)

    print(f"Test images shape: {images.shape}")
    print(f"Multi-view images shape: {multi_view_images.shape}")

    # Create output directory
    os.makedirs(opt.outdir, exist_ok=True)
    print(f"Using output directory: {opt.outdir}")
    
    # Save test images for visualization
    if has_full_deps:
        for i, img in enumerate(images):
            vutils.save_image(img, f"{opt.outdir}/test_image_{i+1}.png")
        grid = vutils.make_grid(images, nrow=2, padding=2, normalize=False)
        vutils.save_image(grid, f"{opt.outdir}/test_images_grid.png")

    # Generate features once for both CLIP methods (for diversity calculation)
    if has_full_deps:
        print("Loading DINOv2 feature extractor...")
        feature_extractor = DINOv2FeatureExtractor(device=device)
        features = feature_extractor(images)
        print(f"Extracted DINOv2 features: {features.shape}")
    else:
        # Create synthetic features with guaranteed diversity
        print("Creating synthetic features with controlled diversity...")
        features = torch.zeros(N, 768, device=device)
        
        # Create deliberately different feature vectors
        for i in range(N):
            if i == 0:
                # First vector: mostly positive values in first half
                features[i, :384] = torch.randn(384, device=device) + 2.0
                features[i, 384:] = torch.randn(384, device=device) - 0.5
            elif i == 1:
                # Second vector: mostly negative values in first half  
                features[i, :384] = torch.randn(384, device=device) - 2.0
                features[i, 384:] = torch.randn(384, device=device) + 0.5
            elif i == 2:
                # Third vector: mostly positive in second half
                features[i, :384] = torch.randn(384, device=device) - 0.5
                features[i, 384:] = torch.randn(384, device=device) + 2.0
            else:
                # Fourth vector: mostly negative in second half
                features[i, :384] = torch.randn(384, device=device) + 0.5
                features[i, 384:] = torch.randn(384, device=device) - 2.0
        
        # Add some noise to make it more realistic
        features += torch.randn_like(features) * 0.1
        print(f"Generated synthetic features: {features.shape}")
        
        # Debug: Check similarity between synthetic features
        features_norm = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features_norm, features_norm.T)
        print(f"Feature similarity matrix (diagonal should be ~1.0):")
        print(sim_matrix.cpu().numpy())

    # Test both CLIP methods
    print(f"\n{'='*60}")
    print("TESTING BOTH CLIP METHODS")
    print(f"{'='*60}")
    print(f"Prompt: '{opt.prompt}'")
    print(f"Number of images: {N}")
    print(f"Image resolution: {resolution}x{resolution}")

    results = {}
    
    for clip_type in ["openclip", "hf_clip"]:
        print(f"\n{'-'*40}")
        print(f"Testing {clip_type.upper()}")
        print(f"{'-'*40}")
        
        try:
            # Create config for current CLIP type
            class TestOpt:
                def __init__(self, clip_type, outdir):
                    self.clip_metrics_type = clip_type
                    self.outdir = outdir
                    self.metrics = False  # Don't initialize CSV files for testing
            
            test_opt = TestOpt(clip_type, opt.outdir)
            metrics = MetricsCalculator(opt=test_opt, prompt=opt.prompt)
            
            # Test 1: compute_clip_fidelity with single images
            print("Testing compute_clip_fidelity...")
            fidelity = metrics.compute_clip_fidelity(images)
            print(f"  Fidelity Score: {fidelity:.4f}")
            
            # Test 2: select_best_views_by_clip_fidelity with multi-view images
            print("Testing select_best_views_by_clip_fidelity...")
            best_images, best_similarities = metrics.select_best_views_by_clip_fidelity(multi_view_images)
            print(f"  Best images shape: {best_images.shape}")
            print(f"  Best similarities shape: {best_similarities.shape}")
            print(f"  Best similarities: {best_similarities.cpu().numpy()}")
            print(f"  Mean best similarity: {best_similarities.mean().item():.4f}")
            
            # Test 3: compute_rlsd_diversity (this should be the same for both since it uses the same features)
            print("Testing compute_rlsd_diversity...")
            diversity = metrics.compute_rlsd_diversity(features)
            if has_full_deps:
                print(f"  Diversity Score: {diversity:.4f}")
            else:
                print(f"  Diversity Score (synthetic features): {diversity:.4f}")
            
            # Store results for comparison
            results[clip_type] = {
                'fidelity': fidelity,
                'best_similarities': best_similarities.cpu().numpy(),
                'mean_best_similarity': best_similarities.mean().item(),
                'diversity': diversity
            }
            
            print(f"✓ {clip_type.upper()} test completed successfully")
            
        except Exception as e:
            print(f"✗ {clip_type.upper()} test failed: {e}")
            import traceback
            traceback.print_exc()
            results[clip_type] = None

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN CLIP METHODS")
    print(f"{'='*60}")
    
    if results["openclip"] is not None and results["hf_clip"] is not None:
        openclip_res = results["openclip"]
        hf_clip_res = results["hf_clip"]
        
        print(f"{'Metric':<25} {'OpenCLIP':<15} {'HF CLIP':<15} {'Difference':<15}")
        print(f"{'-'*70}")
        
        # Fidelity comparison
        fid_diff = abs(openclip_res['fidelity'] - hf_clip_res['fidelity'])
        print(f"{'Fidelity':<25} {openclip_res['fidelity']:<15.4f} {hf_clip_res['fidelity']:<15.4f} {fid_diff:<15.4f}")
        
        # Mean best similarity comparison
        sim_diff = abs(openclip_res['mean_best_similarity'] - hf_clip_res['mean_best_similarity'])
        print(f"{'Mean Best Similarity':<25} {openclip_res['mean_best_similarity']:<15.4f} {hf_clip_res['mean_best_similarity']:<15.4f} {sim_diff:<15.4f}")
        
        # Diversity comparison (should be identical since it uses the same features)
        div_diff = abs(openclip_res['diversity'] - hf_clip_res['diversity'])
        print(f"{'Diversity':<25} {openclip_res['diversity']:<15.4f} {hf_clip_res['diversity']:<15.4f} {div_diff:<15.4f}")
        
        print(f"\n{'Analysis:'}")
        print(f"- Fidelity difference: {fid_diff:.4f} {'(similar)' if fid_diff < 0.1 else '(different)'}")
        print(f"- Similarity difference: {sim_diff:.4f} {'(similar)' if sim_diff < 0.1 else '(different)'}")
        print(f"- Diversity difference: {div_diff:.4f} {'(identical - expected)' if div_diff < 0.001 else '(different - this suggests a bug!)'}")
        
        # Per-image similarity comparison
        print(f"\nPer-image best similarities:")
        print(f"{'Image':<8} {'OpenCLIP':<12} {'HF CLIP':<12} {'Difference':<12}")
        print(f"{'-'*45}")
        for i in range(len(openclip_res['best_similarities'])):
            sim_oc = openclip_res['best_similarities'][i]
            sim_hf = hf_clip_res['best_similarities'][i]
            diff = abs(sim_oc - sim_hf)
            print(f"{'#' + str(i+1):<8} {sim_oc:<12.4f} {sim_hf:<12.4f} {diff:<12.4f}")
        
        # Correlation analysis
        correlation = np.corrcoef(openclip_res['best_similarities'], hf_clip_res['best_similarities'])[0, 1]
        print(f"\nCorrelation between methods: {correlation:.4f}")
        if correlation > 0.8:
            print("✓ High correlation - methods agree well on relative rankings")
        elif correlation > 0.5:
            print("⚠ Moderate correlation - some differences in rankings")
        else:
            print("✗ Low correlation - significant differences in rankings")
            
    else:
        print("Cannot compare - one or both methods failed")
        if results["openclip"] is None:
            print("✗ OpenCLIP failed")
        if results["hf_clip"] is None:
            print("✗ HuggingFace CLIP failed")

    # Save results summary to file
    summary_path = os.path.join(opt.outdir, "metrics_comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("CLIP METHODS COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Prompt: '{opt.prompt}'\n")
        f.write(f"Number of images: {N}\n")
        f.write(f"Image resolution: {resolution}x{resolution}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Test type: {'Synthetic' if opt.use_synthetic else 'Generated'}\n")
        f.write(f"Feature type: {'DINOv2' if has_full_deps else 'Synthetic'}\n\n")
        
        if results["openclip"] is not None and results["hf_clip"] is not None:
            openclip_res = results["openclip"]
            hf_clip_res = results["hf_clip"]
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Metric':<25} {'OpenCLIP':<15} {'HF CLIP':<15} {'Difference':<15}\n")
            f.write("-" * 70 + "\n")
            
            fid_diff = abs(openclip_res['fidelity'] - hf_clip_res['fidelity'])
            sim_diff = abs(openclip_res['mean_best_similarity'] - hf_clip_res['mean_best_similarity'])
            div_diff = abs(openclip_res['diversity'] - hf_clip_res['diversity'])
            
            f.write(f"{'Fidelity':<25} {openclip_res['fidelity']:<15.4f} {hf_clip_res['fidelity']:<15.4f} {fid_diff:<15.4f}\n")
            f.write(f"{'Mean Best Similarity':<25} {openclip_res['mean_best_similarity']:<15.4f} {hf_clip_res['mean_best_similarity']:<15.4f} {sim_diff:<15.4f}\n")
            f.write(f"{'Diversity':<25} {openclip_res['diversity']:<15.4f} {hf_clip_res['diversity']:<15.4f} {div_diff:<15.4f}\n")
            
            f.write(f"\nANALYSIS:\n")
            f.write(f"- Fidelity difference: {fid_diff:.4f} {'(similar)' if fid_diff < 0.1 else '(different)'}\n")
            f.write(f"- Similarity difference: {sim_diff:.4f} {'(similar)' if sim_diff < 0.1 else '(different)'}\n")
            f.write(f"- Diversity difference: {div_diff:.4f} {'(identical - expected)' if div_diff < 0.001 else '(different - this suggests a bug!)'}\n")
            
            correlation = np.corrcoef(openclip_res['best_similarities'], hf_clip_res['best_similarities'])[0, 1]
            f.write(f"- Correlation between methods: {correlation:.4f}\n")
            if correlation > 0.8:
                f.write("  ✓ High correlation - methods agree well on relative rankings\n")
            elif correlation > 0.5:
                f.write("  ⚠ Moderate correlation - some differences in rankings\n")
            else:
                f.write("  ✗ Low correlation - significant differences in rankings\n")
        else:
            f.write("RESULTS: Test failed - one or both methods encountered errors\n")
            if results["openclip"] is None:
                f.write("- OpenCLIP failed\n")
            if results["hf_clip"] is None:
                f.write("- HuggingFace CLIP failed\n")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Results summary saved to: {summary_path}")
    if has_full_deps:
        print(f"Test images saved to:")
        print(f"  - Individual images: {opt.outdir}/test_image_*.png")
        print(f"  - Grid view: {opt.outdir}/test_images_grid.png")


