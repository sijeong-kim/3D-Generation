#!/usr/bin/env python3
"""
Particle Model Evaluation Script

This script loads N particle models from a directory and computes quantitative metrics,
logging them to CSV files. It's designed to evaluate pre-trained 3D Gaussian Splatting
models without running the full training process.

Usage:
    python evaluate_particles.py --config config.yaml --model_dir /path/to/models --output_dir /path/to/output

Features:
- Load multiple particle models from a directory
- Compute fidelity, diversity, and consistency metrics
- Generate multi-viewpoint renderings for evaluation
- Log results to CSV files with detailed statistics
- Support for different evaluation configurations
"""

import os
import argparse
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
from typing import List, Dict, Any, Optional, Tuple

# Import from the existing codebase
from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam
from metrics import MetricsCalculator
from visualizer import GaussianVisualizer


class ParticleEvaluator:
    """
    Evaluator for multiple particle models.
    
    This class loads N particle models from a directory and computes various
    quantitative metrics including fidelity, diversity, and consistency.
    """
    
    def __init__(self, config: Dict[str, Any], model_dir: str, output_dir: str):
        """
        Initialize the particle evaluator.
        
        Args:
            config: Configuration dictionary containing evaluation parameters
            model_dir: Directory containing particle model files
            output_dir: Directory to save evaluation results
        """
        self.config = config
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Camera setup
        self.cam = OrbitCamera(
            config.get('W', 512), 
            config.get('H', 512), 
            r=config.get('radius', 1.0), 
            fovy=config.get('fovy', 49.1)
        )
        
        # Initialize renderers
        self.renderers = []
        self.load_particle_models()
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator(
            opt=config,
            prompt=config.get('prompt', 'a photo of a hamburger'),
            device=self.device.type
        )
        
        # Initialize visualizer if needed
        if config.get('visualize', False):
            self.visualizer = GaussianVisualizer(
                opt=config, 
                renderers=self.renderers, 
                cam=self.cam
            )
        else:
            self.visualizer = None
    
    def load_particle_models(self):
        """Load all particle models from the model directory."""
        print(f"[INFO] Loading particle models from {self.model_dir}")
        
        # Find all model files
        model_files = []
        for file in os.listdir(self.model_dir):
            if file.endswith('.ply') and 'particle' in file:
                model_files.append(os.path.join(self.model_dir, file))
        
        if not model_files:
            raise ValueError(f"No particle model files found in {self.model_dir}")
        
        # Sort files to ensure consistent ordering
        model_files.sort()
        
        print(f"[INFO] Found {len(model_files)} particle models")
        
        # Load each model
        for i, model_path in enumerate(model_files):
            print(f"[INFO] Loading particle {i}: {os.path.basename(model_path)}")
            
            renderer = Renderer(sh_degree=self.config.get('sh_degree', 3))
            renderer.initialize(input=model_path)
            
            self.renderers.append(renderer)
        
        print(f"[INFO] Successfully loaded {len(self.renderers)} particle models")
    
    def render_multi_view_images(self, num_views: int = 8) -> torch.Tensor:
        """
        Render images from multiple viewpoints for all particles.
        
        Args:
            num_views: Number of viewpoints to render from
            
        Returns:
            torch.Tensor: Multi-view images with shape [V, N, 3, H, W]
        """
        print(f"[INFO] Rendering {num_views} viewpoints for {len(self.renderers)} particles")
        
        # Generate viewpoints
        viewpoints = self._generate_viewpoints(num_views)
        
        # Render images
        multi_view_images = []
        
        for view_idx, (ver, hor, radius) in enumerate(tqdm.tqdm(viewpoints, desc="Rendering viewpoints")):
            pose = orbit_camera(ver, hor, self.cam.radius + radius)
            cur_cam = MiniCam(
                pose, 
                self.config.get('W', 512), 
                self.config.get('H', 512), 
                self.cam.fovy, 
                self.cam.fovx, 
                self.cam.near, 
                self.cam.far
            )
            
            view_images = []
            for particle_idx, renderer in enumerate(self.renderers):
                # Set seed for consistent rendering
                torch.manual_seed(42 + view_idx * 1000 + particle_idx)
                
                # Render image
                bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
                out = renderer.render(cur_cam, bg_color=bg_color)
                
                image = out["image"].unsqueeze(0)  # [1, 3, H, W]
                view_images.append(image)
            
            # Stack particles for this view
            view_images = torch.cat(view_images, dim=0)  # [N, 3, H, W]
            multi_view_images.append(view_images)
        
        # Stack views
        multi_view_images = torch.stack(multi_view_images, dim=0)  # [V, N, 3, H, W]
        
        print(f"[INFO] Rendered images shape: {multi_view_images.shape}")
        return multi_view_images
    
    def _generate_viewpoints(self, num_views: int) -> List[Tuple[float, float, float]]:
        """
        Generate viewpoints for rendering.
        
        Args:
            num_views: Number of viewpoints to generate
            
        Returns:
            List of (elevation, azimuth, radius_offset) tuples
        """
        viewpoints = []
        
        # Generate viewpoints around the object
        for i in range(num_views):
            # Elevation: vary between -30 and 30 degrees
            ver = np.sin(i * 2 * np.pi / num_views) * 30
            
            # Azimuth: full 360 degrees
            hor = i * 360 / num_views
            
            # Small radius variation for more diverse views
            radius = np.sin(i * np.pi / num_views) * 0.1
            
            viewpoints.append((ver, hor, radius))
        
        return viewpoints
    
    def compute_metrics(self, multi_view_images: torch.Tensor) -> Dict[str, float]:
        """
        Compute all quantitative metrics.
        
        Args:
            multi_view_images: Multi-view images with shape [V, N, 3, H, W]
            
        Returns:
            Dictionary containing all computed metrics
        """
        print("[INFO] Computing quantitative metrics...")
        
        metrics = {}
        
        # Fidelity (text-image alignment)
        fidelity_mean, fidelity_std = self.metrics_calculator.compute_clip_fidelity_in_multi_viewpoints_stats(
            multi_view_images
        )
        metrics['fidelity_mean'] = fidelity_mean
        metrics['fidelity_std'] = fidelity_std
        
        # Inter-particle diversity
        diversity_mean, diversity_std = self.metrics_calculator.compute_inter_particle_diversity_in_multi_viewpoints_stats(
            multi_view_images
        )
        metrics['inter_particle_diversity_mean'] = diversity_mean
        metrics['inter_particle_diversity_std'] = diversity_std
        
        # Cross-view consistency
        consistency_mean, consistency_std = self.metrics_calculator.compute_cross_view_consistency_stats(
            multi_view_images
        )
        metrics['cross_view_consistency_mean'] = consistency_mean
        metrics['cross_view_consistency_std'] = consistency_std
        
        # LPIPS metrics (if enabled)
        if self.config.get('enable_lpips', False):
            lpips_inter_mean, lpips_inter_std, lpips_consistency_mean, lpips_consistency_std = \
                self.metrics_calculator.compute_lpips_inter_and_consistency(multi_view_images)
            
            metrics['lpips_inter_mean'] = lpips_inter_mean
            metrics['lpips_inter_std'] = lpips_inter_std
            metrics['lpips_consistency_mean'] = lpips_consistency_mean
            metrics['lpips_consistency_std'] = lpips_consistency_std
        
        return metrics
    
    def save_rendered_images(self, multi_view_images: torch.Tensor, save_dir: str):
        """
        Save rendered images for visualization.
        
        Args:
            multi_view_images: Multi-view images with shape [V, N, 3, H, W]
            save_dir: Directory to save images
        """
        if not self.config.get('save_rendered_images', False):
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"[INFO] Saving rendered images to {save_dir}")
        
        # Convert to numpy and save
        images_np = multi_view_images.detach().cpu().numpy()
        
        for view_idx in range(images_np.shape[0]):
            for particle_idx in range(images_np.shape[1]):
                image = images_np[view_idx, particle_idx]  # [3, H, W]
                
                # Convert to uint8
                image = (image * 255).astype(np.uint8)
                image = np.transpose(image, (1, 2, 0))  # [H, W, 3]
                
                # Save image
                filename = f"view_{view_idx:03d}_particle_{particle_idx:03d}.png"
                filepath = os.path.join(save_dir, filename)
                
                import cv2
                cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    def log_metrics_to_csv(self, metrics: Dict[str, float], csv_path: str):
        """
        Log metrics to CSV file.
        
        Args:
            metrics: Dictionary of metrics to log
            csv_path: Path to CSV file
        """
        import csv
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = list(metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Write metrics
            writer.writerow(metrics)
        
        print(f"[INFO] Metrics logged to {csv_path}")
    
    def evaluate(self, num_views: int = 8, save_images: bool = False):
        """
        Run the complete evaluation pipeline.
        
        Args:
            num_views: Number of viewpoints to render
            save_images: Whether to save rendered images
        """
        print(f"[INFO] Starting evaluation of {len(self.renderers)} particles")
        print(f"[INFO] Configuration: {num_views} viewpoints, save_images={save_images}")
        
        # Render multi-view images
        multi_view_images = self.render_multi_view_images(num_views)
        
        # Save rendered images if requested
        if save_images:
            images_dir = os.path.join(self.output_dir, "rendered_images")
            self.save_rendered_images(multi_view_images, images_dir)
        
        # Compute metrics
        metrics = self.compute_metrics(multi_view_images)
        
        # Log metrics to CSV
        csv_path = os.path.join(self.output_dir, "evaluation_metrics.csv")
        self.log_metrics_to_csv(metrics, csv_path)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Number of particles: {len(self.renderers)}")
        print(f"Number of viewpoints: {num_views}")
        print(f"Image resolution: {multi_view_images.shape[-2]}x{multi_view_images.shape[-1]}")
        print()
        print("METRICS:")
        for key, value in metrics.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
        print("="*50)
        
        return metrics


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate particle models")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--model_dir", required=True, help="Directory containing particle models")
    parser.add_argument("--output_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--num_views", type=int, default=8, help="Number of viewpoints to render")
    parser.add_argument("--save_images", action="store_true", help="Save rendered images")
    parser.add_argument("--prompt", type=str, help="Override prompt from config")
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.prompt:
        config.prompt = args.prompt
    
    # Convert config to dict for compatibility
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Create evaluator
    evaluator = ParticleEvaluator(config_dict, args.model_dir, args.output_dir)
    
    # Run evaluation
    metrics = evaluator.evaluate(
        num_views=args.num_views,
        save_images=args.save_images
    )
    
    print(f"[INFO] Evaluation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
