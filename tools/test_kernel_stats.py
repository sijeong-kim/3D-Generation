#!/usr/bin/env python3
"""
Test script for kernel statistics functionality.
"""

import torch
import numpy as np
from metrics import MetricsCalculator

def test_kernel_statistics():
    """Test the kernel statistics computation."""
    
    # Create a mock configuration
    opt = {
        "outdir": "./test_output",
        "metrics": True,
        "enable_lpips": False,
        "eval_clip_model_name": "ViT-L-14",
        "eval_clip_pretrained": "openai",
    }
    
    # Initialize metrics calculator
    calculator = MetricsCalculator(opt, prompt="a photo of a cat")
    
    # Create test data
    N = 4  # number of particles
    D = 768  # feature dimension
    
    # Create a symmetric positive definite kernel matrix
    torch.manual_seed(42)
    features = torch.randn(N, D)
    features = torch.nn.functional.normalize(features, dim=1)  # normalize features
    
    # Create kernel matrix (cosine similarity)
    kernel = torch.mm(features, features.t())  # [N, N]
    
    # Create kernel gradient (mock)
    kernel_grad = torch.randn(N, D)
    
    print("Test data shapes:")
    print(f"  kernel: {kernel.shape}")
    print(f"  kernel_grad: {kernel_grad.shape}")
    print(f"  features: {features.shape}")
    
    # Compute kernel statistics
    neff_mean, neff_std, row_sum_mean, row_sum_std, gradient_norm_mean, gradient_norm_std = calculator.compute_kernel_statistics(kernel, kernel_grad, features)
    
    print("\nKernel Statistics:")
    print(f"  Effective sample size (Neff): {neff_mean:.4f} ± {neff_std:.4f}")
    print(f"  Row sum: {row_sum_mean:.4f} ± {row_sum_std:.4f}")
    print(f"  Gradient norm: {gradient_norm_mean:.4f} ± {gradient_norm_std:.4f}")
    
    # Test logging
    calculator.log_kernel_stats(
        step=0,
        kernel_stats={
            "neff_mean": neff_mean,
            "neff_std": neff_std,
            "row_sum_mean": row_sum_mean,
            "row_sum_std": row_sum_std,
            "gradient_norm_mean": gradient_norm_mean,
            "gradient_norm_std": gradient_norm_std,
        }
    )
    
    print("\nTest completed successfully!")
    print("Check ./test_output/metrics/kernel_stats.csv for the logged data.")
    
    # Clean up
    calculator.close()

if __name__ == "__main__":
    test_kernel_statistics()
