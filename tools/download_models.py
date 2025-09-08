#!/usr/bin/env python3
"""
Script to download all required models for offline use.
This helps avoid network connectivity issues during training.
"""

import os
import sys
import torch
from pathlib import Path

def download_stable_diffusion_models():
    """Download Stable Diffusion models."""
    print("📥 Downloading Stable Diffusion models...")
    
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    
    # Only download the model you're actually using
    models = [
        "stabilityai/stable-diffusion-2-1-base",  # Default SD model
    ]
    
    for model_name in models:
        print(f"  Downloading {model_name}...")
        try:
            # Download pipeline
            pipe = StableDiffusionPipeline.from_pretrained(model_name)
            print(f"    ✅ Pipeline downloaded")
            
            # Download scheduler
            scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
            print(f"    ✅ Scheduler downloaded")
            
        except Exception as e:
            print(f"    ❌ Failed to download {model_name}: {e}")

def download_clip_models():
    """Download CLIP models."""
    print("📥 Downloading CLIP models...")
    
    try:
        import open_clip
        
        # Only download the CLIP model you're actually using
        models = [
            ("ViT-bigG-14", "laion2b_s39b_b160k"),  # Used in metrics
        ]
        
        for model_name, pretrained in models:
            print(f"  Downloading {model_name} ({pretrained})...")
            try:
                model, _, _ = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
                print(f"    ✅ {model_name} downloaded")
            except Exception as e:
                print(f"    ❌ Failed to download {model_name}: {e}")
                
    except ImportError:
        print("  ⚠️  open_clip not available, skipping CLIP models")

def download_dino_models():
    """Download DINO models."""
    print("📥 Downloading DINO models...")
    
    # Try timm first
    try:
        import timm
        print("  Downloading DINO via timm...")
        model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
        print("    ✅ DINO (timm) downloaded")
    except Exception as e:
        print(f"    ❌ Failed to download DINO via timm: {e}")
        
        # Try torch.hub as fallback
        try:
            print("  Downloading DINO via torch.hub...")
            model = torch.hub.load("facebookresearch/dino", "dino_vitb16", trust_repo=True)
            print("    ✅ DINO (torch.hub) downloaded")
        except Exception as e2:
            print(f"    ❌ Failed to download DINO via torch.hub: {e2}")

def download_dinov2_models():
    """Download DINOv2 models."""
    print("📥 Downloading DINOv2 models...")
    
    try:
        from transformers import AutoModel, AutoImageProcessor
        
        # Only download the DINOv2 model you're actually using
        models = [
            "facebook/dinov2-base",  # Used in feature_extractor_model_name
        ]
        
        for model_name in models:
            print(f"  Downloading {model_name}...")
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                print(f"    ✅ {model_name} downloaded")
            except Exception as e:
                print(f"    ❌ Failed to download {model_name}: {e}")
                
    except ImportError:
        print("  ⚠️  transformers not available, skipping DINOv2 models")

def download_lpips_models():
    """Download LPIPS models."""
    print("📥 Downloading LPIPS models...")
    
    try:
        import lpips
        
        nets = ["alex", "vgg", "squeeze"]
        for net in nets:
            print(f"  Downloading LPIPS ({net})...")
            try:
                lpips_model = lpips.LPIPS(net=net)
                print(f"    ✅ LPIPS ({net}) downloaded")
            except Exception as e:
                print(f"    ❌ Failed to download LPIPS ({net}): {e}")
                
    except ImportError:
        print("  ⚠️  lpips not available, skipping LPIPS models")

def main():
    """Download all required models."""
    print("🚀 Starting model download for offline use...")
    print("=" * 60)
    
    # Create cache directory if it doesn't exist
    cache_dir = Path.home() / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Cache directory: {cache_dir}")
    
    # Download only the models you're actually using
    download_stable_diffusion_models()
    print()
    
    download_clip_models()
    print()
    
    download_dino_models()
    print()
    
    download_dinov2_models()
    print()
    
    # download_lpips_models()
    print()
    
    print("=" * 60)
    print("✅ Model download completed!")
    print("💡 You can now run training with offline mode support.")
    print("📝 If you still encounter issues, check your internet connection and try again.")

if __name__ == "__main__":
    main()
