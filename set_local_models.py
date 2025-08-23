#!/usr/bin/env python3
"""
Set environment variables to force local model loading.
Run this before training to ensure models are loaded from local cache.
"""

import os
import sys

def set_local_model_env():
    """Set environment variables for local model loading."""
    
    # Force HuggingFace to use local files only
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # Set cache directories
    home = os.path.expanduser("~")
    os.environ["HF_HOME"] = os.path.join(home, ".cache", "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(home, ".cache", "huggingface")
    
    # Force torch hub to use local cache
    os.environ["TORCH_HOME"] = os.path.join(home, ".cache", "torch")
    
    print("✅ Environment variables set for local model loading:")
    print(f"   HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
    print(f"   TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
    print(f"   HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"   TORCH_HOME: {os.environ.get('TORCH_HOME')}")

if __name__ == "__main__":
    set_local_model_env()
