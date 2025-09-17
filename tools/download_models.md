# Troubleshooting Guide

## Network Connectivity Issues

### Problem
If you encounter errors like:
```
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub and we cannot find the requested files in the local cache. Please check your connection and try again or make sure your Internet connection is on.
```

This indicates that the models cannot be downloaded from Hugging Face Hub due to network connectivity issues.

### Solutions

#### 1. Download Models Offline (Recommended)
Run the model download script to cache all required models locally:

```bash
python download_models.py
```

This script will download:
- Stable Diffusion models (2.1, 2.0, 1.5)
- CLIP models (ViT-L-14, ViT-bigG-14)
- DINO models (via timm and torch.hub)
- DINOv2 models (small, base, large, giant)
- LPIPS models (alex, vgg, squeeze)

#### 2. Check Internet Connection
Ensure you have a stable internet connection and can access:
- https://huggingface.co
- https://github.com

#### 3. Use VPN or Proxy
If you're behind a firewall or proxy, configure your environment:
```bash
export HTTP_PROXY="http://your-proxy:port"
export HTTPS_PROXY="http://your-proxy:port"
```

#### 4. Manual Model Download
If the automatic download fails, you can manually download models:

```python
from diffusers import StableDiffusionPipeline
from transformers import AutoModel, AutoImageProcessor

# Download Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")

# Download DINOv2
model = AutoModel.from_pretrained("facebook/dinov2-base")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
```

#### 5. Use Local Model Cache
The code now includes offline mode support. If models are already cached locally, they will be loaded automatically.

### Model Cache Locations
- **Hugging Face**: `~/.cache/huggingface/`
- **Torch Hub**: `~/.cache/torch/hub/`
- **OpenCLIP**: `~/.cache/open_clip/`

### Verification
To verify models are cached correctly:
```bash
ls ~/.cache/huggingface/hub/
ls ~/.cache/torch/hub/
```

### Additional Notes
- The kernel statistics logger has been added to monitor training progress
- All model loading now includes graceful fallback to local cache
- Error messages provide clear guidance on next steps

If you continue to experience issues, please check:
1. Available disk space (models require several GB)
2. File permissions in cache directories
3. Network firewall settings
