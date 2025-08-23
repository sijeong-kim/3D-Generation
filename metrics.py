"""
3D Gaussian Splatting Evaluation Metrics Module

This module provides comprehensive evaluation capabilities for 3D Gaussian Splatting models,
enabling quantitative assessment of generation quality, diversity, and consistency across
multiple viewpoints and particles.

Key Components:
- MetricsModel: Evaluation-time model wrapper (OpenCLIP, DINO, LPIPS)
- MetricsCalculator: Batch processing and logging utilities for metrics computation

Key Features:
- Multi-model evaluation: CLIP for fidelity, DINO for diversity, LPIPS for perceptual similarity
- Robust device selection with automatic fallback (CUDA → MPS → CPU)
- Mixed precision support (fp16/fp32) for efficient computation
- Graceful handling of optional dependencies with clear warnings
- Comprehensive CSV logging with buffered, newline-safe output
- Multi-viewpoint analysis for 3D consistency evaluation
- Inter-particle diversity measurement for multi-particle scenarios

Supported Metrics:
- Fidelity: Text-image alignment using CLIP embeddings
- Diversity: Inter-particle variation using DINO features
- Consistency: Cross-viewpoint stability using DINO features
- Perceptual Similarity: LPIPS-based distance measurements

Supported Models:
- CLIP: ViT-L-14, ViT-bigG-14 (OpenAI/LAION weights)
- DINO: ViT-Base-Patch16-224 (timm/torch.hub)
- LPIPS: AlexNet, VGG, SqueezeNet backbones

Usage Example:
    >>> mm = MetricsModel(device="auto", eval_clip_model_name="ViT-bigG-14")
    >>> calc = MetricsCalculator({"metrics": True, "outdir": "./out"}, prompt="a photo of a hamburger")
    >>> # Compute metrics with calc.compute_* methods
    >>> # Log results with calc.log_metrics(...)

Author: [Your Name]
Date: [Date]
"""

from __future__ import annotations

import csv
import os
import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

try:
    import open_clip  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "open_clip is required. Install with `pip install open-clip-torch`.\n"
        f"Original error: {e}"
    )

# Optional: torchvision for transforms / interpolation flags
try:
    import torchvision.transforms as T  # type: ignore
    from torchvision.transforms import InterpolationMode  # type: ignore
    _HAS_TORCHVISION = True
except Exception:  # pragma: no cover
    T = None  # type: ignore
    InterpolationMode = None  # type: ignore
    _HAS_TORCHVISION = False

Tensor = torch.Tensor


def _pick_device(requested: Optional[str] = None) -> str:
    """Pick a sensible device string. Priority: requested → cuda → mps → cpu."""
    if requested and requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def _autocast_enabled(device: str) -> bool:
    return device in {"cuda", "mps"}


class MetricsModel:
    """
    Evaluation-time model wrapper for comprehensive 3D Gaussian Splatting metrics.
    
    This class manages multiple pre-trained models for different evaluation tasks:
    - CLIP models for text-image fidelity assessment
    - DINO models for diversity and consistency analysis
    - LPIPS networks for perceptual similarity measurements
    
    The class provides unified interfaces for feature extraction, preprocessing,
    and model management with automatic device placement and precision control.
    
    Attributes:
        device: PyTorch device for model execution
        precision: Computational precision ('fp16' or 'fp32')
        eval_clip_model: OpenCLIP model for evaluation
        eval_clip_tokenizer: CLIP text tokenizer
        eval_preprocess: CLIP image preprocessing pipeline
        dino_model: DINO ViT model for diversity features (optional)
        dino_preprocess: DINO image preprocessing pipeline (optional)
        lpips: LPIPS network for perceptual similarity (optional)
        _cached_text: Cached text prompt for efficiency
        _cached_text_feats: Cached text features for efficiency
    """

    DEFAULT_PRETRAIN = {
        "ViT-L-14": "openai",
        "ViT-bigG-14": "laion2b_s39b_b160k",
    }

    def __init__(
        self,
        device: str = "auto",
        eval_clip_model_name: str = "ViT-bigG-14",
        eval_clip_pretrained: Optional[str] = None,
        lpips_net: str = "alex",
        enable_lpips: bool = False,
        precision: str = "fp16",
    ):
        """
        Initialize the MetricsModel with evaluation models.
        
        Args:
            device: PyTorch device for model execution. Options:
                   - 'auto': Automatically select best available (CUDA → MPS → CPU)
                   - 'cuda': Use CUDA GPU if available
                   - 'cpu': Use CPU
                   - 'mps': Use Apple Metal Performance Shaders (macOS)
            eval_clip_model_name: OpenCLIP model variant for evaluation. Options:
                                 - 'ViT-L-14': Vision Transformer Large (768 features)
                                 - 'ViT-bigG-14': Vision Transformer Giant (1024 features)
            eval_clip_pretrained: Pre-trained weights for CLIP model. If None, uses default:
                                 - 'openai' for ViT-L-14
                                 - 'laion2b_s39b_b160k' for ViT-bigG-14
            lpips_net: LPIPS network architecture for perceptual similarity. Options:
                      - 'alex': AlexNet backbone (default, fastest)
                      - 'vgg': VGG backbone (more accurate)
                      - 'squeeze': SqueezeNet backbone (lightweight)
            enable_lpips: Whether to load LPIPS network for perceptual similarity metrics.
                         Disabled by default due to additional memory requirements.
            precision: Computational precision for model inference. Options:
                      - 'fp16': Half precision (faster, less memory)
                      - 'fp32': Full precision (more accurate)
        
        Note:
            - DINO model is automatically loaded if timm or torch.hub is available
            - All models are set to evaluation mode and frozen
            - Device selection follows priority: requested → CUDA → MPS → CPU
            - Mixed precision is automatically enabled on CUDA/MPS when precision='fp16'
        
        Raises:
            ValueError: If precision is not 'fp16' or 'fp32'
            ImportError: If open_clip is not installed
        """
        # Device & precision
        device_str = _pick_device(device)
        self.device = torch.device(device_str)
        if precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be 'fp16' or 'fp32'")
        self.precision = precision

        # Auto-select pretrained weights if not provided
        used_pretrained = eval_clip_pretrained or self.DEFAULT_PRETRAIN.get(
            eval_clip_model_name, "openai"
        )

        # Evaluation CLIP
        self.eval_clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            eval_clip_model_name, pretrained=used_pretrained, device=self.device
        )
        self.eval_clip_model.eval()

        # Tokenizer for CLIP
        self.eval_clip_tokenizer = open_clip.get_tokenizer(eval_clip_model_name)
        self.eval_preprocess = clip_preprocess if _HAS_TORCHVISION else None

        # RLSD Diversity (DINO)
        self.dino_model = None
        self.dino_preprocess = None
        try:
            import timm  # type: ignore

            self.dino_model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
            self.dino_model.to(self.device).eval()
            if _HAS_TORCHVISION:
                cfg = getattr(timm.data, "resolve_data_config")({}, model=self.dino_model)
                self.dino_preprocess = timm.data.create_transform(**cfg, is_training=False)
        except Exception as e1:
            try:
                # Torch Hub fallback (v1)
                self.dino_model = torch.hub.load(
                    "facebookresearch/dino", "dino_vitb16", trust_repo=True
                ).to(self.device)  # type: ignore
                self.dino_model.eval()
                if _HAS_TORCHVISION:
                    self.dino_preprocess = T.Compose(
                        [
                            T.Resize(256, interpolation=InterpolationMode.BICUBIC) if InterpolationMode else T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ]
                    )
            except Exception as e2:
                warnings.warn(
                    f"[metrics] DINO load failed; diversity features unavailable.\nPrimary: {e1}\nFallback: {e2}"
                )
                self.dino_model = None
                self.dino_preprocess = None

        # LPIPS (optional)
        self.lpips = None
        if enable_lpips:
            try:
                import lpips  # type: ignore

                self.lpips = lpips.LPIPS(net=lpips_net).to(self.device)
                self.lpips.eval()
            except Exception as e:
                warnings.warn(f"LPIPS requested but unavailable: {e}")
                self.lpips = None

        # Cache for text features
        self._cached_text: Optional[str] = None
        self._cached_text_feats: Optional[Tensor] = None

    # ------------------------ Preprocessing helpers ------------------------
    def preprocess_images_for_clip(self, images: Sequence) -> Tensor:
        """
        Preprocess images for CLIP model evaluation.
        
        Applies CLIP-specific preprocessing pipeline including resizing, normalization,
        and tensor conversion. Falls back to basic stacking if torchvision is unavailable.
        
        Args:
            images: Sequence of input images. Can be:
                   - List[PIL.Image]: PIL images
                   - List[torch.Tensor]: CHW or HWC tensors with values in [0, 1]
                   - Mixed sequence of PIL and tensors
        
        Returns:
            torch.Tensor: Preprocessed images with shape [B, 3, H, W] where:
                         - B: Batch size (number of images)
                         - 3: RGB channels
                         - H, W: Height and width (typically 224x224 for CLIP)
                         - Values normalized to CLIP statistics
        
        Note:
            - Images are automatically resized to CLIP's expected input size
            - Normalization uses CLIP-specific mean/std values
            - If torchvision is unavailable, performs basic tensor stacking
            - All images are moved to the model's device
        
        Raises:
            ValueError: If no images are provided
        """
        if not images:
            raise ValueError("No images provided")
        if self.eval_preprocess is None:
            stacked = torch.stack([
                img if isinstance(img, torch.Tensor) else torch.as_tensor(img) for img in images
            ])
            return stacked.to(self.device)
        processed = [self.eval_preprocess(img) for img in images]
        return torch.stack(processed).to(self.device)

    def preprocess_images_for_dino(self, images: Sequence) -> Tensor:
        """
        Preprocess images for DINO model evaluation.
        
        Applies DINO-specific preprocessing pipeline including resizing, normalization,
        and tensor conversion. Falls back to basic stacking if timm is unavailable.
        
        Args:
            images: Sequence of input images. Can be:
                   - List[PIL.Image]: PIL images
                   - List[torch.Tensor]: CHW or HWC tensors with values in [0, 1]
                   - Mixed sequence of PIL and tensors
        
        Returns:
            torch.Tensor: Preprocessed images with shape [B, 3, H, W] where:
                         - B: Batch size (number of images)
                         - 3: RGB channels
                         - H, W: Height and width (224x224 for DINO)
                         - Values normalized to ImageNet statistics
        
        Note:
            - Images are automatically resized to DINO's expected input size (224x224)
            - Normalization uses ImageNet mean/std values
            - If timm is unavailable, performs basic tensor stacking
            - All images are moved to the model's device
        
        Raises:
            ValueError: If no images are provided
        """
        if not images:
            raise ValueError("No images provided")
        if self.dino_preprocess is None:
            stacked = torch.stack([
                img if isinstance(img, torch.Tensor) else torch.as_tensor(img) for img in images
            ])
            return stacked.to(self.device)
        processed = [self.dino_preprocess(img) for img in images]
        return torch.stack(processed).to(self.device)

    # ------------------------ Encoding APIs ------------------------
    @torch.no_grad()
    def clip_encode_text(self, texts: Union[str, Sequence[str]]) -> Tensor:
        """
        Encode text prompts using CLIP text encoder.
        
        Converts text prompts to normalized feature embeddings using the CLIP text encoder.
        Supports both single text and batch processing with automatic mixed precision.
        
        Args:
            texts: Text prompt(s) to encode. Can be:
                  - str: Single text prompt
                  - List[str]: Multiple text prompts for batch processing
        
        Returns:
            torch.Tensor: Normalized text features with shape [B, D] where:
                         - B: Batch size (number of texts)
                         - D: Feature dimension (768 for ViT-L-14, 1024 for ViT-bigG-14)
                         - Features are L2-normalized for consistent magnitude
        
        Note:
            - Uses automatic mixed precision when precision='fp16' and device supports it
            - Text features are automatically L2-normalized
            - All processing is done with torch.no_grad() for efficiency
            - Tokens are automatically moved to the model's device
        
        Example:
            >>> model = MetricsModel()
            >>> features = model.clip_encode_text("a photo of a cat")
            >>> print(features.shape)  # torch.Size([1, 1024])
            >>> 
            >>> batch_features = model.clip_encode_text(["cat", "dog", "bird"])
            >>> print(batch_features.shape)  # torch.Size([3, 1024])
        """
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.eval_clip_tokenizer(list(texts)).to(self.device)
        use_amp = _autocast_enabled(self.device.type) and self.precision == "fp16"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            text_features = self.eval_clip_model.encode_text(tokens)
        return F.normalize(text_features.float(), dim=1)

    @torch.no_grad()
    def clip_encode_images(self, images_clip: Tensor) -> Tensor:
        """
        Encode images using CLIP vision encoder.
        
        Converts preprocessed images to normalized feature embeddings using the CLIP vision encoder.
        Images should be preprocessed using preprocess_images_for_clip() before calling this method.
        
        Args:
            images_clip: Preprocessed images with shape [B, 3, H, W] where:
                        - B: Batch size (number of images)
                        - 3: RGB channels
                        - H, W: Height and width (typically 224x224)
                        - Values should be normalized to CLIP statistics
        
        Returns:
            torch.Tensor: Normalized image features with shape [B, D] where:
                         - B: Batch size (number of images)
                         - D: Feature dimension (768 for ViT-L-14, 1024 for ViT-bigG-14)
                         - Features are L2-normalized for consistent magnitude
        
        Note:
            - Uses automatic mixed precision when precision='fp16' and device supports it
            - Image features are automatically L2-normalized
            - All processing is done with torch.no_grad() for efficiency
            - Images are automatically moved to the model's device
        
        Raises:
            ValueError: If images_clip is not [B, 3, H, W]
        
        Example:
            >>> model = MetricsModel()
            >>> # Preprocess images first
            >>> images = torch.randn(4, 3, 256, 256)  # Raw images
            >>> images_clip = model.preprocess_images_for_clip(images)
            >>> features = model.clip_encode_images(images_clip)
            >>> print(features.shape)  # torch.Size([4, 1024])
        """
        # images_clip: [B, 3, H, W], already normalized to CLIP stats
        if images_clip.ndim != 4 or images_clip.shape[1] != 3:
            raise ValueError("images_clip must be [B,3,H,W]")
        use_amp = _autocast_enabled(self.device.type) and self.precision == "fp16"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            image_features = self.eval_clip_model.encode_image(images_clip.to(self.device))
        return F.normalize(image_features.float(), dim=1)

    @torch.no_grad()
    def dino_encode_images(self, images_dino: Tensor) -> Tensor:
        """
        Encode images using DINO vision encoder.
        
        Converts preprocessed images to normalized feature embeddings using the DINO vision encoder.
        Images should be preprocessed using preprocess_images_for_dino() before calling this method.
        DINO features are particularly useful for diversity and consistency analysis.
        
        Args:
            images_dino: Preprocessed images with shape [B, 3, H, W] where:
                        - B: Batch size (number of images)
                        - 3: RGB channels
                        - H, W: Height and width (224x224 for DINO)
                        - Values should be normalized to ImageNet statistics
        
        Returns:
            torch.Tensor: Normalized DINO features with shape [B, D] where:
                         - B: Batch size (number of images)
                         - D: Feature dimension (768 for ViT-Base-Patch16-224)
                         - Features are L2-normalized for consistent magnitude
        
        Note:
            - Uses automatic mixed precision when precision='fp16' and device supports it
            - DINO features are automatically L2-normalized
            - All processing is done with torch.no_grad() for efficiency
            - Images are automatically moved to the model's device
            - Handles different output formats from timm and torch.hub DINO models
        
        Raises:
            RuntimeError: If DINO model is not available (timm/torch.hub not installed)
            ValueError: If images_dino is not [B, 3, H, W]
        
        Example:
            >>> model = MetricsModel()
            >>> # Preprocess images first
            >>> images = torch.randn(4, 3, 256, 256)  # Raw images
            >>> images_dino = model.preprocess_images_for_dino(images)
            >>> features = model.dino_encode_images(images_dino)
            >>> print(features.shape)  # torch.Size([4, 768])
        """
        if self.dino_model is None:
            raise RuntimeError("DINO model not available; install timm or enable hub load.")
        if images_dino.ndim != 4 or images_dino.shape[1] != 3:
            raise ValueError("images_dino must be [B,3,H,W]")

        use_amp = _autocast_enabled(self.device.type) and self.precision == "fp16"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
            feats = self.dino_model.forward_features(images_dino.to(self.device))
            # --- convert to [B, D] ---
            if isinstance(feats, dict):
                # timm common keys
                for k in ("x_norm_clstoken", "cls_token", "pooler_output", "last_hidden_state"):
                    if k in feats and isinstance(feats[k], torch.Tensor):
                        feats = feats[k]
                        break
                # fallback to first tensor in dict
                if isinstance(feats, dict):
                    feats = next(v for v in feats.values() if isinstance(v, torch.Tensor))

            if feats.ndim == 3:
                # [B, T, D] token sequence → if CLS, take first token, else mean
                feats = feats[:, 0] if feats.shape[1] >= 1 else feats.mean(dim=1)
            elif feats.ndim == 4:
                # [B, C, H, W] feature map → GAP
                feats = feats.mean(dim=(2, 3))

        # final L2 normalization (float32)
        feats = F.normalize(feats.float(), dim=-1)
        return feats  # [B, D]


    # ------------------------ Utilities ------------------------
    def get_or_cache_text_features(self, prompt: str) -> Tensor:
        """
        Get cached text features or compute and cache new ones.
        
        Efficiently retrieves text features for a given prompt, using cached results
        when available to avoid redundant computation. This is particularly useful
        when the same text prompt is used repeatedly (e.g., for batch evaluation).
        
        Args:
            prompt: Text prompt to encode.
        
        Returns:
            torch.Tensor: Normalized text features with shape [1, D] where:
                         - D: Feature dimension (768 for ViT-L-14, 1024 for ViT-bigG-14)
                         - Features are L2-normalized
        
        Note:
            - Features are cached in memory for the lifetime of the model instance
            - Cache is automatically invalidated when a different prompt is provided
            - Uses clip_encode_text() internally for feature computation
            - All processing is done with torch.no_grad() for efficiency
        
        Example:
            >>> model = MetricsModel()
            >>> # First call: computes and caches features
            >>> features1 = model.get_or_cache_text_features("a photo of a cat")
            >>> # Second call: returns cached features (faster)
            >>> features2 = model.get_or_cache_text_features("a photo of a cat")
            >>> torch.allclose(features1, features2)  # True
        """
        if self._cached_text == prompt and self._cached_text_feats is not None:
            return self._cached_text_feats
        feats = self.clip_encode_text(prompt)
        self._cached_text = prompt
        self._cached_text_feats = feats
        return feats

    def to(self, device: str) -> "MetricsModel":
        """
        Move all models to the specified device.
        
        Transfers all loaded models (CLIP, DINO, LPIPS) to the target device.
        This is useful for changing device placement after initialization.
        
        Args:
            device: Target device. Can be:
                   - 'auto': Automatically select best available
                   - 'cuda': Use CUDA GPU
                   - 'cpu': Use CPU
                   - 'mps': Use Apple Metal Performance Shaders
        
        Returns:
            MetricsModel: Self reference for method chaining.
        
        Note:
            - Device selection follows the same priority as initialization
            - All models are moved to the same device
            - Cached features are not automatically moved (will be recomputed if needed)
        
        Example:
            >>> model = MetricsModel(device='cpu')
            >>> model = model.to('cuda')  # Move to GPU
            >>> model = model.to('auto')  # Auto-select best device
        """
        device_str = _pick_device(device)
        self.device = torch.device(device_str)
        self.eval_clip_model.to(device_str)
        if self.dino_model is not None:
            self.dino_model.to(device_str)
        if self.lpips is not None:
            self.lpips.to(device_str)
        return self


class MetricsCalculator:
    """
    Comprehensive metrics calculator for 3D Gaussian Splatting evaluation.
    
    This class provides batch processing capabilities for computing and logging various
    evaluation metrics including fidelity, diversity, consistency, and efficiency metrics.
    It manages CSV logging, preprocessing pipelines, and metric computation workflows.
    
    Key Features:
    - Multi-metric computation: fidelity, diversity, consistency, perceptual similarity
    - Automated CSV logging with configurable intervals
    - Efficient batch processing with preprocessing pipelines
    - Memory and timing efficiency tracking
    - Support for both single-particle and multi-particle scenarios
    
    Attributes:
        opt: Configuration object (dict or argparse-like) containing evaluation parameters
        device: PyTorch device for computation
        save_dir: Directory for saving metric logs and outputs
        prompt: Text prompt for fidelity evaluation
        features_normalized: Whether features are pre-normalized
        train_clip_model: Training-time CLIP model (optional)
        train_clip_tokenizer: Training-time CLIP tokenizer (optional)
        train_clip_preprocess: Training-time CLIP preprocessing (optional)
        metrics_model: Evaluation-time models wrapper
        eval_preprocess: Evaluation preprocessing pipeline
        input_res: Input resolution for evaluation models
        metrics_csv_path: Path to quantitative metrics CSV file
        losses_csv_path: Path to losses CSV file
        efficiency_csv_path: Path to efficiency metrics CSV file
        _warned_metrics: Set of already warned metric names
        _metric_ranges: Expected ranges for metric validation
    """
    
    def __init__(
        self,
        opt: Any,
        prompt: str = "a photo of a hamburger",
        device: Optional[str] = None,
    ):
        """
        Initialize the MetricsCalculator with configuration and models.
        
        Args:
            opt: Configuration object containing evaluation parameters. Can be:
                 - dict: Dictionary with configuration key-value pairs
                 - argparse.Namespace: Command-line argument namespace
                 - Any object with attribute access
                 
                 Required fields:
                 - outdir (str): Output directory for logs and results
                 - metrics (bool): Whether to enable metrics computation
                 - enable_lpips (bool): Whether to enable LPIPS computation
                 
                 Optional fields:
                 - quantitative_metrics_interval (int): Logging interval for quantitative metrics
                 - losses_interval (int): Logging interval for loss values
                 - efficiency_interval (int): Logging interval for efficiency metrics
                 - eval_clip_model_name (str): CLIP model variant for evaluation
                 - eval_clip_pretrained (str): CLIP pre-trained weights
                 - init_train_clip (bool): Whether to initialize training-time CLIP
        
            prompt: Text prompt for fidelity evaluation. Used to compute CLIP text-image
                   similarity scores. Default: "a photo of a hamburger"
            
            device: PyTorch device for computation. If None, uses 'auto' selection.
                   Options: 'auto', 'cuda', 'cpu', 'mps'
        
        Note:
            - CSV files are automatically created with appropriate headers
            - Models are initialized based on configuration parameters
            - Preprocessing pipelines are set up for both CLIP and DINO
            - Metric ranges are defined for validation and debugging
        
        Raises:
            ValueError: If opt parameter is None
            ImportError: If required dependencies are not available
        """
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt
        
        # Initialize all attributes to None
        self.metrics_model = None
        self.metrics_writer, self.losses_writer, self.efficiency_writer, self.kernel_stats_writer = None, None, None, None
        self.metrics_csv_file, self.losses_csv_file, self.efficiency_csv_file, self.kernel_stats_csv_file = None, None, None, None
        self.clip_input_res, self.dino_input_res = None, None
        self.clip_mean, self.clip_std = None, None
        self.imagenet_mean, self.imagenet_std = None, None
        self._warned_metrics = None
        self._metric_ranges = None

        # Accessor for dict or namespace
        def _get(name, default=None):
            if isinstance(self.opt, dict):
                return self.opt.get(name, default)
            return getattr(self.opt, name, default)

        self._get = _get

        # Device selection
        device_str = _pick_device(device or "auto")
        self.device = torch.device(device_str)

        # Outputs
        outdir = _get("outdir", "./outputs")
        self.save_dir = os.path.join(outdir, "metrics")
        os.makedirs(self.save_dir, exist_ok=True)

        self.prompt = prompt

        # Feature normalization is already handled in MetricsModel.encode_*
        self.features_normalized = False

        # (Optional) training-time CLIP; keep if other parts of your pipeline use it
        if _get("init_train_clip", True):
            self.train_clip_model_name = "ViT-bigG-14"
            self.train_clip_pretrained = "laion2b_s39b_b160k"
            self.train_clip_model, _, self.train_clip_preprocess = open_clip.create_model_and_transforms(
                self.train_clip_model_name, pretrained=self.train_clip_pretrained, device=self.device
            )
            self.train_clip_tokenizer = open_clip.get_tokenizer(self.train_clip_model_name)
        else:
            self.train_clip_model = None
            self.train_clip_tokenizer = None
            self.train_clip_preprocess = None

        # Metrics models (evaluation-time): configurable CLIP + LPIPS
        eval_clip_model_name = _get("eval_clip_model_name", "ViT-bigG-14")
        eval_clip_pretrained = _get("eval_clip_pretrained", None)
        


        self.metrics_model = MetricsModel(
            device=self.device,
            eval_clip_model_name=eval_clip_model_name,
            eval_clip_pretrained=eval_clip_pretrained,
            enable_lpips=bool(_get("enable_lpips", False)),
        )
        

        
        
        # Preprocess configuration from the evaluation CLIP model
        self.eval_preprocess = self.metrics_model.eval_preprocess
        self.input_res = getattr(self.metrics_model.eval_clip_model.visual, "image_size", 224)

        # Per-model mean/std (CLIP defaults)
        self._init_preprocess_stats()

        # CSV setup (writers + headers)
        self.metrics_csv_path = os.path.join(self.save_dir, "quantitative_metrics.csv")
        self.losses_csv_path = os.path.join(self.save_dir, "losses.csv")
        self.efficiency_csv_path = os.path.join(self.save_dir, "efficiency.csv")
        self.kernel_stats_csv_path = os.path.join(self.save_dir, "kernel_stats.csv")

        # Open files in newline-safe mode; buffered writes
        self.metrics_csv_file = open(self.metrics_csv_path, "w", newline="", buffering=1)
        self.losses_csv_file = open(self.losses_csv_path, "w", newline="", buffering=1)
        self.efficiency_csv_file = open(self.efficiency_csv_path, "w", newline="", buffering=1)
        self.kernel_stats_csv_file = open(self.kernel_stats_csv_path, "w", newline="", buffering=1)

        self.metrics_writer = csv.writer(self.metrics_csv_file)
        self.losses_writer = csv.writer(self.losses_csv_file)
        self.efficiency_writer = csv.writer(self.efficiency_csv_file)
        self.kernel_stats_writer = csv.writer(self.kernel_stats_csv_file)

        enable_lpips = bool(_get("enable_lpips", False)) and self.metrics_model.lpips is not None
        if enable_lpips:
            self.metrics_writer.writerow([
                "step",
                "inter_particle_diversity_mean", "inter_particle_diversity_std",
                "cross_view_consistency_mean", "cross_view_consistency_std",
                "fidelity_mean", "fidelity_std",
                "lpips_inter_mean", "lpips_inter_std", "lpips_consistency_mean", "lpips_consistency_std",
            ])
        else:
            self.metrics_writer.writerow([
                "step",
                "inter_particle_diversity_mean", "inter_particle_diversity_std",
                "cross_view_consistency_mean", "cross_view_consistency_std",
                "fidelity_mean", "fidelity_std",
            ])

        self.losses_writer.writerow([
            "step", "attraction_loss", "repulsion_loss", "scaled_attraction_loss", "scaled_repulsion_loss", "total_loss",
            "scaled_repulsion_loss_ratio",
        ])

        # Explicit units: time in milliseconds, memory in megabytes
        self.efficiency_writer.writerow(["step", "time_ms", "memory_allocated_MB", "max_memory_allocated_MB"])  # noqa: N806

        # Kernel statistics header
        self.kernel_stats_writer.writerow([
            "step", "neff_mean", "neff_std", "row_sum_mean", "row_sum_std", 
            "gradient_norm_mean", "gradient_norm_std"
        ])

        self._warned_metrics = set()  # already warned metric names

        # expected range specs (adjust as needed)
        self._metric_ranges = {
            "inter_particle_diversity": (0.0, 2.0),            # 1 - mean_cos (normal [0,2])
            "cross_view_consistency": (-1.0, 1.0),             # mean cosine
            "fidelity": (-1.0, 1.0),                           # CLIP cosine
            "lpips": (0.0, 2.0),                               # safe range
            "time_ms": (0.0, float("inf")),
            "memory_MB": (0.0, float("inf")),
        }

    # ------------------------ lifecycle ------------------------
    def close(self) -> None:
        """Close any open CSV files."""
        
        for f in ("metrics_csv_file", "losses_csv_file", "efficiency_csv_file", "kernel_stats_csv_file"):
            if hasattr(self, f) and getattr(self, f):
                try:
                    getattr(self, f).close()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------ helpers ------------------------
    def get_clip_text_features(self) -> Tensor:
        return self.metrics_model.get_or_cache_text_features(self.prompt)

    def _should_log(self, name: str, step: int) -> bool:
        """Handle None/0/absent intervals safely (0/None -> never)."""
        val = self._get(name, None)
        if not isinstance(val, int) or val <= 0:
            return False
        return (step % val) == 0

    # ------------------------ Logging ------------------------
    @torch.no_grad()
    def log_quantitative_metrics(self, step: int, metrics: dict):
        if self.metrics_writer is None:
            return
        enable_lpips = bool(self._get("enable_lpips", False)) and self.metrics_model.lpips is not None
        base_row = [
            step,
            metrics.get("inter_particle_diversity_mean", ""), metrics.get("inter_particle_diversity_std", ""),
            metrics.get("cross_view_consistency_mean", ""), metrics.get("cross_view_consistency_std", ""),
            metrics.get("fidelity_mean", ""), metrics.get("fidelity_std", ""),
        ]
        if enable_lpips:
            base_row.extend([
                metrics.get("lpips_inter_mean", ""), metrics.get("lpips_inter_std", ""),
                metrics.get("lpips_consistency_mean", ""), metrics.get("lpips_consistency_std", ""),
            ])
        self.metrics_writer.writerow(base_row)

    @torch.no_grad()
    def log_losses(self, step: int, losses: dict):
        if self.losses_writer is None:
            return
        self.losses_writer.writerow([
            step,
            losses.get("attraction_loss", ""), losses.get("repulsion_loss", ""),
            losses.get("scaled_attraction_loss", ""), losses.get("scaled_repulsion_loss", ""),
            losses.get("total_loss", ""),
            losses.get("scaled_repulsion_loss_ratio", ""),
        ])

    @torch.no_grad()
    def log_efficiency(self, step: int, efficiency: dict):
        if self.efficiency_writer is None:
            return
        # Accept either 'time' or 'time_ms'; and memory keys in either *_MB or *_mb forms
        t_ms = efficiency.get("time_ms", efficiency.get("time", ""))
        mem = efficiency.get("memory_allocated_MB", efficiency.get("memory_allocated_mb", ""))
        mem_max = efficiency.get("max_memory_allocated_MB", efficiency.get("max_memory_allocated_mb", ""))
        self.efficiency_writer.writerow([step, t_ms, mem, mem_max])

    @torch.no_grad()
    def log_metrics(self, step: int, efficiency: dict, losses: dict, metrics: dict):
        if self.save_dir is None:
            return
        if self._should_log("quantitative_metrics_interval", step):
            self.log_quantitative_metrics(step=step, metrics=metrics)
        if self._should_log("losses_interval", step):
            self.log_losses(step=step, losses=losses)
        if self._should_log("efficiency_interval", step):
            self.log_efficiency(step=step, efficiency=efficiency)

    # ------------------------ Preprocess ------------------------
    def _init_preprocess_stats(self) -> None:
        # --- CLIP ---
        raw_size = getattr(self.metrics_model.eval_clip_model.visual, "image_size", 224)
        if isinstance(raw_size, int):
            self.clip_input_res = (raw_size, raw_size)
        elif isinstance(raw_size, (tuple, list)) and len(raw_size) == 2:
            self.clip_input_res = (int(raw_size[0]), int(raw_size[1]))
        else:
            self.clip_input_res = (224, 224)
        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

        # --- DINO(v1) / ImageNet ---
        self.dino_input_res = (224, 224)
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

    @torch.no_grad()
    def _preprocess_images(self, images: Tensor, *, backbone: str) -> Tensor:
        """Resize + normalize to model stats. `backbone` in {"clip","dino"}."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be [B,3,H,W]")
        if backbone == "dino":
            target_size = self.dino_input_res
            mean, std = self.imagenet_mean, self.imagenet_std
        else:
            target_size = self.clip_input_res
            mean, std = self.clip_mean, self.clip_std

        # Use bicubic if torchvision indicated; otherwise bilinear
        if _HAS_TORCHVISION and InterpolationMode is not None:
            images_resized = F.interpolate(
                images, size=target_size, mode="bicubic", align_corners=False, antialias=True  # type: ignore[arg-type]
            )
        else:
            images_resized = F.interpolate(images, size=target_size, mode="bilinear", align_corners=False)
        return (images_resized - mean) / std

    # ------------------------ Metrics ------------------------
    @torch.no_grad()
    def compute_inter_particle_diversity_in_multi_viewpoints_stats(
        self, multi_view_images: Tensor
    ) -> Tuple[float, float]:
        """
        Compute inter-particle diversity statistics across multiple viewpoints.
        
        This metric measures how different the generated particles are from each other
        across multiple viewpoints. Higher diversity indicates more varied and interesting
        generations. The metric is computed per viewpoint and then aggregated.
        
        Methodology:
        1. Extract DINO features for all particles from all viewpoints
        2. For each viewpoint, compute cosine similarity matrix between particles
        3. Calculate diversity as: 1 - mean(off-diagonal similarities)
        4. Aggregate across viewpoints using mean and standard deviation
        
        Args:
            multi_view_images: Multi-viewpoint images with shape [V, N, 3, H, W] where:
                              - V: Number of viewpoints
                              - N: Number of particles
                              - 3: RGB channels
                              - H, W: Image height and width
                              - Values should be in range [0, 1]
        
        Returns:
            Tuple[float, float]: (mean_diversity, std_diversity) where:
                                - mean_diversity: Average diversity across viewpoints
                                - std_diversity: Standard deviation of diversity across viewpoints
                                - Range: [0, 2] where higher values indicate more diversity
                                - 0: All particles are identical
                                - 2: All particles are maximally different
        
        Note:
            - Uses DINO features for diversity computation (better for visual diversity)
            - Features are automatically L2-normalized
            - Returns (0.0, 0.0) if N < 2 (insufficient particles for diversity)
            - Similarity values are clamped to [-1, 1] for numerical stability
        
        Example:
            >>> calculator = MetricsCalculator(opt, prompt="a cat")
            >>> images = torch.randn(8, 4, 3, 256, 256)  # 8 views, 4 particles
            >>> mean_div, std_div = calculator.compute_inter_particle_diversity_in_multi_viewpoints_stats(images)
            >>> print(f"Diversity: {mean_div:.3f} ± {std_div:.3f}")
        """
        if multi_view_images.ndim != 5 or multi_view_images.shape[2] != 3:
            raise ValueError("multi_view_images must be [V,N,3,H,W]")
        V, N = multi_view_images.shape[0], multi_view_images.shape[1]
        if N < 2:
            return 0.0, 0.0

        mv_flat = multi_view_images.view(-1, 3, multi_view_images.shape[-2], multi_view_images.shape[-1])
        mv_dino = self._preprocess_images(mv_flat, backbone="dino")  # [V*N, 3, R, R]
        multi_view_features = self.metrics_model.dino_encode_images(mv_dino).view(V, N, -1)

        if not self.features_normalized:
            multi_view_features = F.normalize(multi_view_features, dim=2)

        sim_matrix = torch.bmm(multi_view_features, multi_view_features.transpose(1, 2))  # [V, N, N]
        sim_matrix = sim_matrix.clamp(-1, 1) # clamp to [-1, 1]

        total_sum = sim_matrix.sum(dim=(1, 2))
        diag_sum = sim_matrix.diagonal(dim1=1, dim2=2).sum(dim=1)
        off_mean = (total_sum - diag_sum) / (N * (N - 1))
        div_v = 1.0 - off_mean
        mean_div = float(div_v.mean().item())
        std_div = float(div_v.std(unbiased=False).item()) if V > 1 else 0.0

        return mean_div, std_div  # [0,2]

    @torch.no_grad()
    def compute_cross_view_consistency_stats(
        self,
        multi_view_images: Tensor,
        *,
        return_per_particle: bool = False,
    ) -> Union[Tuple[float, float], Tuple[float, float, Tensor]]:
        """
        Compute cross-view consistency statistics across multiple viewpoints.
        
        This metric measures how consistent each particle appears across different viewpoints.
        Higher consistency indicates better 3D structure and viewpoint invariance.
        The metric is computed per particle and then aggregated.
        
        Methodology:
        1. Extract DINO features for all particles from all viewpoints
        2. For each particle, compute cosine similarity matrix between viewpoints
        3. Calculate consistency as: mean(off-diagonal similarities)
        4. Aggregate across particles using mean and standard deviation
        
        Args:
            multi_view_images: Multi-viewpoint images with shape [V, N, 3, H, W] where:
                              - V: Number of viewpoints
                              - N: Number of particles
                              - 3: RGB channels
                              - H, W: Image height and width
                              - Values should be in range [0, 1]
            return_per_particle: Whether to return per-particle consistency values.
                                If True, returns (mean, std, per_particle_consistency)
                                If False, returns (mean, std)
        
        Returns:
            Union[Tuple[float, float], Tuple[float, float, Tensor]]:
                - (mean_consistency, std_consistency): Aggregated statistics
                - (mean_consistency, std_consistency, per_particle_consistency): 
                  If return_per_particle=True, includes per-particle values
                
                Where:
                - mean_consistency: Average consistency across particles
                - std_consistency: Standard deviation of consistency across particles
                - per_particle_consistency: Tensor of shape [N] with per-particle values
                - Range: [-1, 1] where higher values indicate better consistency
                - 1: Perfect consistency across viewpoints
                - 0: No consistency (random appearance across viewpoints)
                - -1: Anti-consistent (opposite appearance across viewpoints)
        
        Note:
            - Uses DINO features for consistency computation (better for 3D structure)
            - Features are automatically L2-normalized
            - Returns (0.0, 0.0) if V < 2 or N == 0 (insufficient data)
            - Similarity values are clamped to [-1, 1] for numerical stability
        
        Example:
            >>> calculator = MetricsCalculator(opt, prompt="a cat")
            >>> images = torch.randn(8, 4, 3, 256, 256)  # 8 views, 4 particles
            >>> mean_cons, std_cons = calculator.compute_cross_view_consistency_stats(images)
            >>> print(f"Consistency: {mean_cons:.3f} ± {std_cons:.3f}")
            >>> 
            >>> # Get per-particle values
            >>> mean_cons, std_cons, per_particle = calculator.compute_cross_view_consistency_stats(
            ...     images, return_per_particle=True)
            >>> print(f"Per-particle consistency: {per_particle}")
        """
        if multi_view_images.ndim != 5 or multi_view_images.shape[2] != 3:
            raise ValueError("multi_view_images must be [V,N,3,H,W]")
        V, N = multi_view_images.shape[0], multi_view_images.shape[1]
        if V < 2 or N == 0:
            z = torch.zeros(N, device=self.device)
            out = (0.0, 0.0, z) if return_per_particle else (0.0, 0.0)
            return out  # type: ignore[return-value]

        mv_flat = multi_view_images.view(-1, 3, multi_view_images.shape[-2], multi_view_images.shape[-1])
        mv_dino = self._preprocess_images(mv_flat, backbone="dino")  # [V*N, 3, R, R]
        features = self.metrics_model.dino_encode_images(mv_dino).view(V, N, -1)

        if not self.features_normalized:
            features = F.normalize(features, dim=2)

        # [N, V, D]
        features_nv = features.permute(1, 0, 2).contiguous()
        sims_nv = torch.bmm(features_nv, features_nv.transpose(1, 2))  # [N, V, V]
        sims_nv = sims_nv.clamp(-1, 1) # clamp to [-1, 1] to avoid nan

        total_sum = sims_nv.sum(dim=(1, 2))         # [N]
        diag_sum = sims_nv.diagonal(dim1=1, dim2=2).sum(dim=1)  # [N]
        per_particle = (total_sum - diag_sum) / (V * (V - 1))   # [N]

        mean_consistency = float(per_particle.mean().item())
        std_consistency = float(per_particle.std(unbiased=False).item()) if N > 1 else 0.0

        if return_per_particle:
            return mean_consistency, std_consistency, per_particle
        return mean_consistency, std_consistency


    @torch.no_grad()
    def compute_clip_fidelity_in_multi_viewpoints_stats(
        self, images: Tensor, view_type: str = "mean"
    ) -> Tuple[float, float]:
        """
        Compute text-image CLIP fidelity statistics across multiple viewpoints.
        
        This metric measures how well the generated images match the text prompt
        across multiple viewpoints. Higher fidelity indicates better text-image alignment.
        The metric aggregates CLIP similarity scores across viewpoints using the specified method.
        
        Methodology:
        1. Extract CLIP features for all images from all viewpoints
        2. Compute cosine similarity between image features and text features
        3. Aggregate similarities across viewpoints using specified method
        4. Compute statistics across particles
        
        Args:
            images: Multi-viewpoint images with shape [V, N, 3, H, W] where:
                   - V: Number of viewpoints
                   - N: Number of particles
                   - 3: RGB channels
                   - H, W: Image height and width
                   - Values should be in range [0, 1]
            view_type: Aggregation method across viewpoints. Options:
                      - 'mean': Average similarity across viewpoints (default)
                      - 'max': Maximum similarity across viewpoints
                      - 'min': Minimum similarity across viewpoints
        
        Returns:
            Tuple[float, float]: (fidelity_mean, fidelity_std) where:
                                - fidelity_mean: Average fidelity across particles
                                - fidelity_std: Standard deviation of fidelity across particles
                                - Range: [-1, 1] where higher values indicate better fidelity
                                - 1: Perfect text-image alignment
                                - 0: No correlation between text and image
                                - -1: Anti-correlation between text and image
        
        Note:
            - Uses CLIP features for fidelity computation (trained for text-image alignment)
            - Features are automatically L2-normalized
            - Text features are cached for efficiency
            - Similarity computation uses dot product of normalized features (equivalent to cosine)
        
        Example:
            >>> calculator = MetricsCalculator(opt, prompt="a photo of a cat")
            >>> images = torch.randn(8, 4, 3, 256, 256)  # 8 views, 4 particles
            >>> mean_fid, std_fid = calculator.compute_clip_fidelity_in_multi_viewpoints_stats(images)
            >>> print(f"Fidelity: {mean_fid:.3f} ± {std_fid:.3f}")
            >>> 
            >>> # Use maximum fidelity across viewpoints
            >>> max_fid, max_std = calculator.compute_clip_fidelity_in_multi_viewpoints_stats(
            ...     images, view_type="max")
            >>> print(f"Max fidelity: {max_fid:.3f} ± {max_std:.3f}")
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must be [V,N,3,H,W]")
        V, N, C, H, W = images.shape

        # Flatten views for batch CLIP processing
        images_flat = images.view(V * N, C, H, W)

        # Preprocess images and encode with evaluation-time CLIP
        images_clip = self._preprocess_images(images_flat, backbone="clip")  # [V*N, 3, R, R]
        image_features = self.metrics_model.clip_encode_images(images_clip)
        text_features = self.get_clip_text_features().expand(V * N, -1)

        # Cosine similarity (since both are normalized, dot product = cosine)
        sims = (image_features * text_features).sum(dim=1).view(V, N)  # [V, N]

        if view_type == "mean":
            sim_vals = sims.mean(dim=0)  # [N]
        elif view_type == "max":
            sim_vals = sims.max(dim=0).values
        elif view_type == "min":
            sim_vals = sims.min(dim=0).values
        else:
            raise ValueError(f"Invalid view_type: {view_type}")

        fidelity_mean = float(sim_vals.mean().item())
        fidelity_std = float(sim_vals.std(unbiased=False).item()) if sim_vals.numel() > 1 else 0.0
        return fidelity_mean, fidelity_std

    @torch.no_grad()
    def compute_lpips_inter_and_consistency(
        self, multi_view_images: Tensor
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute LPIPS-based inter-sample diversity and cross-view consistency.
        
        This method provides perceptual similarity metrics using the Learned Perceptual
        Image Patch Similarity (LPIPS) network. LPIPS is particularly useful for measuring
        perceptual differences that align with human visual perception.
        
        Two metrics are computed:
        1. Inter-sample diversity: How different particles are from each other within the same view
        2. Cross-view consistency: How similar the same particle appears across different views
        
        Methodology:
        - Inter-sample: Compute pairwise LPIPS between all particles within each view, then average
        - Consistency: Compute LPIPS between different views of the same particle, then average
        
        Args:
            multi_view_images: Multi-viewpoint images with shape [V, N, 3, H, W] where:
                              - V: Number of viewpoints
                              - N: Number of particles
                              - 3: RGB channels
                              - H, W: Image height and width
                              - Values should be in range [0, 1]
        
        Returns:
            Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
                (lpips_inter_mean, lpips_inter_std, lpips_consistency_mean, lpips_consistency_std)
                
                Where:
                - lpips_inter_mean: Average inter-sample diversity across views
                - lpips_inter_std: Standard deviation of inter-sample diversity across views
                - lpips_consistency_mean: Average cross-view consistency across particles
                - lpips_consistency_std: Standard deviation of cross-view consistency across particles
                
                All values are None if LPIPS model is not available or insufficient data
        
        Note:
            - LPIPS expects inputs in range [-1, 1], so images are automatically converted
            - Higher LPIPS values indicate greater perceptual differences
            - Returns None values if LPIPS model is not loaded or insufficient data
            - Inter-sample diversity requires N >= 2 particles
            - Cross-view consistency requires V >= 2 viewpoints
        
        Example:
            >>> calculator = MetricsCalculator(opt, prompt="a cat")
            >>> images = torch.randn(8, 4, 3, 256, 256)  # 8 views, 4 particles
            >>> inter_mean, inter_std, cons_mean, cons_std = calculator.compute_lpips_inter_and_consistency(images)
            >>> if inter_mean is not None:
            ...     print(f"Inter-sample diversity: {inter_mean:.3f} ± {inter_std:.3f}")
            ...     print(f"Cross-view consistency: {cons_mean:.3f} ± {cons_std:.3f}")
        """
        if self.metrics_model.lpips is None:
            return None, None, None, None

        if multi_view_images.ndim != 5 or multi_view_images.shape[2] != 3:
            raise ValueError("multi_view_images must be [V,N,3,H,W]")

        V, N, C, H, W = multi_view_images.shape
        device = multi_view_images.device

        # LPIPS expects inputs in [-1, 1]
        images_lpips = (multi_view_images * 2) - 1  # [V, N, 3, H, W]

        # Inter-sample LPIPS across particles within the same view
        if N >= 2:
            i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=device)  # [K]
            inter_x1 = images_lpips[:, i_idx, ...].reshape(-1, C, H, W)  # [V*K, 3, H, W]
            inter_x2 = images_lpips[:, j_idx, ...].reshape(-1, C, H, W)  # [V*K, 3, H, W]
            inter_d = self.metrics_model.lpips(inter_x1, inter_x2).view(-1)  # [V*K]
            lpips_inter_mean = float(inter_d.mean().item())
            lpips_inter_std = float(inter_d.std(unbiased=False).item()) if inter_d.numel() > 1 else 0.0
        else:
            lpips_inter_mean, lpips_inter_std = None, None

        # Cross-view LPIPS consistency for the same particle
        if V >= 2:
            v_i, v_j = torch.triu_indices(V, V, offset=1, device=device)  # [P]
            cons_x1 = images_lpips[v_i, ...].reshape(-1, C, H, W)
            cons_x2 = images_lpips[v_j, ...].reshape(-1, C, H, W)
            cons_d = self.metrics_model.lpips(cons_x1, cons_x2).view(-1)  # [P*N]
            lpips_consistency_mean = float(cons_d.mean().item())
            lpips_consistency_std = float(cons_d.std(unbiased=False).item()) if cons_d.numel() > 1 else 0.0
        else:
            lpips_consistency_mean, lpips_consistency_std = None, None

        return lpips_inter_mean, lpips_inter_std, lpips_consistency_mean, lpips_consistency_std
    
    @torch.no_grad()
    def compute_kernel_statistics(
        self, 
        kernel: Tensor, 
        kernel_grad: Tensor, 
        features: Tensor
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Compute kernel statistics for monitoring kernel behavior during training.
        
        This method computes various statistics that help understand the kernel's behavior:
        1. Effective sample size (Neff): Measures how many effective independent samples we have
        2. Row-sum statistics: Measures the total influence of each particle
        3. Gradient norm: Measures the magnitude of the repulsion gradient
        
        Args:
            kernel: Kernel matrix with shape [N, N] where N is the number of particles
            kernel_grad: Kernel gradient with shape [N, D] where D is the feature dimension
            features: Feature vectors with shape [N, D] where D is the feature dimension
        
        Returns:
            Tuple[float, float, float, float, float, float]: 
                (neff_mean, neff_std, row_sum_mean, row_sum_std, gradient_norm_mean, gradient_norm_std)
                
                Where:
                - neff_mean: Mean effective sample size across particles
                - neff_std: Standard deviation of effective sample size
                - row_sum_mean: Mean of kernel row sums
                - row_sum_std: Standard deviation of kernel row sums  
                - gradient_norm_mean: Mean gradient norm across particles
                - gradient_norm_std: Standard deviation of gradient norm
        
        Note:
            - Effective sample size is computed as Neff = 1 / (sum of squared kernel values)
            - Row sums indicate how much each particle influences others
            - Gradient norm measures the strength of repulsion forces
        
        Example:
            >>> calculator = MetricsCalculator(opt, prompt="a cat")
            >>> kernel = torch.randn(4, 4)
            >>> kernel_grad = torch.randn(4, 768)
            >>> features = torch.randn(4, 768)
            >>> neff_mean, neff_std, row_sum_mean, row_sum_std, grad_norm_mean, grad_norm_std = calculator.compute_kernel_statistics(kernel, kernel_grad, features)
            >>> print(f"Neff: {neff_mean:.3f} ± {neff_std:.3f}")
        """
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("kernel must be a square matrix [N, N]")
        if kernel_grad.ndim != 2 or kernel_grad.shape[0] != kernel.shape[0]:
            raise ValueError("kernel_grad must be [N, D] where N matches kernel")
        if features.ndim != 2 or features.shape[0] != kernel.shape[0]:
            raise ValueError("features must be [N, D] where N matches kernel")
            
        N = kernel.shape[0]
        if N == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        # 1. Effective sample size (Neff) per particle
        # Neff = 1 / (sum of squared kernel values for each row)
        kernel_squared = kernel ** 2
        neff_per_particle = 1.0 / (kernel_squared.sum(dim=1) + 1e-8)  # [N]
        neff_mean = float(neff_per_particle.mean().item())
        neff_std = float(neff_per_particle.std(unbiased=False).item()) if N > 1 else 0.0
        
        # 2. Row-sum statistics
        row_sums = kernel.sum(dim=1)  # [N]
        row_sum_mean = float(row_sums.mean().item())
        row_sum_std = float(row_sums.std(unbiased=False).item()) if N > 1 else 0.0
        
        # 3. Gradient norm statistics
        gradient_norms = kernel_grad.norm(dim=1)  # [N]
        gradient_norm_mean = float(gradient_norms.mean().item())
        gradient_norm_std = float(gradient_norms.std(unbiased=False).item()) if N > 1 else 0.0
        
        return neff_mean, neff_std, row_sum_mean, row_sum_std, gradient_norm_mean, gradient_norm_std

    @torch.no_grad()
    def log_kernel_stats(self, step: int, kernel_stats: dict):
        """
        Log kernel statistics to CSV file.
        
        Args:
            step: Current training step
            kernel_stats: Dictionary containing kernel statistics
        """
        if self.kernel_stats_writer is None:
            return
        self.kernel_stats_writer.writerow([
            step,
            kernel_stats.get("neff_mean", ""),
            kernel_stats.get("neff_std", ""),
            kernel_stats.get("row_sum_mean", ""),
            kernel_stats.get("row_sum_std", ""),
            kernel_stats.get("gradient_norm_mean", ""),
            kernel_stats.get("gradient_norm_std", ""),
        ])
    


__all__ = ["MetricsModel", "MetricsCalculator"]
