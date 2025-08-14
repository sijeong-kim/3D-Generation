"""metrics.py

This module provides two classes:

- MetricsModel: wraps evaluation-time models (OpenCLIP, optional DINO, optional LPIPS)
- MetricsCalculator: convenience utilities for computing and logging metrics over batches

Key features
------------
- Robust device selection (auto → CUDA/MPS/CPU) and optional AMP (fp16) on CUDA/MPS
- Graceful optional deps (torchvision, timm, lpips) with clear warnings
- Normalized feature embeddings and simple CLIP/DINO helpers
- CSV logging using Python's `csv` for safer output (buffered, newline-safe)
- Helpful error messages and thorough type hints

Usage sketch
------------
>>> mm = MetricsModel(device="auto", eval_clip_model_name="ViT-bigG-14")
>>> calc = MetricsCalculator({"metrics": True, "outdir": "./out"}, prompt="a photo of a hamburger")
>>> # compute metrics with calc.compute_* and log with calc.log_metrics(...)
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
    """Holds evaluation-time models for metrics:
    - OpenCLIP (default: ViT-bigG-14) for CLIPScore and feature embeddings
    - DINO ViT (optional, via timm/torch.hub) for diversity features
    - LPIPS network for perceptual distance (optional)
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
        # Device & precision
        self.device = _pick_device(device)
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
        """Apply CLIP eval preprocessing if available, else pass-through stacking.
        Accepts an iterable of PIL Images or CHW/HWC tensors; returns [B,3,H,W].
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
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.eval_clip_tokenizer(list(texts)).to(self.device)
        use_amp = _autocast_enabled(self.device) and self.precision == "fp16"
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=use_amp):
            text_features = self.eval_clip_model.encode_text(tokens)
        return F.normalize(text_features.float(), dim=1)

    @torch.no_grad()
    def clip_encode_images(self, images_clip: Tensor) -> Tensor:
        # images_clip: [B, 3, H, W], already normalized to CLIP stats
        if images_clip.ndim != 4 or images_clip.shape[1] != 3:
            raise ValueError("images_clip must be [B,3,H,W]")
        use_amp = _autocast_enabled(self.device) and self.precision == "fp16"
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=use_amp):
            image_features = self.eval_clip_model.encode_image(images_clip.to(self.device))
        return F.normalize(image_features.float(), dim=1)

    @torch.no_grad()
    def dino_encode_images(self, images_dino: Tensor) -> Tensor:
        if self.dino_model is None:
            raise RuntimeError("DINO model not available; install timm or enable hub load.")
        if images_dino.ndim != 4 or images_dino.shape[1] != 3:
            raise ValueError("images_dino must be [B,3,H,W]")

        use_amp = _autocast_enabled(self.device) and self.precision == "fp16"
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=use_amp):
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
        if self._cached_text == prompt and self._cached_text_feats is not None:
            return self._cached_text_feats
        feats = self.clip_encode_text(prompt)
        self._cached_text = prompt
        self._cached_text_feats = feats
        return feats

    def to(self, device: str) -> "MetricsModel":
        device = _pick_device(device)
        self.device = device
        self.eval_clip_model.to(device)
        if self.dino_model is not None:
            self.dino_model.to(device)
        if self.lpips is not None:
            self.lpips.to(device)
        return self


class MetricsCalculator:
    def __init__(
        self,
        opt: Any,
        prompt: str = "a photo of a hamburger",
        device: Optional[str] = None,
    ):
        """
        opt can be a dict or an argparse-like object with attributes.
        Required fields used here:
          - outdir (str)
          - metrics (bool)
          - enable_lpips (bool)
          - *_interval (ints): quantitative_metrics_interval, losses_interval, efficiency_interval
          - eval_clip_model_name (optional, str)
          - eval_clip_pretrained (optional, str)
          - init_train_clip (optional, bool)
        """
        if opt is None:
            raise ValueError("opt parameter cannot be None")
        self.opt = opt

        # Accessor for dict or namespace
        def _get(name, default=None):
            if isinstance(self.opt, dict):
                return self.opt.get(name, default)
            return getattr(self.opt, name, default)

        self._get = _get

        # Device selection
        self.device = _pick_device(device or "auto")

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

        # Open files in newline-safe mode; buffered writes
        self.metrics_csv_file = open(self.metrics_csv_path, "w", newline="", buffering=1)
        self.losses_csv_file = open(self.losses_csv_path, "w", newline="", buffering=1)
        self.efficiency_csv_file = open(self.efficiency_csv_path, "w", newline="", buffering=1)

        self.metrics_writer = csv.writer(self.metrics_csv_file)
        self.losses_writer = csv.writer(self.losses_csv_file)
        self.efficiency_writer = csv.writer(self.efficiency_csv_file)

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
        ])

        # Explicit units: time in milliseconds, memory in megabytes
        self.efficiency_writer.writerow(["step", "time_ms", "memory_allocated_MB", "max_memory_allocated_MB"])  # noqa: N806

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
        for f in (self.metrics_csv_file, self.losses_csv_file, self.efficiency_csv_file):
            try:
                f and f.close()
            except Exception:
                pass

    def __del__(self):  # pragma: no cover
        self.close()

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
        Inter-particle diversity across views.

        Per view, diversity = 1 - mean off-diagonal cosine similarity between particles.
        Returns mean and std (across views) of that per-view diversity.

        Args:
            multi_view_images: [V, N, 3, H, W]
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
        Cross-view consistency across viewpoints.

        Args:
            multi_view_images: [V, N, 3, H, W]
            return_per_particle: True면 (mean, std, per_particle_consistency) 반환

        Returns:
            (mean_consistency, std_consistency) 또는
            (mean_consistency, std_consistency, per_particle_consistency)
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
        Compute text-image CLIP fidelity across multiple viewpoints and aggregate.

        Args:
            images: [V, N, 3, H, W] in [0, 1]
            view_type: 'mean' | 'max' | 'min' aggregation across views.

        Returns:
            (fidelity_mean, fidelity_std)
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
        LPIPS-based inter-sample diversity and cross-view consistency.

        - Inter-sample: mean/std of pairwise LPIPS across particles within the same view, averaged over views.
        - Consistency: mean/std of LPIPS between views of the same particle, averaged over particles.

        Args:
            multi_view_images: [V, N, 3, H, W] in [0,1]

        Returns:
            (lpips_inter_mean, lpips_inter_std, lpips_consistency_mean, lpips_consistency_std)
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


__all__ = ["MetricsModel", "MetricsCalculator"]
