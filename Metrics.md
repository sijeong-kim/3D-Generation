## Metrics: Definitions, Rationale, and Citations

This document describes the quantitative and efficiency metrics implemented in `metrics.py`, including how they are computed, why we use them, and canonical references.

### Quantitative Metrics (Feature-Based)

All feature-based metrics are computed on multi-view images produced during evaluation. The module uses strong vision backbones to obtain semantically meaningful embeddings:

- CLIP for text–image fidelity (OpenCLIP implementation)
- DINO for diversity and view-consistency in the image space
- Optional LPIPS for perceptual distances

#### Inter-Particle Diversity (↑)
Measures how different particles are from one another within the same view, aggregated over views.

Pipeline:
1. Extract DINO features for each view `v` and particle `i`: \(f_{v,i} \in \mathbb{R}^D\) (L2-normalized).
2. For each view, compute cosine similarity matrix \(S_v\) across particles.
3. Off-diagonal mean similarity: \(\bar{s}_v = \frac{1}{N(N-1)} \sum_{i\neq j} S_v[i,j]\).
4. Diversity per view: \(d_v = 1 - \bar{s}_v\).
5. Report mean and standard deviation across views: \(\text{mean}(d_v), \text{std}(d_v)\).

Why: Penalizing similarity encourages variety in generated samples; DINO embeddings provide view-robust semantic features that correlate with visual diversity.

References: [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)

#### Cross-View Consistency (↑)
Measures how consistent the same particle appears across different camera viewpoints.

Pipeline:
1. Extract DINO features \(f_{v,i}\) as above and L2-normalize.
2. For each particle \(i\), compute cosine similarity across views: \(C_i = \frac{1}{V(V-1)} \sum_{v\neq w} \langle f_{v,i}, f_{w,i} \rangle\).
3. Report mean and standard deviation across particles: \(\text{mean}(C_i), \text{std}(C_i)\).

Why: Higher consistency indicates stronger 3D coherence and stable appearance across viewpoints.

References: [DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)

#### CLIP Fidelity (↑)
Measures text–image alignment using CLIP.

Pipeline:
1. Compute CLIP image features for each rendered view and CLIP text features for the prompt; L2-normalize both.
2. Per-view cosine similarity with the text features; aggregate across views by mean (default) or other reducers.
3. Report mean and standard deviation across particles.

Why: CLIP similarity is a widely adopted proxy for text–image alignment in generative modeling.

References: [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020), [OpenCLIP](https://github.com/mlfoundations/open_clip)

#### Optional: LPIPS Diversity/Consistency (↑/↓)
LPIPS estimates perceptual distance in deep feature space. We optionally compute:
- Inter-sample LPIPS within the same view (diversity): higher = more varied.
- Cross-view LPIPS for the same particle (consistency): lower = more consistent.

References: [LPIPS (Zhang et al., 2018)](https://arxiv.org/abs/1801.03924)

### Efficiency Metrics

Logged per optimization step in `efficiency.csv` with explicit units:

- `step_wall_ms` (ms): Wall-clock time per training step.
- `step_gpu_ms` (ms): GPU time per training step (if available).
- Sub-component times (ms): `render_ms`, `sd_guidance_ms`, `backprop_ms`, `densify_ms`.
- Memory (MB): `memory_allocated_mb` (current), `max_memory_allocated_mb` (peak).
- Throughput: `px_per_s = pixels / (step_wall_ms/1000)` where `pixels = num_particles × H × W`.

Why: These metrics quantify computational efficiency and memory footprint, enabling fair cost/quality trade-off analysis and Pareto comparisons across methods.

### Logging, Aggregation, and Reproducibility

- CSV Files per run (under `outdir/metrics/`):
  - `quantitative_metrics.csv`: diversity, consistency, fidelity (+ optional LPIPS).
  - `efficiency.csv`: step timings and memory.
  - `losses.csv`: training losses and auxiliary signals.
- Intervals are configurable via options (e.g., `quantitative_metrics_interval`, `efficiency_interval`).
- Aggregation in analysis scripts:
  - Step selection: use step==1000 or the last step ≤ 1000 if exact step missing.
  - Per-run: compute means over steps where applicable.
  - Per-prompt: average per-run metrics grouped by prompt.
  - Overall: mean and standard deviation across prompts to summarize experiments.

### Implementation Notes

- All feature embeddings are L2-normalized; cosine similarity reduces to dot product.
- Mixed precision is enabled on CUDA/MPS by default to reduce runtime and memory.
- DINO is loaded via `timm` when available (fallback: Torch Hub); CLIP uses OpenCLIP.
- LPIPS is optional due to added memory cost.

### BibTeX (for convenience)

```bibtex
@article{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and others},
  journal={ICML},
  year={2021}
}

@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and others},
  journal={ICCV},
  year={2021}
}

@article{zhang2018unreasonable,
  title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author={Zhang, Richard and others},
  journal={CVPR},
  year={2018}
}
```


