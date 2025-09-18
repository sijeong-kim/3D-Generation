# Diversify Guided 3D Generation via Repulsive 3D Gaussian Splatting: Comprehensive Ablation and Pareto Selection

## Executive Summary

This report presents an MSc-level ablation for “Diversify Guided 3D Generation via Repulsive 3D Gaussian Splatting.” We ablate repulsion method, kernel type, λ (repulsion), guidance scale, and RBF β. The goal is to enhance diversity while maintaining fidelity and stability. We adopt a weighted Pareto selector to recommend a single balanced configuration tailored to the dissertation’s diversity focus.

## Key Findings

- Repulsion Method: RLSD (Repulsive Latent Score Distillation) outperforms SVGD in fidelity and diversity after prompt- and kernel-averaging.
- Kernel Type: RBF substantially improves diversity vs COS with comparable fidelity.
- λ (Repulsion): Non-monotonic; a mid-range band (≈600–1200) balances fidelity and diversity.
- Guidance Scale: Moderate values (50–70) balance text adherence and geometry.
- RBF β: Lower-to-mid values (≈0.5–1.0) favor diversity; higher values lean toward fidelity.

### Recommended (Pareto-weighted, diversity-focused)

Selected with weighted utopia-distance (w_div=0.60, w_fid=0.40) and a relaxed ε-constraint on consistency (ε=0.02):

- Repulsion Method: RLSD
- Kernel Type: RBF
- λ (Repulsion): 1000.0
- Guidance Scale: 50
- RBF β: 0.5

See `results/ablation/plots/selection_summary.txt` for the reproducible selection snapshot.

---

## 1. Repulsion Method (RLSD vs SVGD)

Methodology: Average over prompts; for method comparison, average over kernels; statistics across seeds. RLSD = Repulsive Latent Score Distillation.

Results (means ± std):

- RLSD: Fidelity ≈ 0.390 ± 0.003; Diversity ≈ 0.232 ± 0.037; Consistency ≈ 0.838 ± 0.008
- SVGD: Fidelity ≈ 0.374 ± 0.006; Diversity ≈ 0.218 ± 0.039; Consistency ≈ 0.840 ± 0.006

Interpretation: RLSD improves fidelity (~+4.3%) and diversity (~+6–7%) vs SVGD while retaining high cross-view consistency. Latent-space repulsion spreads modes and mitigates collapse.

Figures:
- Repulsion Methods (Pareto): `results/ablation/plots/repulsion_methods_pareto.png`
- Repulsion Methods (Bars): `results/ablation/plots/repulsion_methods_bars.png`

## 2. Kernel Type (COS vs RBF)

Methodology: Average over prompts; for kernel comparison, average over methods; statistics across seeds.

Results (means ± std):

- COS: Fidelity ≈ 0.381 ± 0.013; Diversity ≈ 0.189 ± 0.010; Consistency ≈ 0.846 ± 0.004
- RBF: Fidelity ≈ 0.384 ± 0.004; Diversity ≈ 0.261 ± 0.007; Consistency ≈ 0.833 ± 0.003

Interpretation: RBF raises diversity (~+37.6%) with similar fidelity. Distance-based similarity better supports repulsion in 3D.

Figure: `results/ablation/plots/kernel_types_bars.png`

## 3. λ (Repulsion Strength)

Methodology: Coarse and fine sweeps; averaged over prompts; statistics across seeds.

Key points:
- Non-monotonic fidelity; diversity rises with larger λ but may reduce fidelity beyond ≈1000–1200.
- Stable operating region around 600–1200.

Figures:
- λ Pareto: `results/ablation/plots/lambda_repulsion_pareto.png`
- λ Lines: `results/ablation/plots/lambda_repulsion_lines.png`

## 4. Guidance Scale

Methodology: Values {30, 50, 70, 100}; averaged over prompts.

Observations:
- 50–70 offers a good balance; 30 can over-diversify; 100 may reduce diversity.

Figures:
- Guidance Pareto: `results/ablation/plots/guidance_scale_pareto.png`
- Guidance Lines: `results/ablation/plots/guidance_scale_lines.png`

## 5. RBF β

Methodology: Values {0.5, 1.0, 1.5, 2.0}; averaged over prompts.

Observations:
- β≈0.5–1.0 supports higher diversity; β≈2.0 leans to fidelity.

Figures:
- β Pareto: `results/ablation/plots/rbf_beta_pareto.png`
- β Lines: `results/ablation/plots/rbf_beta_lines.png`

---

## 6. Pareto Selection: Method and Rationale

Objectives: Maximize fidelity and diversity while maintaining cross-view consistency.

Selector:
- Normalize (fidelity, diversity) to [0,1].
- Weighted Euclidean distance to Utopia (1,1): weights (w_fid, w_div) with w_div≥w_fid for diversity-focused selection (default 0.4, 0.6).
- ε-constraint on consistency: consistency ≥ max(consistency) − ε, default ε=0.02.
- Choose the point with minimum weighted distance under the constraint.

Why this works:
- Encodes topic priority (diversity) while guarding against instability.
- Standard in multi-objective decision making (utopia/knee-point, NSGA-II).

Selected (w_div=0.60, w_fid=0.40, ε=0.02): RLSD, RBF, λ=1000.0, guidance=50, β=0.5

All selected points are annotated on plots with numeric values and dashed guides for clarity.

---

## 7. Practical Recommendations

- Default (diversity-focused): RLSD + RBF, λ≈1000, guidance≈50, β≈0.5.
- Stability-upweighting: increase w_fid or tighten ε to favor slightly higher consistency/fidelity.
- Task-tuned: adjust weights/ε per application; re-run selector to get a new single-point.

## 8. Reproducibility

Command (adjust weights/ε as needed):

```bash
WEIGHT_FID=0.4 WEIGHT_DIV=0.6 EPSILON_CONS=0.02 python3 analysis/ablation_study_analysis.py
```

Outputs:
- Figures (PNG + PDF, thesis-ready): `results/ablation/plots/`
- Selection snapshot: `results/ablation/plots/selection_summary.txt`

---

## References

- Kerbl, B., et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering.
- Liu, L., et al. (2023). Repulsion Loss for 3D Gaussian Splatting.
- Liu, Q., & Wang, D. (2016). Stein Variational Gradient Descent.
- Deb, K., et al. (2002). NSGA-II for multi-objective optimization.

*Prepared for MSc thesis-quality presentation with colorblind palette, readable annotations, and vector exports.*
