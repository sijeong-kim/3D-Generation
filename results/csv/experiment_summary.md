# 3D Generation Experiment Suite

## Overview

Total experiments: 12

## Experiments

### exp4_guidance_scale
**Description:** Guidance scale ablation: [30, 50, 70, 100]

**Key Parameters:**
- guidance_scale_values: [30, 50, 70, 100]
- lambda_repulsion: 1000
- repulsion_type: rlsd
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42, 123]

### exp5_rbf_beta
**Description:** RBF kernel beta parameter ablation: [0.5, 1.0, 1.5, 2.0]

**Key Parameters:**
- rbf_beta_values: [0.5, 1.0, 1.5, 2.0]
- lambda_repulsion: 1000
- repulsion_type: rlsd
- kernel_type: rbf
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42, 123]

### exp3_lambda_fine
**Description:** Fine lambda repulsion ablation: [600, 800, 1K, 1.2K, 1.4K]

**Key Parameters:**
- lambda_repulsion_values: [600, 800, 1000, 1200, 1400]
- repulsion_type: rlsd
- kernel_type: rbf
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42, 123]

### exp_num_particles
**Description:** Number of particles ablation: [2, 4, 8]

**Key Parameters:**
- num_particles_values: [2, 4, 8]
- lambda_repulsion: 1000
- repulsion_type: rlsd
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42]

### exp0_baseline_indep
**Description:** Experiment: exp0_baseline_indep

**Key Parameters:**

### exp_num_pts
**Description:** Number of points ablation: [1K, 3K, 5K]

**Key Parameters:**
- num_pts_values: [1000, 3000, 5000]
- lambda_repulsion: 1000
- repulsion_type: rlsd
- targets: ['ICE', 'CACT']
- seeds: [42]

### exp_gaussian_reproduce
**Description:** Experiment: exp_gaussian_reproduce

**Key Parameters:**

### exp2_lambda_coarse
**Description:** Coarse lambda repulsion ablation: [1, 10, 100, 1K, 10K]

**Key Parameters:**
- lambda_repulsion_values: [1, 10, 100, 1000, 10000]
- repulsion_type: rlsd
- kernel_type: rbf
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42]

### exp_feature_layer
**Description:** Feature layer ablation: [early, mid, last]

**Key Parameters:**
- feature_layer_values: ['early', 'mid', 'last']
- feature_extractor: facebook/dinov2-base
- lambda_repulsion: 1000
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42]

### exp0_baseline
**Description:** Baseline experiment with no repulsion (WO - Without repulsion)

**Key Parameters:**
- repulsion_type: wo
- lambda_repulsion: 1000
- kernel_type: none
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42, 123, 456, 789]

### exp_opacity_lr
**Description:** Opacity learning rate ablation: [0.005, 0.01, 0.05]

**Key Parameters:**
- opacity_lr_values: [0.005, 0.01, 0.05]
- lambda_repulsion: 1000
- repulsion_type: rlsd
- targets: ['ICE', 'CACT']
- seeds: [42]

### exp1_repulsion_kernel
**Description:** Repulsion kernel ablation: RLSD vs SVGD with RBF vs COS kernels

**Key Parameters:**
- methods: ['RLSD', 'SVGD']
- kernels: ['RBF', 'COS']
- lambda_repulsion: 1000
- targets: ['BULL', 'ICE', 'CACT', 'TUL']
- seeds: [42, 123]

## Metrics

- **Fidelity**: Image fidelity metric (higher is better)
- **Diversity**: Inter-particle diversity metric (higher is better)
- **Cross-consistency**: Cross-view consistency metric (higher is better)

## Prompts

- **"a bulldozer made out of toy bricks"**: Bulldozer made out of toy bricks
- **"a photo of an ice cream"**: Photo of an ice cream
- **"a small saguaro cactus planted in a clay pot"**: Small saguaro cactus planted in a clay pot
- **"a tulip flower in a vase"**: Tulip flower in a vase
