# kernels.py
"""
Kernel Functions for Stein Variational Gradient Descent (SVGD) and Related Methods

This module provides comprehensive kernel implementations for particle-based optimization
methods, specifically designed for Stein Variational Gradient Descent (SVGD) and its
variants including Regularized Least-Squares Descent (RLSD).

Key Features:
- RBF (Radial Basis Function) kernel with automatic bandwidth selection
- Cosine similarity kernel with optional scaling
- Support for both SVGD and RLSD repulsion types
- Automatic gradient computation with numerical stability
- Comprehensive testing and validation utilities
- Median heuristic bandwidth selection for RBF kernels

Supported Kernels:
- RBF Kernel: k(x,y) = exp(-||x-y||²/h) with bandwidth h
- Cosine Kernel: k(x,y) = cos(x,y) = (x·y)/(||x||·||y||)

Supported Repulsion Types:
- SVGD: Standard Stein Variational Gradient Descent
- RLSD: Regularized Least-Squares Descent (smooth SVGD)

Mathematical Background:
The kernels are used to compute repulsion gradients that push particles apart:
- SVGD: ∇ᵢ∑ⱼ Kᵢⱼ (gradient of kernel sum)
- RLSD: ∇ᵢlog(∑ⱼ Kᵢⱼ) (gradient of log kernel sum)

Author: [Your Name]
Date: [Date]
"""

import torch
import torch.nn.functional as F


def rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd", eps=1e-12):
    r"""
    Compute the RBF kernel matrix and its gradients for particle-based optimization.
    
    This function implements the Radial Basis Function (RBF) kernel with automatic
    bandwidth selection and gradient computation for Stein Variational Gradient Descent
    (SVGD) and its variants.
    
    Mathematical Formulation:
    Given feature vectors :math:`\{z_i\}_{i=1}^N`, with :math:`z_i \in \mathbb{R}^D`,
    define squared distances :math:`d_{ij}^2 = \|z_i - z_j\|_2^2`.

    Bandwidth Selection:
    .. math::
        h =
        \begin{cases}
            \tau^2, & \tau > 0, \\
            \dfrac{\mathrm{median}\big(\{ \|z_i - z_j\| \}_{i \neq j}\big)^2}{\log N + \epsilon}, & \tau = 0.
        \end{cases}

    RBF Kernel:
    .. math::
        K_{ij} = \exp\!\left(-\frac{d_{ij}^2}{h}\right).

    Gradient Computation:

    - SVGD (Stein Variational Gradient Descent):
      .. math::
          \nabla_{z_i} \sum_j K_{ij}
          \;=\; \frac{2}{h} \sum_j K_{ij}\,(z_j - z_i).

    - RLSD (Regularized Least-Squares Descent):
      .. math::
          \nabla_{z_i}\,\log \Big(\sum_j K_{ij}\Big)
          \;=\; \frac{\nabla_{z_i} \sum_j K_{ij}}{\sum_j K_{ij}}.

    Args:
        features (torch.Tensor): Feature tensor of shape [N, D] where:
                                - N: Number of particles
                                - D: Feature dimension
        tau (float): Bandwidth parameter. If tau > 0, uses fixed bandwidth tau².
                    If tau = 0, uses median heuristic: h = median²(distances) / log(N).
                    Default: 0.5
        repulsion_type (str): Type of repulsion gradient to compute. Options:
                             - "svgd": Standard Stein Variational Gradient Descent
                             - "rlsd": Regularized Least-Squares Descent (smooth SVGD)
                             Default: "svgd"
        eps (float): Numerical stability constant for logs and divisions.
                    Used to prevent division by zero and log(0). Default: 1e-12

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (K, grad) where:
            - K: Kernel matrix of shape [N, N] with entries K[i, j] = k(z_i, z_j)
            - grad: Gradient tensor of shape [N, D] containing gradients for each particle

    Note:
        - The median heuristic (tau=0) automatically adapts bandwidth to data scale
        - All gradients are computed analytically for efficiency
        - Numerical stability is maintained through eps parameter
        - The kernel matrix K is symmetric and positive semi-definite

    Example:
        >>> features = torch.randn(10, 32)  # 10 particles, 32-dimensional features
        >>> K, grad = rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd")
        >>> print(f"Kernel shape: {K.shape}")  # torch.Size([10, 10])
        >>> print(f"Gradient shape: {grad.shape}")  # torch.Size([10, 32])
        
        >>> # Use median heuristic for automatic bandwidth selection
        >>> K, grad = rbf_kernel_and_grad(features, tau=0.0, repulsion_type="rlsd")
    """
    
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    
    N, D_feature = features.shape
    
    # Compute pairwise differences and squared distances
    z_i = features.unsqueeze(1)  # [N, 1, D_feature]
    z_j = features.unsqueeze(0)  # [1, N, D_feature]
    
    diffs = z_i - z_j  # [N, N, D_feature] where diffs[i,j] = z_i - z_j
    sq_dists = (diffs ** 2).sum(-1)  # [N, N] # sq_dists[i,j] = ||z_i - z_j||^2
    
    # Bandwidth selection: fixed tau or median heuristic
    if tau == 0:  # Use median bandwidth heuristic
        # Median bandwidth heuristic: h = median²(distances) / log(N)
        # This automatically adapts bandwidth to the scale of the data
        dists = sq_dists.sqrt()  # [N, N] # pairwise distances between particles
        off_diag = ~torch.eye(N, dtype=torch.bool, device=features.device)  # [N, N]
        valid_dists = dists[off_diag]  # [N*(N-1)] # exclude diagonal elements
        median_dist = torch.median(valid_dists) if valid_dists.numel() > 0 else dists.median()  # [1]
        logN = torch.log(torch.tensor(float(N), dtype=torch.float32, device=features.device))
        h = median_dist ** 2 / (logN + eps)  # [1] # h = median(dist)² / log(N)
        h = h.clamp_min(eps)  # Ensure positive bandwidth
    else:
        # Use fixed bandwidth parameter
        h = torch.tensor(tau ** 2, dtype=torch.float32, device=features.device).clamp_min(eps)
    
    # RBF kernel: k(z_i, z_j) = exp(-||z_i - z_j||² / h)
    K = torch.exp(-sq_dists / h)  # [N, N]
    
    # Compute gradient with respect to z_i:
    # ∇_{z_i} sum_j K_{ij} = (2/h) * sum_j K_{ij} * (z_j - z_i)
    K_weight = K.unsqueeze(-1)  # [N, N, 1]
    grad_svgd = (2.0 / h) * (K_weight * (-diffs)).sum(dim=1)  # Use (z_j - z_i) = -diffs -> [N, D]
    
    # Choose repulsion type and compute final gradient
    if repulsion_type == "svgd":
        # Standard SVGD: gradient of kernel sum
        grad = grad_svgd
    elif repulsion_type == "rlsd":
        # RLSD: gradient of log kernel sum (smooth SVGD)
        denom = K.sum(dim=1, keepdim=True) + eps  # [N, 1]
        grad = grad_svgd / denom
    else:
        raise ValueError("Invalid repulsion type. Use 'svgd' or 'rlsd'.")
    
    return K, grad.detach()  # [N, N], [N, D_feature]


def cosine_kernel_and_grad(
    features: torch.Tensor,
    repulsion_type: str = "svgd",
    *,
    eps: float = 1e-12,
    eps_shift: float = 1e-3,  # minimal positive shift for RLSD stability
):
    r"""
    Shifted-cosine kernel (positive + PSD) and its gradients for SVGD/RLSD.

    K = cos(z_i, z_j) ∈ [-1, 1]
    K01 = 0.5 * (K + 1) ∈ [0, 1]
    Kpos = (1 - eps_shift) * K01 + eps_shift ∈ [eps_shift, 1]
         = c0 * K + c1,  where c0 = 0.5*(1-eps_shift), c1 = 0.5*(1-eps_shift) + eps_shift.

    dKpos/dK = c = 0.5*(1 - eps_shift)

    SVGD:  ∇_{z_i} ∑_j Kpos_ij = c * ∇_{z_i} ∑_j K_ij
    RLSD:  ∇_{z_i} log ∑_j Kpos_ij = [c * ∇_{z_i} ∑_j K_ij] / [∑_j Kpos_ij]
    """
    assert features.dim() == 2, "`features` must be [N, D]"
    N, D = features.shape

    # norms and cosine similarity
    norms = features.norm(dim=1, keepdim=True).clamp_min(eps)        # [N,1]
    inv_norms = 1.0 / norms                                          # [N,1]
    dot = features @ features.t()                                    # [N,N]
    denom = norms @ norms.t()                                        # [N,N]
    K = dot / denom                                                  # cos ∈ [-1,1]

    # SVGD gradient for raw cosine (no shift yet)
    # term1_i = (1/n_i) * sum_j (z_j / n_j)
    sum_z_over_n = (features * inv_norms).sum(dim=0, keepdim=True)   # [1,D]
    term1 = inv_norms * sum_z_over_n                                 # [N,D]
    # term2_i = (∑_j K_ij) * (z_i / n_i^2)
    K_sum_j = K.sum(dim=1, keepdim=True)                             # [N,1]
    term2 = K_sum_j * (features / (norms * norms))                   # [N,D]
    grad_svgd_raw = term1 - term2                                    # [N,D]

    # Build positive + PSD kernel (shifted cosine)
    c = 0.5 * (1.0 - eps_shift)                                      # dKpos/dK
    Kpos = c * (K + 1.0) + eps_shift                                 # = (1-εs)*0.5*(K+1)+εs

    if repulsion_type.lower() == "svgd":
        # ∇ sum_j Kpos = c * ∇ sum_j K
        grad = c * grad_svgd_raw
        return Kpos, grad.detach()

    elif repulsion_type.lower() == "rlsd":
        # ∇ log sum_j Kpos = [c * ∇ sum_j K] / [sum_j Kpos]
        denom_pos = Kpos.sum(dim=1, keepdim=True) + eps               # [N,1], no +N constant
        grad = (c * grad_svgd_raw) / denom_pos
        return Kpos, grad.detach()

    else:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'")

def exp_cosine_kernel_and_grad(
    features: torch.Tensor,
    beta: float = 6.0, # TODO: change temperature(beta): 4, 6, 8, 10
    repulsion_type: str = "svgd",
    *,
    eps: float = 1e-12,
):
    r"""
    Exponential-cosine kernel (positive + PSD) and its gradients for SVGD/RLSD.

    Cosine:     C_ij = (z_i^T z_j) / (||z_i|| ||z_j||) ∈ [-1, 1]
    Kernel:     K_ij = exp(beta * C_ij)               ∈ (0, +inf)
                (양수이므로 RLSD에 바로 사용 가능, beta≥0에서 PSD 유지)

    dC_ij/dz_i  = (1/||z_i||)*(z_j/||z_j||) - C_ij * (z_i/||z_i||^2)

    SVGD:  ∇_{z_i} ∑_j K_ij
         = ∑_j [ beta * K_ij * dC_ij/dz_i ]
         = beta * { (1/||z_i||) * ∑_j [K_ij * (z_j/||z_j||)]
                    - [∑_j K_ij * C_ij] * (z_i/||z_i||^2) }

    RLSD: ∇_{z_i} log ∑_j K_ij
         = (∇_{z_i} ∑_j K_ij) / (∑_j K_ij).
    """
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    N, D = features.shape

    # norms and cosine similarity
    norms = features.norm(dim=1, keepdim=True).clamp_min(eps)   # [N,1]
    inv_norms = 1.0 / norms                                     # [N,1]
    dot = features @ features.t()                                # [N,N]
    denom = norms @ norms.t()                                    # [N,N]
    C = dot / denom                                             # cosine ∈ [-1,1]

    # Exponential-cosine kernel (positive, PSD for beta >= 0)
    K = torch.exp(beta * C)                                     # [N,N]

    # Precompute helpful terms
    Z_over_n = features * inv_norms                             # [N,D]   (z_i / ||z_i||)
    # ∑_j K_ij * (z_j/||z_j||)  -> [N,D]
    sum_w_z_over_n = K @ Z_over_n
    # ∑_j K_ij * C_ij           -> [N,1]
    KC_sum = (K * C).sum(dim=1, keepdim=True)

    # Weighted cosine gradient per the chain rule
    term1_w = inv_norms * sum_w_z_over_n                        # [N,D]
    term2_w = KC_sum * (features / (norms * norms))             # [N,D]
    grad_svgd_weighted = beta * (term1_w - term2_w)             # [N,D]

    rep = repulsion_type.lower()
    if rep == "svgd":
        grad = grad_svgd_weighted
        return K, grad.detach()

    elif rep == "rlsd":
        denom_pos = K.sum(dim=1, keepdim=True).clamp_min(eps)   # [N,1]
        grad = grad_svgd_weighted / denom_pos
        return K, grad.detach()

    else:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'.")

