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
    
    return K, grad  # [N, N], [N, D_feature]


def cosine_kernel_and_grad(
    features: torch.Tensor,
    repulsion_type: str = "svgd",
    *,
    scale01: bool = True,  # map [-1, 1] -> [0, 1] by default
    eps: float = 1e-12,
):
    r"""
    Compute cosine-similarity kernel and its gradients for particle-based optimization.
    
    This function implements the cosine similarity kernel with optional scaling and
    gradient computation for Stein Variational Gradient Descent (SVGD) and its variants.
    
    Mathematical Formulation:
    Let z_i ∈ ℝ^D, n_i = ||z_i||_2, and
        k_{ij} = cos(z_i, z_j) = (z_i^T z_j)/(n_i n_j).
    
    The kernel can be optionally shifted to [0, 1] range:
        K^{(eff)}_{ij} = 0.5 * (k_{ij} + 1) ∈ [0, 1].

    Gradient Computation:

    - SVGD (Stein Variational Gradient Descent):
        ∇_{z_i} ∑_j k_{ij} = (1/n_i) ∑_j (z_j / n_j) - (∑_j k_{ij}) * (z_i / n_i^2).

    - RLSD (Regularized Least-Squares Descent):
        ∇_{z_i} log ∑_j K^{(eff)}_{ij} = ( ∇_{z_i} ∑_j k_{ij} ) / ( ∑_j K^{(eff)}_{ij} ).
        
    Scaling Behavior:
    If `scale01=True`, use K01 = 0.5 * (K + 1) ∈ [0,1].
    - SVGD on K01: grad = 0.5 * grad_svgd(K)
    - RLSD on K01: grad = (0.5 * grad_svgd(K)) / (0.5 * sum_j K01[i,j] + 0.5*N)
                  = grad_svgd(K) / (sum_j K01[i,j] + N)

    Args:
        features (torch.Tensor): Feature tensor of shape [N, D] where:
                                - N: Number of particles
                                - D: Feature dimension
        repulsion_type (str): Type of repulsion gradient to compute. Options:
                             - "svgd": Standard Stein Variational Gradient Descent
                             - "rlsd": Regularized Least-Squares Descent (smooth SVGD)
                             Default: "svgd"
        scale01 (bool): Whether to scale kernel from [-1, 1] to [0, 1] range.
                       This ensures positive kernel values for RLSD.
                       Default: True
        eps (float): Numerical stability constant for norms and divisions.
                    Used to prevent division by zero. Default: 1e-12

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (K_eff, grad) where:
            - K_eff: Effective kernel matrix of shape [N, N]
                    - If scale01=False: K_eff[i, j] = cos(z_i, z_j) ∈ [-1, 1]
                    - If scale01=True: K_eff[i, j] = 0.5 * (cos(z_i, z_j) + 1) ∈ [0, 1]
            - grad: Gradient tensor of shape [N, D] containing gradients for each particle

    Note:
        - The cosine kernel is invariant to scaling of individual features
        - RLSD requires positive kernel values, hence scale01=True is recommended
        - All gradients are computed analytically for efficiency
        - Numerical stability is maintained through eps parameter

    Example:
        >>> features = torch.randn(10, 32)  # 10 particles, 32-dimensional features
        >>> K, grad = cosine_kernel_and_grad(features, repulsion_type="svgd", scale01=True)
        >>> print(f"Kernel range: [{K.min():.3f}, {K.max():.3f}]")  # [0.0, 1.0]
        >>> print(f"Gradient shape: {grad.shape}")  # torch.Size([10, 32])
        
        >>> # Use unscaled kernel (not recommended for RLSD)
        >>> K, grad = cosine_kernel_and_grad(features, repulsion_type="svgd", scale01=False)
        >>> print(f"Kernel range: [{K.min():.3f}, {K.max():.3f}]")  # [-1.0, 1.0]
    """
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    
    N, D = features.shape

    # Compute norms and safe inverses for numerical stability
    norms = features.norm(dim=1, keepdim=True).clamp_min(eps)  # [N, 1]
    inv_norms = 1.0 / norms  # [N, 1]

    # Compute cosine kernel: K = (Z Z^T) / (||z_i|| ||z_j||)
    dot = features @ features.t()  # [N, N] # dot products
    denom = norms @ norms.t()  # [N, N] # norm products
    K = dot / denom  # [N, N] # cosine similarities ∈ [-1, 1]

    # Compute SVGD gradient numerator: ∇_{z_i} ∑_j k_{ij}
    # term1_i = (1/n_i) * sum_j (z_j / n_j)
    sum_z_over_n = (features * inv_norms).sum(dim=0, keepdim=True)  # [1, D] = ∑_j z_j / n_j
    term1 = inv_norms * sum_z_over_n  # [N, 1] * [1, D] -> [N, D]

    # term2_i = (∑_j k_{ij}) * (z_i / n_i^2)
    K_sum_j = K.sum(dim=1, keepdim=True)  # [N, 1]
    term2 = K_sum_j * (features / (norms * norms))  # z_i / n_i^2 -> [N, D]

    grad_svgd_raw = term1 - term2  # [N, D]

    # Choose repulsion type and handle scaling
    if not scale01:
        # Use unscaled kernel (not recommended for RLSD)
        if repulsion_type == "svgd":
            grad = grad_svgd_raw
        elif repulsion_type == "rlsd":
            denom = K_sum_j + eps  # [N, 1]
            
            # Check for negative kernel sums (invalid for RLSD)
            if torch.any(denom <= 0):
                raise ValueError("For 'rlsd' without scaling, sum_j K[i, j] must be positive. "
                                 "Use scale01=True to shift to [0, 1].")
                
            grad = grad_svgd_raw / denom
            return K, grad
        else:
            raise ValueError("Invalid repulsion type. Use 'svgd' or 'rlsd'.")
    
    # scale01 = True: K01 = 0.5 * (K + 1) ∈ [0, 1]
    K01 = 0.5 * (K + 1)
    
    if repulsion_type == "svgd":
        # SVGD on scaled kernel: d/dz sum_j k_{ij} = 0.5 * d/dz sum_j K[i, j]
        grad = 0.5 * grad_svgd_raw
    elif repulsion_type == "rlsd":
        # RLSD on scaled kernel: log sum_j K[i, j] = log(0.5*sum_j K[i, j] + 0.5*N)
        denom = 0.5 * K_sum_j + (0.5 * N)  # [N, 1]
        grad = 0.5 * grad_svgd_raw / (denom + eps)        
    else:
        raise ValueError("Invalid repulsion type. Use 'svgd' or 'rlsd'.")
    return K01, grad


def rbf_kernel_and_autograd(features, tau=0.5, repulsion_type="svgd", eps=1e-12):
    r"""
    Autograd-based version of RBF kernel gradient computation for validation.
    
    This function provides an alternative implementation of RBF kernel gradients
    using PyTorch's automatic differentiation. It is primarily used for validation
    and testing of the analytical gradient computation in `rbf_kernel_and_grad`.
    
    The function computes the same mathematical quantities as `rbf_kernel_and_grad`
    but uses autograd instead of analytical derivatives, making it useful for:
    - Validating analytical gradient implementations
    - Debugging gradient computation issues
    - Educational purposes (understanding the mathematical formulation)
    
    Args:
        features (torch.Tensor): Feature tensor of shape [N, D] where:
                                - N: Number of particles
                                - D: Feature dimension
        tau (float): Bandwidth parameter. If tau > 0, uses fixed bandwidth tau².
                    If tau = 0, uses median heuristic. Default: 0.5
        repulsion_type (str): Type of repulsion gradient to compute. Options:
                             - "svgd": Standard Stein Variational Gradient Descent
                             - "rlsd": Regularized Least-Squares Descent (smooth SVGD)
                             Default: "svgd"
        eps (float): Numerical stability constant. Default: 1e-12

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (K, grad) where:
            - K: Kernel matrix of shape [N, N] with entries K[i, j] = k(z_i, z_j)
            - grad: Gradient tensor of shape [N, D] containing gradients for each particle

    Note:
        - This function is slower than the analytical version due to autograd overhead
        - Results should match `rbf_kernel_and_grad` within numerical precision
        - Used primarily for testing and validation purposes
        - All gradients are computed with `torch.no_grad()` for efficiency

    Example:
        >>> features = torch.randn(5, 3)
        >>> K_auto, grad_auto = rbf_kernel_and_autograd(features, tau=0.5, repulsion_type="svgd")
        >>> K_anal, grad_anal = rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd")
        >>> print(f"Kernels match: {torch.allclose(K_auto, K_anal)}")
        >>> print(f"Gradients match: {torch.allclose(grad_auto, grad_anal)}")
    """
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    N, D = features.shape
    rep = repulsion_type.lower()
    if rep not in {"svgd", "rlsd"}:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'.")

    # Create leaf tensor with gradient tracking enabled
    Z = features.detach().clone().requires_grad_(True)

    # Compute pairwise squared distances and kernel
    z_i = Z.unsqueeze(1)  # [N,1,D]
    z_j = Z.unsqueeze(0)  # [1,N,D]
    diffs = z_i - z_j  # [N,N,D]
    sq_dists = (diffs**2).sum(-1)  # [N,N]

    # Bandwidth selection (treat as constant w.r.t. Z for autograd)
    if tau == 0:
        with torch.no_grad():
            dists = sq_dists.sqrt()
            off = ~torch.eye(N, dtype=torch.bool, device=Z.device)
            valid = dists[off]
            median_dist = (torch.median(valid) if valid.numel() > 0 else dists.median())
            logN = torch.log(torch.tensor(float(N), dtype=torch.float32, device=Z.device))
            h_val = (median_dist**2) / (logN + eps)
            h_val = float(h_val.clamp_min(eps).item())
        h = torch.tensor(h_val, dtype=Z.dtype, device=Z.device)
    else:
        h = torch.tensor(tau**2, dtype=Z.dtype, device=Z.device).clamp_min(eps).detach()

    # Compute RBF kernel
    K = torch.exp(-sq_dists / h)  # [N,N]

    # Compute gradients using autograd
    grads = []
    for i in range(N):
        if rep == "svgd":
            Fi = K[i].sum()  # \sum_j K[i,j]
        else:  # rlsd
            Fi = torch.log(K[i].sum() + eps)  # log \sum_j K[i,j]

        g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
        grads.append(g_full[i])

    grad = torch.stack(grads, dim=0)  # [N,D]
    return K.detach(), grad.detach()


def cosine_kernel_and_autograd(
    features: torch.Tensor,
    repulsion_type: str = "svgd",
    *,
    scale01: bool = True,   # map [-1,1] -> [0,1]
    eps: float = 1e-12,
):
    r"""
    Autograd-based version of cosine kernel gradient computation for validation.
    
    This function provides an alternative implementation of cosine kernel gradients
    using PyTorch's automatic differentiation. It is primarily used for validation
    and testing of the analytical gradient computation in `cosine_kernel_and_grad`.
    
    The function computes the same mathematical quantities as `cosine_kernel_and_grad`
    but uses autograd instead of analytical derivatives, making it useful for:
    - Validating analytical gradient implementations
    - Debugging gradient computation issues
    - Educational purposes (understanding the mathematical formulation)
    
    Args:
        features (torch.Tensor): Feature tensor of shape [N, D] where:
                                - N: Number of particles
                                - D: Feature dimension
        repulsion_type (str): Type of repulsion gradient to compute. Options:
                             - "svgd": Standard Stein Variational Gradient Descent
                             - "rlsd": Regularized Least-Squares Descent (smooth SVGD)
                             Default: "svgd"
        scale01 (bool): Whether to scale kernel from [-1, 1] to [0, 1] range.
                       Default: True
        eps (float): Numerical stability constant. Default: 1e-12

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (K_out, grad) where:
            - K_out: Kernel matrix of shape [N, N]
                    - If scale01=False: K_out[i, j] = cos(z_i, z_j) ∈ [-1, 1]
                    - If scale01=True: K_out[i, j] = 0.5 * (cos(z_i, z_j) + 1) ∈ [0, 1]
            - grad: Gradient tensor of shape [N, D] containing gradients for each particle

    Note:
        - This function is slower than the analytical version due to autograd overhead
        - Results should match `cosine_kernel_and_grad` within numerical precision
        - Used primarily for testing and validation purposes
        - All gradients are computed with `torch.no_grad()` for efficiency

    Example:
        >>> features = torch.randn(5, 3)
        >>> K_auto, grad_auto = cosine_kernel_and_autograd(features, repulsion_type="svgd", scale01=True)
        >>> K_anal, grad_anal = cosine_kernel_and_grad(features, repulsion_type="svgd", scale01=True)
        >>> print(f"Kernels match: {torch.allclose(K_auto, K_anal)}")
        >>> print(f"Gradients match: {torch.allclose(grad_auto, grad_anal)}")
    """
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    N, D = features.shape
    rep = repulsion_type.lower()
    if rep not in {"svgd", "rlsd"}:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'.")

    # Create leaf tensor with gradient tracking enabled
    Z = features.detach().clone().requires_grad_(True)

    # Compute cosine kernel
    norms = Z.norm(dim=1, keepdim=True).clamp_min(eps)  # [N,1]
    K = (Z @ Z.t()) / (norms @ norms.t())  # [N,N], in [-1,1]

    if not scale01:
        # Use unscaled kernel directly
        if rep == "rlsd":
            with torch.no_grad():
                sums = K.sum(dim=1)
                if torch.any(sums <= 0):
                    raise ValueError("For 'rlsd' without scaling, sum_j K[i,j] must be positive. "
                                     "Use scale01=True to shift to [0,1].")

        grads = []
        for i in range(N):
            if rep == "svgd":
                Fi = K[i].sum()  # \sum_j K[i,j]
            else:
                Fi = torch.log(K[i].sum() + eps)  # log \sum_j K[i,j]
            g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
            grads.append(g_full[i])
        grad = torch.stack(grads, dim=0)
        return K.detach(), grad.detach()

    # scale01=True: K01 = 0.5*(K + 1) ∈ [0,1]
    K01 = 0.5 * (K + 1.0)

    grads = []
    for i in range(N):
        if rep == "svgd":
            Fi = K01[i].sum()  # \sum_j K01[i,j]
        else:
            Fi = torch.log(K01[i].sum() + eps)  # log \sum_j K01[i,j]
        g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
        grads.append(g_full[i])

    grad = torch.stack(grads, dim=0)
    return K01.detach(), grad.detach()


def compare_rbf(N=5, D=3, tau=0.7, repulsion_type="svgd"):
    """
    Compare analytical and autograd implementations of RBF kernel gradients.
    
    This function validates the analytical gradient computation in `rbf_kernel_and_grad`
    by comparing it with the autograd-based implementation in `rbf_kernel_and_autograd`.
    
    Args:
        N (int): Number of particles for testing. Default: 5
        D (int): Feature dimension for testing. Default: 3
        tau (float): Bandwidth parameter for RBF kernel. Default: 0.7
        repulsion_type (str): Type of repulsion gradient to test. Options:
                             - "svgd": Standard Stein Variational Gradient Descent
                             - "rlsd": Regularized Least-Squares Descent (smooth SVGD)
                             Default: "svgd"
    
    Note:
        - Uses fixed random seed for reproducible results
        - Reports whether kernels and gradients match within numerical precision
        - Shows maximum gradient difference if they don't match
        - Useful for validating analytical gradient implementations
    """
    print(f"\n=== RBF kernel test (repulsion_type={repulsion_type}) ===")
    torch.manual_seed(0)
    Z = torch.randn(N, D)

    # Compute gradients using both methods
    K_vec, G_vec = rbf_kernel_and_grad(Z, tau=tau, repulsion_type=repulsion_type)
    K_auto, G_auto = rbf_kernel_and_autograd(Z, tau=tau, repulsion_type=repulsion_type)

    # Compare results
    print("K close?   ", torch.allclose(K_vec, K_auto, atol=1e-6, rtol=1e-5))
    print("Grad close?", torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5))
    if not torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5):
        print("Max grad diff:", (G_vec - G_auto).abs().max().item())


def compare_cosine(N=5, D=3, repulsion_type="svgd"):
    """
    Compare analytical and autograd implementations of cosine kernel gradients.
    
    This function validates the analytical gradient computation in `cosine_kernel_and_grad`
    by comparing it with the autograd-based implementation in `cosine_kernel_and_autograd`.
    
    Args:
        N (int): Number of particles for testing. Default: 5
        D (int): Feature dimension for testing. Default: 3
        repulsion_type (str): Type of repulsion gradient to test. Options:
                             - "svgd": Standard Stein Variational Gradient Descent
                             - "rlsd": Regularized Least-Squares Descent (smooth SVGD)
                             Default: "svgd"
    
    Note:
        - Uses fixed random seed for reproducible results
        - Reports whether kernels and gradients match within numerical precision
        - Shows maximum gradient difference if they don't match
        - Useful for validating analytical gradient implementations
    """
    print(f"\n=== Cosine kernel test (repulsion_type={repulsion_type}) ===")
    torch.manual_seed(0)
    Z = torch.randn(N, D)

    # Compute gradients using both methods
    K_vec, G_vec = cosine_kernel_and_grad(Z, repulsion_type=repulsion_type)
    K_auto, G_auto = cosine_kernel_and_autograd(Z, repulsion_type=repulsion_type)

    # Compare results
    print("K close?   ", torch.allclose(K_vec, K_auto, atol=1e-6, rtol=1e-5))
    print("Grad close?", torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5))
    if not torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5):
        print("Max grad diff:", (G_vec - G_auto).abs().max().item())


def summarize_kernel(K: torch.Tensor, name: str):
    """
    Print a comprehensive summary of kernel matrix properties.
    
    This function provides detailed statistics about a kernel matrix, including
    shape, value ranges, and diagonal properties. Useful for debugging and
    understanding kernel behavior.
    
    Args:
        K (torch.Tensor): Kernel matrix to analyze
        name (str): Name of the kernel for reporting purposes
    
    Note:
        - Shows basic statistics: shape, min, max, mean
        - Displays first 5 diagonal values (should be 1 or close to 1 for normalized kernels)
        - Useful for validating kernel computation and identifying numerical issues
    """
    print(f"\n=== {name} Kernel Summary ===")
    print("Shape:", K.shape)
    print("Min  :", K.min().item())
    print("Max  :", K.max().item())
    print("Mean :", K.mean().item())
    print("Diag (first 5):", torch.diag(K)[:5])  # Check diagonal values (usually 1 or offset+1 nearby)
    
    
def step_and_report(kernel_fn, name, N=6, D=32, eta=1e-2, seed=0):
    """
    Test kernel gradient descent behavior and report results.
    
    This function tests whether the repulsion gradient correctly pushes particles apart
    by checking if the kernel sum (attraction measure) decreases after a gradient step.
    The test simulates one step of gradient descent using the computed repulsion gradient.
    
    Methodology:
    1. Generate random particle positions
    2. Compute kernel matrix and repulsion gradient
    3. Apply gradient descent step: x_new = x_old - η * gradient
    4. Compute new kernel matrix and compare attraction measures
    5. Report whether attraction decreased (success) or increased (failure)
    
    Args:
        kernel_fn: Function that returns (K, grad) where:
                  - K: Kernel matrix of shape [N, N]
                  - grad: Repulsion gradient of shape [N, D]
        name (str): Name of the kernel for reporting purposes
        N (int): Number of particles for testing. Default: 6
        D (int): Feature dimension for testing. Default: 32
        eta (float): Learning rate for gradient descent step. Default: 1e-2
        seed (int): Random seed for reproducible results. Default: 0
    
    Returns:
        Tuple[float, float, float]: (A_old, A_new, change) where:
            - A_old: Attraction measure before gradient step (sum of kernel values)
            - A_new: Attraction measure after gradient step
            - change: Difference A_new - A_old (negative indicates successful repulsion)
    
    Note:
        - Successful repulsion should decrease the attraction measure (change < 0)
        - The test uses a single gradient descent step with fixed learning rate
        - Results may vary depending on the learning rate and initial particle positions
        - Useful for validating that repulsion gradients work as expected
    
    Example:
        >>> def my_kernel_fn(x):
        ...     return rbf_kernel_and_grad(x, tau=0.5, repulsion_type="svgd")
        ...
        >>> A_old, A_new, change = step_and_report(my_kernel_fn, "My RBF Kernel")
        >>> print(f"Repulsion successful: {change < 0}")
    """
    torch.manual_seed(seed)
    x = torch.randn(N, D, requires_grad=True)

    # Compute kernel and repulsion gradient
    K, g_rep = kernel_fn(x)  # g_rep = repulsion gradient (∇R)
    A_old = K.sum()          # attraction objective function (sum of kernel values)

    # Apply gradient descent step using repulsion gradient
    x_new = (x - eta * g_rep).detach().requires_grad_(True)
    K_new, _ = kernel_fn(x_new)
    A_new = K_new.sum()

    # Report results
    change = A_new - A_old
    change_pct = (change / A_old.item()) * 100 if A_old.item() != 0 else float('inf')
    
    print(f"{name}: A(old)={A_old.item():.6f}, A(new)={A_new.item():.6f}")
    print(f"  Change: {change.item():.6f} ({change_pct:+.2f}%)")
    
    # Check if repulsion is working correctly
    if change < 0:
        print(f"  ✓ Repulsion working: attraction decreased")
    else:
        print(f"  ⚠ Repulsion may not be working: attraction increased")
    
    # Additional diagnostics
    print(f"  Gradient norm: {g_rep.norm().item():.6f}")
    print(f"  Step size: {eta}")
    
    return A_old.item(), A_new.item(), change.item()


def comprehensive_kernel_test(kernel_fn, name, N=10, D=16, num_tests=5):
    """
    Comprehensive test of kernel behavior with multiple random initializations.
    
    This function performs multiple tests of kernel gradient descent behavior
    with different random initializations to assess the robustness and reliability
    of the repulsion mechanism.
    
    Methodology:
    1. Run multiple tests with different random seeds
    2. For each test, perform gradient descent step and measure attraction change
    3. Compute success rate (percentage of tests where attraction decreased)
    4. Analyze distribution of changes across all tests
    
    Args:
        kernel_fn: Function that returns (K, grad) where:
                  - K: Kernel matrix of shape [N, N]
                  - grad: Repulsion gradient of shape [N, D]
        name (str): Name of the kernel for reporting purposes
        N (int): Number of particles for testing. Default: 10
        D (int): Feature dimension for testing. Default: 16
        num_tests (int): Number of random tests to run. Default: 5
    
    Returns:
        Tuple[float, float]: (success_rate, mean_change) where:
            - success_rate: Percentage of successful tests (attraction decreased)
            - mean_change: Average change in attraction measure across all tests
    
    Note:
        - Higher success rates indicate more reliable repulsion behavior
        - Mean change should be negative for effective repulsion
        - Results help assess the robustness of different kernel configurations
        - Useful for comparing different kernel types and parameters
    
    Example:
        >>> def my_kernel_fn(x):
        ...     return rbf_kernel_and_grad(x, tau=0.5, repulsion_type="svgd")
        ...
        >>> success_rate, mean_change = comprehensive_kernel_test(my_kernel_fn, "My RBF Kernel")
        >>> print(f"Success rate: {success_rate:.1f}%")
        >>> print(f"Mean change: {mean_change:.6f}")
    """
    print(f"\n=== Comprehensive Test: {name} ===")
    
    results = []
    for i in range(num_tests):
        A_old, A_new, change = step_and_report(kernel_fn, f"{name} (test {i+1})", N=N, D=D, seed=i)
        results.append(change)
    
    # Analyze results
    changes = torch.tensor(results)
    success_rate = (changes < 0).float().mean().item() * 100
    
    print(f"\nSummary for {name}:")
    print(f"  Success rate: {success_rate:.1f}% ({num_tests - (changes < 0).sum().item()}/{num_tests} failures)")
    print(f"  Mean change: {changes.mean().item():.6f}")
    print(f"  Std change: {changes.std().item():.6f}")
    print(f"  Min change: {changes.min().item():.6f}")
    print(f"  Max change: {changes.max().item():.6f}")
    
    return success_rate, changes.mean().item()


if __name__ == "__main__":
    """
    Main execution block for kernel testing and validation.
    
    This section runs comprehensive tests of all kernel implementations,
    including gradient validation, repulsion testing, and parameter sensitivity analysis.
    """
    torch.manual_seed(0)
    Z = torch.randn(5, 3)
    
    # Basic kernel computation tests
    print("=== RBF SVGD ===")
    K, G = rbf_kernel_and_grad(Z, tau=0.5, repulsion_type="svgd")
    print("K shape:", K.shape)
    print("G shape:", G.shape)
    print("K[0]:", K[0])
    print("G[0]:", G[0])

    print("\n=== RBF RLSD ===")
    K, G = rbf_kernel_and_grad(Z, tau=0.5, repulsion_type="rlsd")
    print("K[0]:", K[0])
    print("G[0]:", G[0])

    print("\n=== Cosine SVGD ===")
    K, G = cosine_kernel_and_grad(Z, repulsion_type="svgd")
    print("K[0]:", K[0])
    print("G[0]:", G[0])

    print("\n=== Cosine RLSD ===")
    K, G = cosine_kernel_and_grad(Z, repulsion_type="rlsd")
    print("K[0]:", K[0])
    print("G[0]:", G[0])

    # Kernel summary analysis
    print("=== Summary ===")
    # RBF SVGD
    K, _ = rbf_kernel_and_grad(Z, tau=0.5, repulsion_type="svgd")
    summarize_kernel(K, "RBF SVGD")

    # RBF RLSD
    K, _ = rbf_kernel_and_grad(Z, tau=0.5, repulsion_type="rlsd")
    summarize_kernel(K, "RBF RLSD")

    # Cosine SVGD
    K, _ = cosine_kernel_and_grad(Z, repulsion_type="svgd")
    summarize_kernel(K, "Cosine SVGD")

    # Cosine RLSD
    K, _ = cosine_kernel_and_grad(Z, repulsion_type="rlsd")
    summarize_kernel(K, "Cosine RLSD")

    # Gradient validation tests
    print("=== Gradient Validation Tests ===")
    # RBF test
    compare_rbf(repulsion_type="svgd")
    compare_rbf(repulsion_type="rlsd")
    compare_rbf(tau=0.0, repulsion_type="svgd")   # check median heuristic
    
    # Cosine test
    compare_cosine(repulsion_type="svgd")
    compare_cosine(repulsion_type="rlsd")  # RLSD requires offset > 0
    
    # Repulsion behavior tests
    print("=== Repulsion Gradient Tests ===")
    step_and_report(lambda z: rbf_kernel_and_grad(z, tau=0.5, repulsion_type="svgd"), "RBF SVGD")
    step_and_report(lambda z: rbf_kernel_and_grad(z, tau=0.5, repulsion_type="rlsd"), "RBF RLSD")
    step_and_report(lambda z: cosine_kernel_and_grad(z, repulsion_type="svgd"), "Cosine SVGD")
    step_and_report(lambda z: cosine_kernel_and_grad(z, repulsion_type="rlsd"), "Cosine RLSD")
    
    # Parameter sensitivity tests
    print("=== Parameter Sensitivity Tests ===")
    step_and_report(lambda z: rbf_kernel_and_grad(z, tau=0.1, repulsion_type="svgd"), "RBF (tau=0.1)", eta=1e-3)
    step_and_report(lambda z: rbf_kernel_and_grad(z, tau=1.0, repulsion_type="svgd"), "RBF (tau=1.0)", eta=1e-3)
    step_and_report(lambda z: rbf_kernel_and_grad(z, tau=0.0, repulsion_type="svgd"), "RBF (median)", eta=1e-3)
    
    # Comprehensive robustness tests
    print("=== Comprehensive Kernel Tests ===")
    comprehensive_kernel_test(lambda z: rbf_kernel_and_grad(z, tau=0.5, repulsion_type="svgd"), "RBF SVGD")
    comprehensive_kernel_test(lambda z: rbf_kernel_and_grad(z, tau=0.5, repulsion_type="rlsd"), "RBF RLSD")
    comprehensive_kernel_test(lambda z: cosine_kernel_and_grad(z, repulsion_type="svgd"), "Cosine SVGD")
    comprehensive_kernel_test(lambda z: cosine_kernel_and_grad(z, repulsion_type="rlsd"), "Cosine RLSD")
