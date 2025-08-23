# kernels.py
import torch

def cosine_kernel_and_grad(
    features: torch.Tensor,
    repulsion_type: str = "svgd",
    *,
    beta: float = 1.0,        # temperature exponent (β). β>1: sharpen, 0<β<1: soften
    eps_shift: float = 1e-3,  # small positive shift to keep kernel strictly > 0
    eps: float = 1e-12,
):
    r"""
    Shifted + temperature-scaled cosine kernel and its gradients for SVGD / RLSD.

    Steps
    -----
    1) Raw cosine:
         K = cos(z_i, z_j) ∈ [-1, 1]
    2) Positive, PSD-shifted kernel:
         K01  = 0.5 * (K + 1)          ∈ [0, 1]
         Kpos = (1 - εs) * K01 + εs    ∈ [εs, 1],  εs = eps_shift
         dKpos/dK = c = 0.5 * (1 - εs)
    3) Temperature (exponent) on positive kernel:
         Ktemp = Kpos^β                 (β > 0)

    Gradients
    ---------
    Let n_i = ||z_i||, ẑ_i = z_i / n_i, and define pairwise weights
         W_ij = β * (dKpos/dK) * Kpos_ij^(β - 1) = β * c * Kpos_ij^(β - 1).

    Then:
      SVGD:  ∇_{z_i} ∑_j Ktemp_ij
             = (1/n_i) * ∑_j W_ij * ẑ_j  -  ( ∑_j W_ij * K_ij ) * ( z_i / n_i^2 )

      RLSD:  ∇_{z_i} log ∑_j Ktemp_ij
             = [ ∇_{z_i} ∑_j Ktemp_ij ] / [ ∑_j Ktemp_ij ].

    Args
    ----
    features : (N, D) tensor of feature vectors.
    repulsion_type : "svgd" or "rlsd".
    beta : temperature exponent (>0). Larger → sharper kernel mass.
    eps_shift : small positive shift to keep Kpos strictly positive.
    eps : numerical epsilon.

    Returns
    -------
    K_eff : (N, N) tensor
        The effective kernel actually summed (i.e., Ktemp).
    grad : (N, D) tensor
        Gradient w.r.t. 'features' implementing the chosen repulsion.
    """
    assert features.dim() == 2, "`features` must be [N, D]"
    assert beta > 0.0, "`beta` must be > 0"

    N, D = features.shape

    # norms & cosine similarity
    norms = features.norm(dim=1, keepdim=True).clamp_min(eps)   # [N,1]
    inv_norms = 1.0 / norms                                     # [N,1]
    dot = features @ features.t()                               # [N,N]
    denom = norms @ norms.t()                                   # [N,N]
    K = dot / denom                                            # cos ∈ [-1,1]

    # positive + PSD shift
    c = 0.5 * (1.0 - eps_shift)                                # dKpos/dK
    Kpos = c * (K + 1.0) + eps_shift                           # ∈ [eps_shift, 1]

    # temperature (exponent)
    if beta == 1.0:
        Ktemp = Kpos
        W = c * torch.ones_like(Kpos)  # β*c*Kpos^(β-1) = c
    else:
        Ktemp = Kpos.pow(beta)                      # Kpos^β
        W = (beta * c) * Kpos.clamp_min(eps).pow(beta - 1.0)  # β*c*Kpos^(β-1)

    # common terms
    z_over_n = features * inv_norms                 # [N,D]
    term1 = inv_norms * (W @ z_over_n)             # [N,1] * ([N,N]@[N,D]) = [N,D]
    s = (W * K).sum(dim=1, keepdim=True)           # [N,1]
    term2 = s * (features / (norms * norms))       # [N,D]

    grad_svgd = term1 - term2                      # [N,D]

    if repulsion_type.lower() == "svgd":
        return Ktemp, grad_svgd.detach()

    elif repulsion_type.lower() == "rlsd":
        denom_pos = Ktemp.sum(dim=1, keepdim=True).clamp_min(eps)  # [N,1]
        grad = grad_svgd / denom_pos
        return Ktemp, grad.detach()

    else:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'")



def rbf_kernel_and_grad(
    features: torch.Tensor,
    beta: float = 1.0,
    repulsion_type: str = "svgd",
    eps: float = 1e-12,
):
    r"""
    RBF kernel with median heuristic bandwidth and temperature scaling.
    
    Given features z_i ∈ R^D:
      d_ij^2 = ||z_i - z_j||^2

    Median heuristic bandwidth:
      h_med = median_{i≠j}(||z_i - z_j||)^2 / log(N)

    Temperature scaling:
      h = h_med / beta

    Kernel:
      K_ij = exp(-d_ij^2 / h)

    Gradients:
    - SVGD:
        ∇_{z_i} ∑_j K_ij
        = (2/h) ∑_j K_ij (z_j - z_i)

    - RLSD:
        ∇_{z_i} log ∑_j K_ij
        = ( (2/h) ∑_j K_ij (z_j - z_i) ) / (∑_j K_ij)

    Args:
        features: [N, D] particle features
        beta: temperature scaling factor (>0). Larger beta → sharper kernel
        repulsion_type: "svgd" or "rlsd"
        eps: numerical stability constant

    Returns:
        K: [N, N] kernel matrix
        grad: [N, D] repulsion gradient
    """
    assert features.dim() == 2, "`features` must be [N, D]."
    N, D = features.shape

    # pairwise differences
    z_i = features.unsqueeze(1)  # [N,1,D]
    z_j = features.unsqueeze(0)  # [1,N,D]
    diffs = z_i - z_j
    sq_dists = (diffs ** 2).sum(-1)  # [N,N]

    # median heuristic
    dists = sq_dists.sqrt()
    off_diag = ~torch.eye(N, dtype=torch.bool, device=features.device)
    valid_dists = dists[off_diag]
    median_dist = torch.median(valid_dists) if valid_dists.numel() > 0 else dists.median()
    logN = torch.log(torch.tensor(float(N), device=features.device))
    h_med = (median_dist ** 2) / (logN + eps)

    # temperature scaling
    h = h_med / beta
    h = h.clamp_min(eps)

    # kernel
    K = torch.exp(-sq_dists / h)

    # gradient (SVGD form)
    grad_svgd = (2.0 / h) * (K.unsqueeze(-1) * (-diffs)).sum(dim=1)

    if repulsion_type.lower() == "svgd":
        grad = grad_svgd
    elif repulsion_type.lower() == "rlsd":
        denom = K.sum(dim=1, keepdim=True) + eps
        grad = grad_svgd / denom
    else:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'")

    return K, grad.detach()
