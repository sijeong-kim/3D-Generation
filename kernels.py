# kernels.py
import torch
import torch.nn.functional as F

# def mask_diagonal_(K: torch.Tensor) -> torch.Tensor:
#     K.fill_diagonal_(0.0)
#     return K

# def rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd"):
#     """
#     Compute RBF kernel matrix and its gradients.
    
#     Args:
#         features: [N, D] tensor of feature vectors
#         tau: kernel bandwidth
#         normalize: whether to normalize features
    
#     Returns:
#         K: [N, N] kernel matrix where K[i,j] = k(x_j, x_i)
#         K_grad: [N,  D_feature] kernel gradients where K_grad[i,j] = ∇_{x_j} (∑ over j k(x_j, x_i)) or ∇_{x_j} log(∑ over j k(x_j, x_i))
#     """
#     N, D_feature = features.shape
    
#     # features: [N, D]
#     z_i = features.unsqueeze(1)           # [N,1,D]
#     z_j = features.unsqueeze(0)           # [1,N,D]
#     sq = (z_i - z_j).pow(2).sum(-1)       # [N,N]

#     if tau > 0:
#         h = torch.tensor(tau**2, device=features.device, dtype=features.dtype)
#     else:
#         # mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
#         # med = sq[mask].median()
#         # denom = torch.log(torch.tensor(N, device=features.device, dtype=features.dtype))
#         # h = (med / denom).clamp_min(1e-6)
        
#         if N < 2:
#             h = torch.tensor(1.0, dtype=features.dtype, device=features.device)
#         else:
#             # use unique pairs i<j
#             iu, ju = torch.triu_indices(N, N, offset=1, device=features.device)
#             m2 = sq[iu, ju].median()               # = (median distance)^2
#             logN = torch.log(torch.tensor(float(N), dtype=torch.float32, device=features.device))
#             h = (m2 / torch.clamp(logN, min=1e-6)).clamp_min(1e-6)

#     K = torch.exp(-sq / h)                # [N,N]
#     # K = K * (1 - torch.eye(N, device=features.device, dtype=features.dtype))
    
#     if repulsion_type == "svgd":
#         S = K.sum()                       # scalar
#     elif repulsion_type == "rlsd":
#         s_i = torch.log(K.sum(dim=1) + 1e-9)  # [N]
#         S = s_i.sum()                         # scalar
#     else:
#         raise ValueError("repulsion_type must be 'svgd' or 'rlsd'")

#     # Autograd gradient wrt features (keeps graph)
#     grad_similarity = torch.autograd.grad(S, features, create_graph=True)[0]  # [N,D]
    
#     return K, grad_similarity # grad_similarity = ∇S


def rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd", eps=1e-12):
    r"""
    Compute the RBF kernel matrix and its gradients.

    Given feature vectors :math:`\{z_i\}_{i=1}^N`, with :math:`z_i \in \mathbb{R}^D`,
    define squared distances :math:`d_{ij}^2 = \|z_i - z_j\|_2^2`.

    Bandwidth:
    .. math::
        h =
        \begin{cases}
            \tau^2, & \tau > 0, \\
            \dfrac{\mathrm{median}\big(\{ \|z_i - z_j\| \}_{i \neq j}\big)^2}{\log N + \epsilon}, & \tau = 0.
        \end{cases}

    RBF kernel:
    .. math::
        K_{ij} = \exp\!\left(-\frac{d_{ij}^2}{h}\right).

    Gradients w.r.t. :math:`z_i`:

    - SVGD:
      .. math::
          \nabla_{z_i} \sum_j K_{ij}
          \;=\; \frac{2}{h} \sum_j K_{ij}\,(z_j - z_i).

    - RLSD (smooth SVGD):
      .. math::
          \nabla_{z_i}\,\log \Big(\sum_j K_{ij}\Big)
          \;=\; \frac{\nabla_{z_i} \sum_j K_{ij}}{\sum_j K_{ij}}.

    Args:
        features (torch.Tensor): [N, D] feature tensor.
        tau (float): Bandwidth parameter; if 0, use the median heuristic.
        repulsion_type (str): "svgd" or "rlsd".
        eps (float): Numerical stability constant for logs/divisions.

    Returns:
        K (torch.Tensor): [N, N] kernel matrix with entries ``K[i, j] = k(z_i, z_j)``.
        grad (torch.Tensor): [N, D] gradient for each feature ``z_i`` under the chosen repulsion.
    """
    
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    
    N, D_feature = features.shape
    
    # Compute pairwise differences and squared distances
    z_i = features.unsqueeze(1)  # [N, 1, D_feature]
    z_j = features.unsqueeze(0)  # [1, N, D_feature]
    
    diffs = z_i - z_j  # [N, N, D_feature] where diffs[i,j] = z_i - z_j
    sq_dists = (diffs ** 2).sum(-1)  # [N, N] # sq_dists[i,j] = ||z_i - z_j||^2
    
    if tau == 0: # if tau is 0, use median bandwidth heuristic
        # Median bandwidth heuristic h = median(sq_dists) ** 2 / log(N)
        # median = the median of the pairwise distances between particles
        dists = sq_dists.sqrt() # [N, N] # pairwise distances between particles
        off_diag = ~torch.eye(N, dtype=torch.bool, device=features.device) # [N, N]
        valid_dists = dists[off_diag] # [N*(N-1)] # pairwise distances between particles
        median_dist = torch.median(valid_dists) if valid_dists.numel() > 0 else dists.median() # [1] # median of pairwise distances between particles
        logN = torch.log(torch.tensor(float(N), dtype=torch.float32, device=features.device))
        h = median_dist ** 2 / (logN + eps) # [1] # h = median(dist) ** 2 / log(N)
        h = h.clamp_min(eps)
    else:
        h = torch.tensor(tau ** 2, dtype=torch.float32, device=features.device).clamp_min(eps) # if tau is not 0, use tau
    
    # RBF kernel k(z_i, z_j) = exp(-||z_i - z_j||^2 / h)
    K = torch.exp(-sq_dists / h)  # [N, N]
    
    # Gradient wrt z_i:
    # ∇_{z_i} sum_j K_{ij} = (2/h) * sum_j K_{ij} * (z_j - z_i)
    K_weight = K.unsqueeze(-1)           # [N, N, 1]
    grad_svgd = (2.0 / h) * (K_weight * (-diffs)).sum(dim=1)  # use (z_j - z_i) = -diffs  -> [N, D]
    
    if repulsion_type == "svgd":
        grad = grad_svgd
    elif repulsion_type == "rlsd":
        denom = K.sum(dim=1, keepdim=True) + (eps) # [N, 1]
        grad = grad_svgd / denom
    else:
        raise ValueError("Invalid repulsion type. Use 'svgd' or 'rlsd'.")
    
    return K, grad # [N, N], [N, D_feature]


def cosine_kernel_and_grad(
    features: torch.Tensor,
    repulsion_type: str = "svgd",
    *,
    scale01: bool = True, # map [-1, 1] -> [0, 1] by default
    eps: float = 1e-12,
):
    r"""
    Compute cosine-similarity kernel K ∈ ℝ^{N×N} and its gradient w.r.t. each feature z_i.

    Let z_i ∈ ℝ^D, n_i = ||z_i||_2, and
        k_{ij} = cos(z_i, z_j) = (z_i^T z_j)/(n_i n_j).
    We optionally shift the kernel by a constant `offset`:
        K^{(eff)}_{ij} = k_{ij} + offset.
    (Note: ∂/∂z k_{ij} is unaffected by a constant offset.)

    Gradient (w.r.t. z_i) of the SVGD objective term ∑_j k_{ij}:
        ∇_{z_i} ∑_j k_{ij} = (1/n_i) ∑_j (z_j / n_j) - (∑_j k_{ij}) * (z_i / n_i^2).

    For RLSD (smooth-SVGD), we use the log-sum:
        ∇_{z_i} log ∑_j K^{(eff)}_{ij} = ( ∇_{z_i} ∑_j k_{ij} ) / ( ∑_j K^{(eff)}_{ij} ).
        
    If `scale01=True`, use K01 = 0.5 * (K + 1) ∈ [0,1].
    - SVGD on K01:  grad = 0.5 * grad_svgd(K)
    - RLSD on K01:  grad = (0.5 * grad_svgd(K)) / (0.5 * sum_j K01[i,j] + 0.5*N)
                  = grad_svgd(K) / (sum_j K01[i,j] + N)


    Args:
        features: Tensor of shape [N, D].
        repulsion_type: "svgd" or "rlsd".
        scale01: bool, map [-1, 1] -> [0, 1] by default
        eps: Numerical stability for norms.

    Returns:
        K_eff: Tensor [N, N], where K_eff[i, j] = cos(z_i, z_j) + offset.
        grad:  Tensor [N, D], the gradient w.r.t. z_i under the chosen repulsion.
    """
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    
    N, D = features.shape

    # Norms and safe inverses
    norms = features.norm(dim=1, keepdim=True).clamp_min(eps)        # [N, 1]
    inv_norms = 1.0 / norms                                          # [N, 1]

    # Cosine kernel: K = (Z Z^T) / (||z_i|| ||z_j||)
    dot = features @ features.t()                                    # [N, N]
    denom = norms @ norms.t()                                        # [N, N]
    K = dot / denom                                                  # [N, N]

    # # Effective kernel (optionally shifted) — returned to the caller
    # K_eff = K + offset                                               # [N, N]

    # -------- SVGD gradient numerator: ∇_{z_i} ∑_j k_{ij} ----------
    # term1_i = (1/n_i) * sum_j (z_j / n_j)
    sum_z_over_n = (features * inv_norms).sum(dim=0, keepdim=True)   # [1, D] = ∑_j z_j / n_j
    term1 = inv_norms * sum_z_over_n                                 # [N, 1] * [1, D] -> [N, D]

    # term2_i = (∑_j k_{ij}) * (z_i / n_i^2)
    K_sum_j = K.sum(dim=1, keepdim=True)                             # [N, 1]
    term2 = K_sum_j * (features / (norms * norms))                   # z_i / n_i^2 -> [N, D]

    grad_svgd_raw = term1 - term2                                        # [N, D]

    # -------- Choose repulsion type ----------
    if not scale01:
        if repulsion_type == "svgd":
            grad = grad_svgd_raw
        elif repulsion_type == "rlsd":
            denom = K_sum_j + (eps) # [N, 1]
            
            if torch.any(denom <= 0):
                raise ValueError("For 'rlsd', without scaling, sum_j K[i, j] must be positive."
                                 "Use scale01=True to shift to [0, 1].")
                
            grad = grad_svgd_raw / denom
            return K, grad
        else:
            raise ValueError("Invalid repulsion type. Use 'svgd' or 'rlsd'.")
    
    # scale01 = True: K01 = 0.5 * (K + 1) ∈ [0, 1]
    K01 = 0.5 * (K + 1)
    
    if repulsion_type == "svgd":
        # d/dz sum_j k_{ij} = 0.5 * d/dz sum_j K[i, j]
        grad = 0.5 * grad_svgd_raw
    elif repulsion_type == "rlsd":
        # RLSD:  log sum_j K[i, j] = log(0.5*sum_j K[i, j] + 0.5*N)
        denom = 0.5 * K_sum_j + (0.5 * N) # [N, 1]
        grad = 0.5 * grad_svgd_raw / (denom + eps)        
    else:
        raise ValueError("Invalid repulsion type. Use 'svgd' or 'rlsd'.")
    return K01, grad



# def rbf_kernel_and_autograd(features, tau=0.5, repulsion_type="svgd", eps=1e-12):
#     r"""
#     Compute the RBF kernel matrix and its gradients (via autograd).

#     Given feature vectors :math:`\{z_i\}_{i=1}^N`, with :math:`z_i \in \mathbb{R}^D`,
#     define squared distances :math:`d_{ij}^2 = \|z_i - z_j\|_2^2`.

#     Bandwidth:
#     .. math::
#         h =
#         \begin{cases}
#             \tau^2, & \tau > 0, \\
#             \dfrac{\mathrm{median}\big(\{ \|z_i - z_j\| \}_{i \neq j}\big)^2}{\log N + \epsilon}, & \tau = 0.
#         \end{cases}

#     RBF kernel:
#     .. math::
#         K_{ij} = \exp\!\left(-\frac{d_{ij}^2}{h}\right).

#     Gradients w.r.t. :math:`z_i`:
#     - SVGD: :math:`\nabla_{z_i} \sum_j K_{ij}`
#     - RLSD: :math:`\nabla_{z_i}\log \sum_j K_{ij}`

#     Notes:
#         - When :math:`\tau=0`, the bandwidth `h` is computed by the median heuristic
#           and treated as a constant w.r.t. features (detached).

#     Args:
#         features (torch.Tensor): [N, D] tensor.
#         tau (float): bandwidth; 0 -> median heuristic.
#         repulsion_type (str): {"svgd","rlsd"}.
#         eps (float): numerical stability.

#     Returns:
#         K (torch.Tensor): [N, N], K[i, j] = k(z_i, z_j)
#         grad (torch.Tensor): [N, D], gradient w.r.t. each z_i
#     """
#     assert features.dim() == 2, "`features` must be 2D [N, D]."
#     N, D = features.shape

#     # Work on a leaf tensor with grad enabled, keeping original tensor intact
#     Z = features.detach().clone().requires_grad_(True)

#     # Pairwise distances
#     z_i = Z.unsqueeze(1)                 # [N, 1, D]
#     z_j = Z.unsqueeze(0)                 # [1, N, D]
#     diffs = z_i - z_j                    # [N, N, D]
#     sq_dists = (diffs ** 2).sum(-1)      # [N, N]

#     # Bandwidth h (detach to treat as constant)
#     if tau == 0:
#         with torch.no_grad():
#             dists = sq_dists.sqrt()
#             off = ~torch.eye(N, dtype=torch.bool, device=Z.device)
#             valid = dists[off]
#             median_dist = (torch.median(valid) if valid.numel() > 0 else dists.median())
#             logN = torch.log(torch.tensor(float(N), dtype=torch.float32, device=Z.device))
#             h_val = (median_dist ** 2) / (logN + eps)
#             h_val = float(h_val.clamp_min(eps).item())
#         h = torch.tensor(h_val, dtype=Z.dtype, device=Z.device)
#     else:
#         h = torch.tensor(tau**2, dtype=Z.dtype, device=Z.device).clamp_min(eps).detach()

#     # Kernel
#     K = torch.exp(-sq_dists / h)         # [N, N]

#     grads = []
#     for i in range(N):
#         if repulsion_type == "svgd":
#             Fi = K[i].sum()
#         else:  # rlsd
#             Fi = torch.log(K[i].sum() + eps)

#         # grad w.r.t. all Z, then take only row i
#         g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
#         grads.append(g_full[i])

#     grad = torch.stack(grads, dim=0)     # [N, D]
#     return K.detach(), grad.detach()


# def cosine_kernel_and_autograd(
#     features: torch.Tensor,
#     repulsion_type: str = "svgd",
#     *,
#     offset: float = 0.0,
#     eps: float = 1e-12,
# ):
#     r"""
#     Compute cosine-similarity kernel and its gradient (via autograd).

#     Let :math:`n_i=\|z_i\|_2`, :math:`k_{ij}=\frac{z_i^\top z_j}{n_i n_j}`.
#     Optionally shift by a constant :math:`\text{offset}`: :math:`K^{(eff)}_{ij}=k_{ij}+\text{offset}`.

#     Gradients w.r.t. :math:`z_i`:
#     - SVGD: :math:`\nabla_{z_i}\sum_j k_{ij}`
#     - RLSD: :math:`\nabla_{z_i}\log\sum_j K^{(eff)}_{ij}`

#     Notes:
#         - For RLSD, ensure :math:`\sum_j K^{(eff)}_{ij} > 0`. Use positive `offset` if needed.

#     Args:
#         features (torch.Tensor): [N, D]
#         repulsion_type (str): {"svgd","rlsd"}
#         offset (float): kernel shift (useful for RLSD positivity)
#         eps (float): numerical stability

#     Returns:
#         K_eff (torch.Tensor): [N, N], cos + offset
#         grad (torch.Tensor): [N, D], gradient w.r.t. each z_i
#     """
#     assert features.dim() == 2, "`features` must be 2D [N, D]."
#     N, D = features.shape

#     # Leaf with grad
#     Z = features.detach().clone().requires_grad_(True)

#     norms = Z.norm(dim=1, keepdim=True).clamp_min(eps)    # [N,1]
#     K = (Z @ Z.t()) / (norms @ norms.t())                 # [N,N]
#     K_eff = K + offset

#     if repulsion_type == "rlsd":
#         # Check positivity (no autograd needed)
#         with torch.no_grad():
#             s = K_eff.sum(dim=1)
#             if torch.any(s <= 0):
#                 raise ValueError(
#                     "For 'rlsd', sum_j K_eff[i,j] must be positive. "
#                     "Increase `offset` (e.g., 1.0)."
#                 )

#     grads = []
#     for i in range(N):
#         if repulsion_type == "svgd":
#             Fi = K[i].sum()
#         else:  # rlsd
#             Fi = torch.log(K_eff[i].sum() + eps)

#         g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
#         grads.append(g_full[i])

#     grad = torch.stack(grads, dim=0)     # [N, D]
#     return K_eff.detach(), grad.detach()


import torch

def rbf_kernel_and_autograd(features, tau=0.5, repulsion_type="svgd", eps=1e-12):
    r"""
    Autograd-based version matching `rbf_kernel_and_grad`.
    Returns:
        K (torch.Tensor): [N, N]
        grad (torch.Tensor): [N, D], \nabla_{z_i} objective_i (per-row)
    """
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    N, D = features.shape
    rep = repulsion_type.lower()
    if rep not in {"svgd", "rlsd"}:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'.")

    # Work on a leaf tensor with grad enabled
    Z = features.detach().clone().requires_grad_(True)

    # Pairwise squared distances and kernel
    z_i = Z.unsqueeze(1)                          # [N,1,D]
    z_j = Z.unsqueeze(0)                          # [1,N,D]
    diffs = z_i - z_j                             # [N,N,D]
    sq_dists = (diffs**2).sum(-1)                 # [N,N]

    # Bandwidth h (treat as constant w.r.t. Z)
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

    K = torch.exp(-sq_dists / h)                  # [N,N]

    grads = []
    for i in range(N):
        if rep == "svgd":
            Fi = K[i].sum()                       # \sum_j K[i,j]
        else:  # rlsd
            Fi = torch.log(K[i].sum() + eps)      # log \sum_j K[i,j]

        g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
        grads.append(g_full[i])

    grad = torch.stack(grads, dim=0)              # [N,D]
    return K.detach(), grad.detach()


def cosine_kernel_and_autograd(
    features: torch.Tensor,
    repulsion_type: str = "svgd",
    *,
    scale01: bool = True,   # map [-1,1] -> [0,1]
    eps: float = 1e-12,
):
    r"""
    Autograd-based version matching `cosine_kernel_and_grad` (with `scale01`).
    Returns:
        K_out (torch.Tensor): [N, N] (K if scale01=False, else K01)
        grad (torch.Tensor): [N, D], \nabla_{z_i} objective_i (per-row)
    """
    assert features.dim() == 2, "`features` must be 2D [N, D]."
    N, D = features.shape
    rep = repulsion_type.lower()
    if rep not in {"svgd", "rlsd"}:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'.")

    # Leaf with grad
    Z = features.detach().clone().requires_grad_(True)

    norms = Z.norm(dim=1, keepdim=True).clamp_min(eps)       # [N,1]
    K = (Z @ Z.t()) / (norms @ norms.t())                    # [N,N], in [-1,1]

    if not scale01:
        # Directly use K
        if rep == "rlsd":
            with torch.no_grad():
                sums = K.sum(dim=1)
                if torch.any(sums <= 0):
                    raise ValueError("For 'rlsd' without scaling, sum_j K[i,j] must be positive. "
                                     "Use scale01=True to shift to [0,1].")

        grads = []
        for i in range(N):
            if rep == "svgd":
                Fi = K[i].sum()                              # \sum_j K[i,j]
            else:
                Fi = torch.log(K[i].sum() + eps)             # log \sum_j K[i,j]
            g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
            grads.append(g_full[i])
        grad = torch.stack(grads, dim=0)
        return K.detach(), grad.detach()

    # scale01=True: K01 = 0.5*(K + 1) ∈ [0,1]
    K01 = 0.5 * (K + 1.0)

    grads = []
    for i in range(N):
        if rep == "svgd":
            Fi = K01[i].sum()                                # \sum_j K01[i,j]
        else:
            Fi = torch.log(K01[i].sum() + eps)               # log \sum_j K01[i,j]
        g_full, = torch.autograd.grad(Fi, Z, retain_graph=True, create_graph=False)
        grads.append(g_full[i])

    grad = torch.stack(grads, dim=0)
    return K01.detach(), grad.detach()

def compare_rbf(N=5, D=3, tau=0.7, repulsion_type="svgd"):
    print(f"\n=== RBF kernel test (repulsion_type={repulsion_type}) ===")
    torch.manual_seed(0)
    Z = torch.randn(N, D)

    K_vec, G_vec = rbf_kernel_and_grad(Z, tau=tau, repulsion_type=repulsion_type)
    K_auto, G_auto = rbf_kernel_and_autograd(Z, tau=tau, repulsion_type=repulsion_type)

    print("K close?   ", torch.allclose(K_vec, K_auto, atol=1e-6, rtol=1e-5))
    print("Grad close?", torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5))
    if not torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5):
        print("Max grad diff:", (G_vec - G_auto).abs().max().item())


def compare_cosine(N=5, D=3, repulsion_type="svgd"):
    print(f"\n=== Cosine kernel test (repulsion_type={repulsion_type}) ===")
    torch.manual_seed(0)
    Z = torch.randn(N, D)

    K_vec, G_vec = cosine_kernel_and_grad(Z, repulsion_type=repulsion_type)
    K_auto, G_auto = cosine_kernel_and_autograd(Z, repulsion_type=repulsion_type)

    print("K close?   ", torch.allclose(K_vec, K_auto, atol=1e-6, rtol=1e-5))
    print("Grad close?", torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5))
    if not torch.allclose(G_vec, G_auto, atol=1e-6, rtol=1e-5):
        print("Max grad diff:", (G_vec - G_auto).abs().max().item())


def summarize_kernel(K: torch.Tensor, name: str):
    print(f"\n=== {name} Kernel Summary ===")
    print("Shape:", K.shape)
    print("Min  :", K.min().item())
    print("Max  :", K.max().item())
    print("Mean :", K.mean().item())
    print("Diag (first 5):", torch.diag(K)[:5])  # 대각 값 확인 (보통 1 또는 offset+1 근처)

if __name__ == "__main__":
    torch.manual_seed(0)
    Z = torch.randn(5, 3)
    
    # RBF
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

    # Cosine
    print("\n=== Cosine SVGD ===")
    K, G = cosine_kernel_and_grad(Z, repulsion_type="svgd")
    print("K[0]:", K[0])
    print("G[0]:", G[0])

    print("\n=== Cosine RLSD ===")
    K, G = cosine_kernel_and_grad(Z, repulsion_type="rlsd")
    print("K[0]:", K[0])
    print("G[0]:", G[0])

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

    
    
    # RBF test
    compare_rbf(repulsion_type="svgd")
    compare_rbf(repulsion_type="rlsd")
    compare_rbf(tau=0.0, repulsion_type="svgd")   # median heuristic도 체크
    
    # Cosine test
    compare_cosine(repulsion_type="svgd")
    compare_cosine(repulsion_type="rlsd")  # RLSD는 offset > 0 필요






# def cosine_kernel_and_grad(features, tau=5, repulsion_type="svgd"): # tau = 2.0~5.0 soft, 0.1~0.5 sharp
#     """
#     Compute a cosine-similarity kernel matrix and its gradient w.r.t. the features.

#     Args:
#         features (torch.Tensor): [N, D] feature matrix (requires_grad=True if you want nonzero grads)
#         tau (float): positive temperature (> 0). Smaller values make the kernel sharper.
#         repulsion_type (str): "svgd" or "rlsd".
#             - "svgd": objective S = sum_{i,j} K_ij
#             - "rlsd": objective S = sum_i log( sum_j K_ij ) = sum_i logsumexp_j( C_ij / tau )

#     Returns:
#         K (torch.Tensor): [N, N] kernel matrix where K[i, j] = exp( cos(x_i, x_j) / tau )
#         repulsion_grad (torch.Tensor): [N, D] gradient dS/d(features)
#     """
#     assert tau is not None and tau > 0, "tau must be > 0 for cosine kernel"
#     N, D = features.shape
#     eps = 1e-9

#     # 1) Cosine similarity matrix C = \hat{z} \hat{z}^T
#     z = features
#     z_norm = z / (z.norm(dim=1, keepdim=True) + eps)   # [N, D]
#     C = (z_norm @ z_norm.t()).clamp(-1 + 1e-7, 1 - 1e-7)  # [N, N]

#     # 2) Exponentiated cosine kernel to keep values positive / PSD
#     #    K[i,j] = exp( cos(x_i, x_j) / tau )
#     inv_tau = 1.0 / torch.clamp(torch.tensor(tau, dtype=z.dtype, device=z.device), min=1e-6)
#     K = torch.exp(C * inv_tau)
#     K = K * (1 - torch.eye(N, device=z.device, dtype=z.dtype))

#     # 3) Build the scalar objective S (same structure as the RBF version)
#     if repulsion_type == "svgd":
#         S = K.sum()
#     elif repulsion_type == "rlsd":
#         # log(sum_j K_ij) = logsumexp_j( C_ij / tau )
#         s_i = torch.logsumexp(C * inv_tau, dim=1)  # [N]
#         S = s_i.sum()
#     else:
#         raise ValueError("repulsion_type must be 'svgd' or 'rlsd'")

#     # 4) Gradient w.r.t. the features (autograd)
#     grad_similarity = torch.autograd.grad(S, features, create_graph=True)[0]
#     return K, grad_similarity # grad_similarity = ∇S



def step_and_report(kernel_fn, name):
    torch.manual_seed(0)
    x = torch.randn(6, 32, requires_grad=True)

    K, g_rep = kernel_fn(x)  # g_rep = 끌림(∇A)
    A = K.sum()              # 끌림 목적함수

    # 1) g_rep이 진짜 반발인지: A가 증가하면 안 됨 (하강 스텝에서 A가 ↓)
    eta = 1e-2
    x1 = (x - eta * g_rep).detach().requires_grad_(True)   # 경사하강
    K1, _ = kernel_fn(x1)
    print(f"{name}: A(old)={A.item():.6f}, A(new)={K1.sum().item():.6f} (expect new < old)")

if __name__ == "__main__":
    
    # 1) g_rep이 진짜 반발인지: A가 증가하면 안 됨 (하강 스텝에서 A가 ↓)    
    step_and_report(lambda z: rbf_kernel_and_grad(z, tau=0.5, repulsion_type="svgd"), "RBF")
    step_and_report(lambda z: cosine_kernel_and_grad(z, repulsion_type="svgd"), "Cosine")

########################################################
# # loss_utils.py
# import torch
# import torch.nn.functional as F

# def rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd"):
#     X = features
#     N, D = X.shape
#     Xi = X.unsqueeze(1)            # [N,1,D]
#     Xj = X.unsqueeze(0)            # [1,N,D]
#     diffs = Xi - Xj                # [N,N,D]
#     sq = (diffs**2).sum(-1)        # [N,N]
#     if tau == 0:
#         mask = ~torch.eye(N, dtype=torch.bool, device=X.device)
#         med = torch.median(sq[mask])
#         h = (med / torch.log(torch.tensor(N+1.0, device=X.device))).clamp_min(1e-6)
#     else:
#         h = torch.tensor(tau**2, device=X.device, dtype=X.dtype)

#     K = torch.exp(-sq / h)         # define: K[i,j] = k(x_i, x_j)
#     eye = torch.eye(N, device=X.device, dtype=X.dtype)
#     K = K * (1 - eye)
    
#     G_i = (-(2.0 / h)) * (K.unsqueeze(-1) * diffs).sum(dim=1)  # [N,D], sum over j

#     if repulsion_type == "svgd":
#         return K, G_i
#     elif repulsion_type == "rlsd":
#         den = K.sum(dim=1, keepdim=True).clamp_min(1e-9)           # [N,1]
#         return K, G_i / den
#     else:
#         raise ValueError("repulsuion_type must be 'svgd' or 'rlsd'")


########################
# # loss_utils.py
# import torch
# import torch.nn.functional as F

# def rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd"):
#     """
#     Compute RBF kernel matrix and its gradients.
    
#     Args:
#         features: [N, D] tensor of feature vectors
#         tau: kernel bandwidth
#         normalize: whether to normalize features
    
#     Returns:
#         K: [N, N] kernel matrix where K[i,j] = k(x_j, x_i)
#         K_grad: [N,  D_feature] kernel gradients where K_grad[i,j] = ∇_{x_j} (∑ over j k(x_j, x_i)) or ∇_{x_j} log(∑ over j k(x_j, x_i))
#     """
#     N, D_feature = features.shape
    
#     # Compute pairwise differences and squared distances
#     z_i = features.unsqueeze(1)  # [N, 1, D_feature]
#     z_j = features.unsqueeze(0)  # [1, N, D_feature]
    
#     diffs = z_i - z_j  # [N, N, D_feature] where diffs[i,j] = z_i - z_j
#     sq_dists = (diffs ** 2).sum(-1)  # [N, N] # sq_dists[i,j] = ||z_i - z_j||^2
    
#     if tau == 0: # if tau is 0, use median bandwidth heuristic
#         # Median bandwidth heuristic h = median(sq_dists) ** 2 / log(N)
#         # median = the median of the pairwise distances between particles
#         mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
#         med = torch.median(sq_dists[mask])
#         h = (med / torch.log(torch.tensor(N+1.0, device=features.device))).clamp_min(1e-6)
#     else:
#         h = torch.tensor(tau**2, device=features.device, dtype=features.dtype) # if tau is not 0, use tau

#     # RBF kernel k(z_i, z_j) = exp(-||z_i - z_j||^2 / h)
#     K = torch.exp(-sq_dists / h)  # [N, N]
#     K = K * (1 - torch.eye(N, device=features.device, dtype=features.dtype)) # mask out diagonal elements

#     if repulsion_type == "svgd": # SVGD
#         # RBF kernel gradient ∇_{z_j} sum_j k(z_i, z_j) 
#         # = ∇_{z_j} (sum_j exp(-||z_i - z_j||^2 / h)
#         # = sum_j ((2/h) * (z_i - z_j) * k(z_i, z_j))
#         K_grad = -(2 / h) * K.unsqueeze(-1) * diffs  # [N, N, 1] *  [N, N, D_feature] / h # [N, N, D_feature]
#         K_grad_sum_j = K_grad.sum(dim=1) # [N, D_feature] (sum over particles j)
#         return K, K_grad_sum_j # [N, N], [N, D_feature]
#     elif repulsion_type == "rlsd":   # smooth SVGD (RLSD)
#         # RBF kernel gradient ∇_{z_j} log( sum_j k(z_i, z_j) )
#         # = ∇_{z_j} (sum_j k(z_i, z_j)) / (sum_j k(z_i, z_j))
#         # = sum_j ((2/h) * (z_i - z_j) * k(z_i, z_j)) / (sum_j k(z_i, z_j))
#         K_grad = -(2 / h) * K.unsqueeze(-1) * diffs  # [N, N, 1] *  [N, N, D_feature] / h # [N, N, D_feature]
#         K_sum_j = K.sum(dim=1) # [N] (sum over particles j)
#         K_grad_sum_j = K_grad.sum(dim=1) / K_sum_j.unsqueeze(-1) # [N, D_feature] / [N, 1] -> [N, D_feature] (sum over particles j)  # ∇_{z_j} log(∑ over j k(z_i, z_j))
#         return K, K_grad_sum_j # [N, N], [N, D_feature]
#     else:
#         raise ValueError("Invalid kernel gradient type")

# if __name__ == "__main__":
#     N, D = 4, 128
#     features = torch.randn(N, D) # [N, D]
#     print("Original features:")
#     print(features)
#     print(f"\nFeature norms - min: {torch.norm(features, dim=1).min():.4f}, max: {torch.norm(features, dim=1).max():.4f}")
    
#     for tau in [0.0, 0.5, 1.0, 2.0]:
#         for repulsion_type in ["svgd", "rlsd"]:
#             K, K_grad_sum_j = rbf_kernel_and_grad(features, tau=tau, repulsion_type=repulsion_type)
#             print(f"\nKernel matrix:")
#             print(K)
#             print(f"\nKernel gradient sum over j:")
#             print(K_grad_sum_j)