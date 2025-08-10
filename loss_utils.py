# loss_utils.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd", device="cuda"):
    """
    Compute RBF kernel matrix and gradient sums used in SVGD-like updates.

    Args:
        features: [N, D] tensor (x_i)
        tau: kernel bandwidth. If 0, use median heuristic on pairwise squared distances.
        gradient_type: "svgd" or "rlsd"

    Returns:
        K: [N, N], where K[i,j] = k(x_i, x_j)
        G: [N, D]
           if gradient_type == "svgd":
               G[i] = sum_j ∇_{x_i} k(x_i, x_j)          # wrt x_i
           if gradient_type == "rlsd":
               G[i] = ∇_{x_i} log( sum_j k(x_i, x_j) )
    """
    features = features.to(device)
    X = features  # [N, D]
    N, D = X.shape

    Xi = X.unsqueeze(1)  # [N,1,D]
    Xj = X.unsqueeze(0)  # [1,N,D]
    diffs = Xi - Xj      # [N,N,D]; diffs[i,j] = x_i - x_j
    sq_dists = (diffs ** 2).sum(-1)  # [N,N]

    if tau == 0:
        # Median heuristic on squared distances (exclude diagonal)
        with torch.no_grad():
            mask = ~torch.eye(N, dtype=torch.bool, device=X.device)
            med = torch.median(sq_dists[mask])
            h = med / torch.log(torch.tensor(N + 1.0, device=X.device))
            h = torch.clamp(h, min=1e-6)
    else:
        h = torch.tensor(tau**2, device=X.device, dtype=X.dtype)

    K = torch.exp(-sq_dists / h)  # [N,N], K[i,j] = k(x_i,x_j)

    if repulsion_type == "svgd":
        # ∇_{x_i} k(x_i,x_j) = -(2/h) (x_i - x_j) k
        G = -(2.0 / h) * (K.unsqueeze(-1) * diffs).sum(dim=1)  # [N,D], sum over j
        return K, G

    elif repulsion_type == "rlsd":
        # ∇_{x_i} log( sum_j k(x_i,x_j) ) = (sum_j ∇_{x_i}k) / (sum_j k)
        num = -(2.0 / h) * (K.unsqueeze(-1) * diffs).sum(dim=1)   # [N,D]
        den = K.sum(dim=1, keepdim=True).clamp_min(1e-9)          # [N,1]
        G = num / den                                             # [N,D]
        return K, G

    else:
        raise ValueError("gradient_type must be 'svgd' or 'rlsd'")


if __name__ == "__main__":
    N, D = 4, 128
    features = torch.randn(N, D) # [N, D]
    print("Original features:")
    print(features)
    print(f"\nFeature norms - min: {torch.norm(features, dim=1).min():.4f}, max: {torch.norm(features, dim=1).max():.4f}")
    
    for tau in [0.0, 0.5, 1.0, 2.0]:
        for repulsion_type in ["svgd", "rlsd"]:
            K, K_grad_sum_j = rbf_kernel_and_grad(features, tau=tau, repulsion_type=repulsion_type)
            print(f"\nKernel matrix:")
            print(K)
            print(f"\nKernel gradient sum over j:")
            print(K_grad_sum_j)