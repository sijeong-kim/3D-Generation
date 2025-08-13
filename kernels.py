# kernel_utils.py
import torch
import torch.nn.functional as F

def rbf_kernel_and_grad(features, tau=0.5, repulsion_type="svgd"):
    """
    Compute RBF kernel matrix and its gradients.
    
    Args:
        features: [N, D] tensor of feature vectors
        tau: kernel bandwidth
        normalize: whether to normalize features
    
    Returns:
        K: [N, N] kernel matrix where K[i,j] = k(x_j, x_i)
        K_grad: [N,  D_feature] kernel gradients where K_grad[i,j] = ∇_{x_j} (∑ over j k(x_j, x_i)) or ∇_{x_j} log(∑ over j k(x_j, x_i))
    """
    N, D_feature = features.shape
    
    # features: [N, D]
    z_i = features.unsqueeze(1)           # [N,1,D]
    z_j = features.unsqueeze(0)           # [1,N,D]
    sq = (z_i - z_j).pow(2).sum(-1)       # [N,N]

    h = (tau**2 if tau > 0 else (sq[~torch.eye(N, dtype=torch.bool, device=features.device)].median()
        / torch.log(torch.tensor(N+1.0, device=features.device)))).clamp_min(1e-6)

    K = torch.exp(-sq / h)                # [N,N]
    
    if repulsion_type == "svgd":
        S = K.sum()                       # scalar
    elif repulsion_type == "rlsd":
        s_i = torch.log(K.sum(dim=1) + 1e-9)  # [N]
        S = s_i.sum()                         # scalar

    # Autograd gradient wrt features (keeps graph)
    repulsion_grad = torch.autograd.grad(S, features, create_graph=True)[0]  # [N,D]
    
    return K, repulsion_grad

    # # Repulsion loss (same form you used)
    # repulsion_loss = (repulsion_grad * features).sum(dim=1).mean()


def cosine_kernel_and_grad(features, tau=0.5, repulsion_type="svgd"):
    """
    Compute cosine-similarity-based kernel matrix and its gradients.

    Args:
        features: [N, D] tensor of feature vectors
        tau: temperature (>0). smaller -> sharper kernel
        repulsion_type: "svgd" or "rlsd"

    Returns:
        K: [N, N] kernel matrix where K[i,j] = exp(cos(x_i, x_j) / tau)
        repulsion_grad: [N, D] = ∇_{features} S
                         (S = sum_ij K_ij for 'svgd';
                          S = sum_i log sum_j K_ij for 'rlsd')
    """
    assert tau is not None and tau > 0, "tau must be > 0 for cosine kernel"
    N, D = features.shape
    eps = 1e-9

    # 1) 코사인 유사도 행렬 C = \hat z \hat z^T
    z = features
    z_norm = z / (z.norm(dim=1, keepdim=True) + eps)   # [N, D]
    C = (z_norm @ z_norm.t()).clamp(-1 + 1e-7, 1 - 1e-7)  # [N, N]

    # 2) 항상 양수/PSD로 만들기 위해 exponentiated cosine kernel 사용
    #    K[i,j] = exp( cos(x_i, x_j) / tau )
    inv_tau = 1.0 / torch.clamp(torch.tensor(tau, dtype=z.dtype, device=z.device), min=1e-6)
    K = torch.exp(C * inv_tau)

    # 3) 스칼라 S 구성 (RBF 버전과 동일한 목적식 구조)
    if repulsion_type == "svgd":
        S = K.sum()
    elif repulsion_type == "rlsd":
        # log(sum_j K_ij) = logsumexp_j( C_ij / tau )
        s_i = torch.logsumexp(C * inv_tau, dim=1)  # [N]
        S = s_i.sum()
    else:
        raise ValueError("repulsion_type must be 'svgd' or 'rlsd'")

    # 4) 특징에 대한 그래디언트 (autograd)
    repulsion_grad = torch.autograd.grad(S, features, create_graph=True)[0]
    return K, repulsion_grad



@torch.no_grad()
def rbf_kernel_and_grad_old(features, tau=0.5, repulsion_type="svgd"):
    """
    Compute RBF kernel matrix and its gradients.
    
    Args:
        features: [N, D] tensor of feature vectors
        tau: kernel bandwidth
        normalize: whether to normalize features
    
    Returns:
        K: [N, N] kernel matrix where K[i,j] = k(x_j, x_i)
        K_grad: [N,  D_feature] kernel gradients where K_grad[i,j] = ∇_{x_j} (∑ over j k(x_j, x_i)) or ∇_{x_j} log(∑ over j k(x_j, x_i))
    """
    # N, D_feature = features.shape
    
    # # Compute pairwise differences and squared distances
    # z_i = features.unsqueeze(1)  # [N, 1, D_feature]
    # z_j = features.unsqueeze(0)  # [1, N, D_feature]
    
    # diffs = z_i - z_j  # [N, N, D_feature] where diffs[i,j] = z_i - z_j
    # sq_dists = (diffs ** 2).sum(-1)  # [N, N] # sq_dists[i,j] = ||z_i - z_j||^2
    
    # if tau == 0: # if tau is 0, use median bandwidth heuristic
    #     # Median bandwidth heuristic h = median(sq_dists) ** 2 / log(N)
    #     # median = the median of the pairwise distances between particles
    #     dists = sq_dists.sqrt() # [N, N] # pairwise distances between particles
    #     median_dist = torch.median(dists) # [1] # median of pairwise distances between particles
    #     h = median_dist ** 2 / torch.log(torch.tensor(N, dtype=torch.float32, device=features.device)) # [1] # h = median(dist) ** 2 / log(N)
        
    # else:
    #     h = tau ** 2 # if tau is not 0, use tau
    
    # # RBF kernel k(z_i, z_j) = exp(-||z_i - z_j||^2 / h)
    # K = torch.exp(-sq_dists / h)  # [N, N]
    
    # if repulsion_type == "svgd": # SVGD
    #     # RBF kernel gradient ∇_{z_j} sum_j k(z_i, z_j) 
    #     # = ∇_{z_j} (sum_j exp(-||z_i - z_j||^2 / h)
    #     # = sum_j ((2/h) * (z_i - z_j) * k(z_i, z_j))
    #     K_grad = -(2 / h) * K.unsqueeze(-1) * diffs  # [N, N, 1] *  [N, N, D_feature] / h # [N, N, D_feature]
    #     K_grad_sum_j = K_grad.sum(dim=1) # [N, D_feature] (sum over particles j)
    #     return K, K_grad_sum_j # [N, N], [N, D_feature]
    # elif repulsion_type == "rlsd":   # smooth SVGD (RLSD)
    #     # RBF kernel gradient ∇_{z_j} log( sum_j k(z_i, z_j) )
    #     # = ∇_{z_j} (sum_j k(z_i, z_j)) / (sum_j k(z_i, z_j))
    #     # = sum_j ((2/h) * (z_i - z_j) * k(z_i, z_j)) / (sum_j k(z_i, z_j))
    #     K_grad = -(2 / h) * K.unsqueeze(-1) * diffs  # [N, N, 1] *  [N, N, D_feature] / h # [N, N, D_feature]
    #     K_sum_j = K.sum(dim=1) # [N] (sum over particles j)
    #     K_grad_sum_j = K_grad.sum(dim=1) / K_sum_j.unsqueeze(-1) # [N, D_feature] / [N, 1] -> [N, D_feature] (sum over particles j)  # ∇_{z_j} log(∑ over j k(z_i, z_j))
    #     return K, K_grad_sum_j # [N, N], [N, D_feature]
    
    # exclude diagonal when computing the median bandwidth; clamp h and denom
    N, D_feature = features.shape
    z_i = features.unsqueeze(1)
    z_j = features.unsqueeze(0)
    diffs = z_i - z_j
    sq_dists = (diffs ** 2).sum(-1)  # [N,N]

    if tau == 0:
        mask = ~torch.eye(N, dtype=torch.bool, device=features.device)
        med = torch.median(sq_dists[mask])
        h = (med / torch.log(torch.tensor(N + 1.0, device=features.device))).clamp_min(1e-6)
    else:
        h = torch.tensor(tau**2, device=features.device, dtype=features.dtype).clamp_min(1e-12)

    K = torch.exp(-sq_dists / h)  # keep diagonal; it stabilizes sums

    # sign of K_grad is opposite to the original paper
    if repulsion_type == "svgd":
        K_grad = (2 / h) * K.unsqueeze(-1) * diffs
        return K, K_grad.sum(dim=1)
    elif repulsion_type == "rlsd":
        K_grad = (2 / h) * K.unsqueeze(-1) * diffs
        K_sum = K.sum(dim=1, keepdim=True).clamp_min(1e-9)  # avoid 0/0
        return K, K_grad.sum(dim=1) / K_sum
    else:
        raise ValueError("Invalid kernel gradient type")

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


########################################################
# # kernel_utils.py
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
# # kernel_utils.py
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