# loss_utils.py
import torch
import torch.nn.functional as F

@torch.no_grad()
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