import torch
import torch.nn.functional as F


def compute_V(x, y_pos, y_neg, T):
    """
    Compute the drifting field V based on positive and negative samples.

    x: [N, D] generated samples
    y_pos: [N_pos, D] real/positive samples
    y_neg: [N_neg, D] negative samples (often the same as x)
    T: temperature (tau)
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    N_neg = y_neg.shape[0]

    # compute pairwise distance
    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]

    # ignore self (if y_neg is x)
    # Using a small epsilon to avoid exactly zero distance if not ignoring self
    if x is y_neg or torch.allclose(x, y_neg):
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    # compute logits
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T

    # concat for normalization
    logit = torch.cat([logit_pos, logit_neg], dim=1)  # [N, N_pos + N_neg]

    # normalize along both dimensions (A_row and A_col in paper)
    A_row = F.softmax(logit, dim=-1)
    A_col = F.softmax(logit, dim=-2)
    A = torch.sqrt(A_row * A_col)

    # back to [N, N_pos] and [N, N_neg]
    A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)

    # compute the weights (W_pos, W_neg in paper)
    W_pos = A_pos  # [N, N_pos]
    W_neg = A_neg  # [N, N_neg]

    # W_pos *= A_neg.sum(dim=1,keepdim=True)
    # W_neg *= A_pos.sum(dim=1,keepdim=True)
    W_pos = W_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = W_neg * A_pos.sum(dim=1, keepdim=True)

    drift_pos = W_pos @ y_pos  # [N, D]
    drift_neg = W_neg @ y_neg  # [N, D]

    V = drift_pos - drift_neg
    
    # Debug prints for V computation
    print(f"  [compute_V] drift_pos mean: {drift_pos.mean().item():.6f}, std: {drift_pos.std().item():.6f}")
    print(f"  [compute_V] drift_neg mean: {drift_neg.mean().item():.6f}, std: {drift_neg.std().item():.6f}")
    print(f"  [compute_V] V mean: {V.mean().item():.6f}, max: {V.max().item():.6f}, min: {V.min().item():.6f}")
    return V


def compute_drifting_loss(x, y_pos, y_neg, temperatures=[0.05]):
    """
    Compute aggregated drifting loss over multiple temperatures.
    """
    B, D = x.shape

    # Flatten if needed, but here we expect [B, D] where D is the flattened dimension

    # Feature normalization as per Appendix A.8
    # "Intuitively, we want the average distance to be sqrt(C_j)"
    # dist_j(x, y) = ||phi_j(x) - phi_j(y)|| / S_j
    # S_j = 1/sqrt(C_j) * E[||phi_j(x) - phi_j(y)||]

    # Concatenate all samples for global distance normalization
    all_samples = torch.cat([x, y_pos], dim=0)
    pairwise_dist = torch.cdist(all_samples, all_samples)
    S_j = torch.mean(pairwise_dist)/ (D ** 0.5 + 1e-6)
    
    print(f"[Drift Loss] Feature Normalization Scale S_j: {S_j.item():.6f}")

    # Normalized samples
    x_norm = x / (S_j.detach() + 1e-6)
    y_pos_norm = y_pos / (S_j.detach() + 1e-6)
    y_neg_norm = y_neg / (S_j.detach() + 1e-6)
    
    print(f"[Drift Loss] x_norm mean: {x_norm.mean().item():.6f}")

    metrics = {
        "train/drifting_S_j": S_j.item()
    }

    V_total = torch.zeros_like(x)
    for T in temperatures:
        print(f"[Drift Loss] Processing Temperature T={T}")
        # Note: Paper uses T * sqrt(D) for normalized features
        scaled_T = T * (D ** 0.5)
        V_t = compute_V(x_norm, y_pos_norm, y_neg_norm, scaled_T)
        
        # Drift normalization as per Appendix A.8
        # lambda_j = sqrt( E[ ||V_j||^2 / D ] )
        lambda_j = torch.sqrt(torch.mean(torch.sum(V_t**2, dim=-1)) / D + 1e-6)
        
        print(f"[Drift Loss] T={T}: lambda_j (Drift Magnitude - TRACK THIS FOR CONVERGENCE): {lambda_j.item():.6f}")
        metrics[f"train/drifting_lambda_T{T}"] = lambda_j.item()
        
        V_t = V_t / (lambda_j.detach() + 1e-6)
        V_total = V_total + V_t

    print(f"[Drift Loss] V_total mean: {V_total.mean().item():.6f}, std: {V_total.std().item():.6f}")

    # MSE loss: MSE(phi_j(x) - sg(phi_j(x) + V_j))
    # Note: drifting utility works on normalized features.
    target = (x_norm + V_total).detach()
    loss = F.mse_loss(x_norm, target)
    print(f"[Drift Loss] Final Loss: {loss.item():.6f}")

    return loss, metrics
