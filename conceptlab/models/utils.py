import torch.nn.functional as F

import torch
import ot

def sigmoid(x, alpha=1):
    return F.sigmoid(alpha * x)



def optimal_transport_coupling(x0, x1):
    """
    Computes the optimal transport coupling between two batches of tensors.

    This function takes a batch of noise (x0) and a batch of data (x1),
    calculates the entropic optimal transport plan between them, and then
    samples from this plan to create a new noise tensor where each point
    is optimally paired with a corresponding point in the data tensor.

    Args:
        x0 (torch.Tensor): Source tensor (noise) of shape (b, ...).
        x1 (torch.Tensor): Target tensor (data) of shape (b, ...).
        reg (float): Entropic regularization strength.

        torch.Tensor: The OT-coupled noise tensor, with the same shape as x0.
    """
    b, d = x0.shape
    device = x0.device

    # Uniform marginals for the distributions
    a = torch.ones(b, device=device) / b
    b_marg = torch.ones(b, device=device) / b

    # Flatten tensors for cost matrix calculation


    # Compute cost matrix and OT plan
    M = ot.dist(x0, x1)
    G = ot.emd(a, b_marg, M/M.max())

    # Sample from the OT plan to get assignments.
    # For each point in x1 (each column of G), we sample a paired point from x0 (a row index).
    # We sample from G.T because torch.multinomial expects distributions in rows.
    # assignment_indices = torch.multinomial(G.T, 1, replacement=False).squeeze()
    assignment_indices = torch.argmax(G, dim=1)

    # Create the new coupled x0 by gathering based on the assignments
    x0_coupled = x0[assignment_indices]
    
   # assert the frobenium norm distance between x0_coupled and x1 is close to G * M
    ot_plan_cost = torch.sum(G * M)
    actual_pairing_cost = torch.sum(M[torch.arange(b, device=device), assignment_indices])

    assert torch.allclose(actual_pairing_cost / b, ot_plan_cost), \
        f"Cost mismatch: {(actual_pairing_cost / b).item():.4f} != {ot_plan_cost.item():.4f}"

    cost_improvement = torch.sum(M[torch.arange(b, device=device), assignment_indices])/torch.sum(torch.diag(M))
    return x0_coupled, cost_improvement