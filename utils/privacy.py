import torch
import numpy as np


def composed_epsilon(epsilon_per_round: float, rounds: int) -> float:
    """
    End-to-end privacy budget under sequential composition.

    Each federated round releases one Laplace-perturbed prototype bank per
    client, consuming epsilon_per_round. By the basic sequential composition
    theorem for pure DP, T rounds consume T * epsilon_per_round in total.
    The manuscript reports both the per-round and the composed budget.
    """
    if epsilon_per_round == float('inf'):
        return float('inf')
    return epsilon_per_round * rounds


def clip_l1(x: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Project each row of x onto the L1 ball of the given radius:
    x_i <- x_i * min(1, radius / ||x_i||_1).

    Clipping the per-sample embedding norm is what makes the sensitivity of
    the prototype mean bounded, so the Laplace mechanism below yields a real
    epsilon-DP guarantee rather than a heuristic one.
    """
    norms = x.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (radius / norms).clamp(max=1.0)
    return x * scale


def add_calibrated_laplace_noise(
    prototypes: torch.Tensor,
    counts: torch.Tensor,
    epsilon: float,
    clip_radius: float,
) -> torch.Tensor:
    """
    Per-class Laplace mechanism for CWE-conditioned prototypes.

    Each prototype p_c is the mean of N_c per-sample embeddings whose L1
    norm has been clipped to `clip_radius` R. Replacing one record changes
    p_c by at most Delta_1 = 2R / N_c in L1 norm, and a record contributes
    to exactly one class, so releasing the whole bank per round is
    epsilon-DP by parallel composition across classes.

    Args:
        prototypes:  (num_cwes, hidden_dim) prototype bank.
        counts:      (num_cwes,) number of samples contributing to each row.
        epsilon:     Per-round privacy budget (inf disables noise).
        clip_radius: L1 clipping bound R applied to per-sample embeddings.
    """
    if epsilon == float('inf') or epsilon <= 0:
        return prototypes

    noisy = prototypes.clone()
    for c in range(prototypes.size(0)):
        n_c = float(counts[c])
        if n_c <= 0:
            continue    # empty rows are all-zero and carry no information
        b = (2.0 * clip_radius / n_c) / epsilon
        noise = np.random.laplace(loc=0.0, scale=b, size=prototypes.size(1))
        noisy[c] = noisy[c] + torch.tensor(
            noise, dtype=prototypes.dtype, device=prototypes.device)
    return noisy


def add_laplace_noise(prototypes: torch.Tensor, epsilon: float, delta_f: float) -> torch.Tensor:
    """
    Applies Laplace Differential Privacy to the local prototypes.
    
    Args:
        prototypes: Tensor of shape (num_cwes, hidden_dim) containing local prototypes
        epsilon: Privacy budget. Lower epsilon means more privacy, more noise.
                 If epsilon == float('inf') or epsilon <= 0, no noise is added.
        delta_f: Global sensitivity of the prototype function. 
                 In VulMorph-Fed, this is reduced due to morphological abstraction.
    
    Returns:
        Noisy prototypes of the same shape.
    """
    if epsilon == float('inf') or epsilon <= 0:
        return prototypes
        
    # Scale of Laplace noise: b = Delta_f / epsilon
    b = delta_f / epsilon
    
    # Generate Laplace noise
    # PyTorch doesn't have a direct Laplace distribution generator in base, 
    # so we use Exponential distributions or numpy
    noise = np.random.laplace(loc=0.0, scale=b, size=prototypes.shape)
    noise_tensor = torch.tensor(noise, dtype=prototypes.dtype, device=prototypes.device)
    
    return prototypes + noise_tensor
