import torch
import numpy as np

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
