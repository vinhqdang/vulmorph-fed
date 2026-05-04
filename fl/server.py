import torch
import torch.nn.functional as F
from typing import List

class VulMorphServer:
    def __init__(self, num_cwes: int, hidden_dim: int, device: str = 'cpu'):
        self.num_cwes = num_cwes
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.global_prototypes = None

    def aggregate_prototypes(self, client_prototypes_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Morphology-Conditioned Federated Prototype Aggregation (MCFPA).
        Aggregates prototypes from all clients using CWE-affinity weighting.
        
        Args:
            client_prototypes_list: List of K tensors, each shape (num_cwes, hidden_dim)
        Returns:
            global_prototypes: Tensor of shape (num_cwes, hidden_dim)
        """
        K = len(client_prototypes_list)
        if K == 0:
            return None
            
        # Stack into (K, num_cwes, hidden_dim)
        stacked_protos = torch.stack(client_prototypes_list).to(self.device)
        
        new_global_prototypes = torch.zeros((self.num_cwes, self.hidden_dim), device=self.device)
        
        # Aggregate per CWE
        for c in range(self.num_cwes):
            # Extract prototypes for CWE c from all clients: shape (K, hidden_dim)
            p_c = stacked_protos[:, c, :]
            
            # Find which clients actually have data for this CWE
            # We assume an all-zero vector means no data (norm is 0)
            norms = torch.norm(p_c, dim=1)
            active_clients = (norms > 1e-6).nonzero(as_tuple=True)[0]
            
            num_active = len(active_clients)
            
            if num_active == 0:
                # No client has data for this CWE, keep global prototype unchanged (or zero)
                if self.global_prototypes is not None:
                    new_global_prototypes[c] = self.global_prototypes[c]
                continue
                
            if num_active == 1:
                # Only one client has data, just use their prototype
                new_global_prototypes[c] = p_c[active_clients[0]]
                continue
                
            # Compute CWE-affinity matrix for active clients
            # A_{jk,c} = cosine(p_{c,j}, p_{c,k})
            active_p_c = p_c[active_clients] # (num_active, hidden_dim)
            
            # Normalize for cosine similarity
            norm_p_c = F.normalize(active_p_c, p=2, dim=1)
            
            # Cosine similarity matrix (num_active, num_active)
            sim_matrix = torch.mm(norm_p_c, norm_p_c.t())
            
            # A_{k,c} = 1/(K-1) * sum_{j != k} A_{jk,c}
            # Mask out self-similarity (diagonal)
            mask = 1.0 - torch.eye(num_active, device=self.device)
            sim_matrix_off_diag = sim_matrix * mask
            
            # Affinity weight per active client
            A_k_c = sim_matrix_off_diag.sum(dim=1) / (num_active - 1)
            
            # Prevent negative weights (cosine can be [-1, 1])
            A_k_c = torch.clamp(A_k_c, min=0.0)
            
            sum_weights = A_k_c.sum()
            if sum_weights > 1e-6:
                # Weighted average
                weighted_sum = torch.sum(A_k_c.unsqueeze(1) * active_p_c, dim=0)
                new_global_prototypes[c] = weighted_sum / sum_weights
            else:
                # Fallback to simple average if similarities are zero/negative
                new_global_prototypes[c] = active_p_c.mean(dim=0)
                
        self.global_prototypes = new_global_prototypes
        return self.global_prototypes
