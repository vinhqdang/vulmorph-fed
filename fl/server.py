import torch
import torch.nn.functional as F
from typing import List, Optional


class VulMorphServer:
    """
    VulMorph-Fed Federated Server.

    Implements Morphology-Conditioned Federated Prototype Aggregation (MCFPA):
      - Gathers Laplace-noised CWE prototypes from all K clients.
      - Computes a CWE-affinity weight matrix A_{jk,c} = cosine(p̃_{c,j}, p̃_{c,k}).
      - Produces a semantics-aware global prototype bank P*.

    When `use_cwe_affinity=False` reverts to uniform averaging (FedAvg-style)
    for ablation study variant "w/o CWE-affinity".

    Reference: Section 3.3, Algorithm steps 6-8.
    """

    def __init__(
        self,
        num_cwes: int,
        hidden_dim: int,
        device: str = 'cpu',
        use_cwe_affinity: bool = True,
    ):
        self.num_cwes = num_cwes
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.use_cwe_affinity = use_cwe_affinity
        self.global_prototypes: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def aggregate_prototypes(
        self, client_prototypes_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        MCFPA aggregation.

        Args:
            client_prototypes_list: K tensors of shape (num_cwes, hidden_dim).
        Returns:
            Updated global prototype bank (num_cwes, hidden_dim).
        """
        K = len(client_prototypes_list)
        if K == 0:
            return self.global_prototypes

        # Stack → (K, num_cwes, hidden_dim)
        stacked = torch.stack(
            [p.to(self.device) for p in client_prototypes_list]
        )
        new_global = torch.zeros(
            (self.num_cwes, self.hidden_dim), device=self.device
        )

        for c in range(self.num_cwes):
            p_c = stacked[:, c, :]                             # (K, hidden_dim)

            # Active clients: those with non-zero prototypes for CWE c
            norms = p_c.norm(dim=1)                             # (K,)
            active = (norms > 1e-6).nonzero(as_tuple=True)[0]
            n_active = len(active)

            if n_active == 0:
                # No client has data for this CWE — retain previous global
                if self.global_prototypes is not None:
                    new_global[c] = self.global_prototypes[c]
                continue

            if n_active == 1:
                new_global[c] = p_c[active[0]]
                continue

            active_p = p_c[active]                              # (n_active, d)

            if not self.use_cwe_affinity:
                # Uniform average (ablation: w/o CWE-affinity)
                new_global[c] = active_p.mean(dim=0)
                continue

            # Cosine-affinity-weighted aggregation
            norm_p = F.normalize(active_p, p=2, dim=1)
            sim = norm_p @ norm_p.t()                          # (n_active, n_active)

            mask = 1.0 - torch.eye(n_active, device=self.device)
            A = (sim * mask).sum(dim=1) / (n_active - 1)       # per-client weight
            A = A.clamp(min=0.0)

            denom = A.sum()
            if denom > 1e-6:
                new_global[c] = (A.unsqueeze(1) * active_p).sum(0) / denom
            else:
                new_global[c] = active_p.mean(dim=0)

        self.global_prototypes = new_global
        return self.global_prototypes
