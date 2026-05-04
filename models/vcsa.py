import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class VCSA(nn.Module):
    """
    Vulnerability-Critical Subgraph Abstraction (VCSA).

    Learns a differentiable soft edge-importance mask ε ∈ [0,1]^|E|
    via a two-layer MLP over edge embeddings (source || target).
    The mask is trained end-to-end with the classifier.

    Reference: Section 3.2 of the VulMorph-Fed research plan.
    """

    def __init__(self, node_dim: int, hidden_dim: int = 64):
        """
        Args:
            node_dim:   Dimensionality of node feature vectors.
            hidden_dim: Hidden layer size of the edge MLP.
        """
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:          Node embeddings (N, node_dim)
            edge_index: Graph connectivity (2, E)
        Returns:
            edge_mask: Soft edge importance scores ∈ [0, 1], shape (E,)
        """
        row, col = edge_index
        edge_attr = torch.cat([x[row], x[col]], dim=-1)    # (E, 2·node_dim)
        edge_mask = self.edge_mlp(edge_attr).squeeze(-1)   # (E,)
        return edge_mask


def structural_contrastive_loss(
    graph_embeddings: torch.Tensor,
    labels: torch.Tensor,
    cwe_types: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Structural Contrastive Loss (L_SCL).

    Positive pairs:  same-CWE vulnerable functions → minimise distance.
    Negative pairs:  (vulnerable, benign) pairs    → push beyond margin.

    Reference: Section 3.2, Equation for L_SCL.

    Args:
        graph_embeddings: (B, d)
        labels:           (B,)  — 1 vulnerable, 0 benign
        cwe_types:        (B,)  — CWE class id; -1 if benign
        margin:           Contrastive margin m.
    Returns:
        Scalar loss.
    """
    B = graph_embeddings.size(0)
    if B < 2:
        return graph_embeddings.new_tensor(0.0)

    # Pairwise squared L2 distances  (B, B)
    diff = graph_embeddings.unsqueeze(0) - graph_embeddings.unsqueeze(1)
    dist_sq = (diff ** 2).sum(-1)

    loss = graph_embeddings.new_tensor(0.0)
    n_pairs = 0

    for i in range(B):
        for j in range(i + 1, B):
            # Positive pair: both vulnerable & same CWE
            if (labels[i] == 1 and labels[j] == 1
                    and cwe_types[i] == cwe_types[j]
                    and cwe_types[i] >= 0):
                loss = loss + dist_sq[i, j]
                n_pairs += 1
            # Negative pair: one vulnerable, one benign
            elif (labels[i] != labels[j]):
                loss = loss + torch.clamp(margin - dist_sq[i, j], min=0.0)
                n_pairs += 1

    return loss / max(n_pairs, 1)
