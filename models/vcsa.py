import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

class VCSA(nn.Module):
    """
    Vulnerability-Critical Subgraph Abstraction (VCSA).
    Learns a differentiable edge mask to highlight vulnerability-critical subgraphs.
    """
    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Edge features will be concatenation of source and target node embeddings
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, tau: float = 0.5):
        """
        Args:
            x: Node embeddings, shape (num_nodes, node_dim)
            edge_index: Graph connectivity, shape (2, num_edges)
            tau: Threshold for hard masking (used during inference or later stages)
        Returns:
            edge_mask: Soft edge masks, shape (num_edges,)
        """
        row, col = edge_index
        # Concatenate source and target node features
        edge_attr = torch.cat([x[row], x[col]], dim=-1)
        
        # Compute soft mask in [0, 1]
        edge_mask = self.edge_mlp(edge_attr).squeeze(-1)
        return edge_mask

def structural_contrastive_loss(graph_embeddings: torch.Tensor, labels: torch.Tensor, cwe_types: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Computes L_SCL to encourage same-CWE vulnerable pairs to have similar graph embeddings,
    and (vulnerable, non-vulnerable) pairs to be distant.
    Args:
        graph_embeddings: (batch_size, embed_dim)
        labels: (batch_size,) - 1 for vulnerable, 0 for benign
        cwe_types: (batch_size,) - CWE class IDs, -1 if benign
        margin: Margin for the contrastive loss
    """
    batch_size = graph_embeddings.size(0)
    if batch_size < 2:
        return torch.tensor(0.0, device=graph_embeddings.device)

    loss = torch.tensor(0.0, device=graph_embeddings.device)
    num_pairs = 0

    # Extremely simplified O(N^2) pairwise comparison for illustration.
    # In a real batch with many graphs, vectorization should be used.
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            dist_sq = torch.norm(graph_embeddings[i] - graph_embeddings[j], p=2) ** 2
            
            # Positive pair: Both vulnerable and same CWE
            if labels[i] == 1 and labels[j] == 1 and cwe_types[i] == cwe_types[j] and cwe_types[i] != -1:
                loss += dist_sq
                num_pairs += 1
            # Negative pair: One vulnerable, one benign
            elif (labels[i] == 1 and labels[j] == 0) or (labels[i] == 0 and labels[j] == 1):
                loss += torch.clamp(margin - dist_sq, min=0.0)
                num_pairs += 1

    if num_pairs > 0:
        loss /= num_pairs
    return loss
