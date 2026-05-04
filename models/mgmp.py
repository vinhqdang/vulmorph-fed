import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MGMPLayer(MessagePassing):
    """
    Morphology-Guided Message Passing (MGMP) Layer.

    Augments local structural message passing (GCN-style) with a global
    semantic prototype signal, weighted by a node-level morphology gate (β)
    and a client-level balance scalar (λ_k).

    Reference: Section 3.4 of the VulMorph-Fed research plan.
    """

    def __init__(self, in_channels: int, out_channels: int, num_cwes: int, morph_dim: int):
        """
        Args:
            in_channels:  Dimensionality of incoming node features.
            out_channels: Dimensionality of output node features.
            num_cwes:     Number of CWE types in the global prototype bank.
            morph_dim:    Dimensionality of morphological embeddings (0 if ablated).
        """
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_cwes = num_cwes
        self.morph_dim = morph_dim

        # Linear transformation for local neighbourhood aggregation
        self.lin = nn.Linear(in_channels, out_channels)

        # β gate: which nodes should absorb the global prototype signal.
        # Input: [h_v || x_morph_v] (or just h_v when morph_dim == 0)
        beta_in_dim = out_channels + morph_dim if morph_dim > 0 else out_channels
        self.beta_gate = nn.Sequential(
            nn.Linear(beta_in_dim, 1),
            nn.Sigmoid()
        )

        # λ_k: learnable scalar balancing local vs global signal (per layer)
        self.lambda_k = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        prototypes: torch.Tensor,
        x_morph: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:           Node features  (N, in_channels)
            edge_index:  Graph edges    (2, E)
            edge_weight: Soft VCSA mask (E,)
            prototypes:  Global P*      (num_cwes, out_channels) or None
            x_morph:     Morphology embeddings (N, morph_dim) or empty tensor
        Returns:
            Updated node representations (N, out_channels)
        """
        # ── 1. Local aggregation (GCN-style with edge-weight) ──────────
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=x.size(0)
        )
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        x_trans = self.lin(x)                                   # (N, out_channels)
        h_local = self.propagate(edge_index, x=x_trans, norm=norm)  # (N, out_channels)

        # ── 2. Global ProtoAttn signal ─────────────────────────────────
        if prototypes is not None and prototypes.size(0) > 0:
            d_k = x_trans.size(-1) ** 0.5
            scores = (x_trans @ prototypes.t()) / d_k            # (N, num_cwes)
            attn_w = F.softmax(scores, dim=-1)
            h_global = attn_w @ prototypes                       # (N, out_channels)

            # β gate
            if self.morph_dim > 0 and x_morph.size(-1) > 0:
                beta_in = torch.cat([x_trans, x_morph], dim=-1)
            else:
                beta_in = x_trans
            beta_v = self.beta_gate(beta_in)                     # (N, 1)

            lam = torch.clamp(self.lambda_k, 0.0, 1.0)
            h_out = (1.0 - lam) * h_local + lam * (beta_v * h_global)
        else:
            h_out = h_local

        return F.relu(h_out)

    # ------------------------------------------------------------------
    # Message function
    # ------------------------------------------------------------------

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j
