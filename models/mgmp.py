import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MGMPLayer(MessagePassing):
    """
    Morphology-Guided Message Passing (MGMP) Layer.
    Augments local structural message passing (like GCN/GAT) with a global prototype signal.
    """
    def __init__(self, in_channels: int, out_channels: int, num_cwes: int, morph_dim: int):
        super().__init__(aggr='add') # Add aggregation as in GCN
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_cwes = num_cwes
        self.morph_dim = morph_dim
        
        # Linear transformation for node features
        self.lin = nn.Linear(in_channels, out_channels)
        
        # Beta gate network: determines how much a node should pay attention to prototypes
        # Input to beta is [h_v || x_morph_v], but we'll simplify to just h_v for now 
        # (assuming h_v already has morphological information integrated)
        self.beta_gate = nn.Sequential(
            nn.Linear(out_channels + morph_dim, 1),
            nn.Sigmoid()
        )
        
        # Lambda gate: balances local and global signals
        self.lambda_k = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: torch.Tensor, prototypes: torch.Tensor, x_morph: torch.Tensor):
        """
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
            edge_weight: Soft edge mask from VCSA (E,)
            prototypes: Global prototype bank P* (num_cwes, out_channels)
            x_morph: Abstract morphological embeddings for gating (N, morph_dim)
        """
        # 1. Local Message Passing
        # Add self-loops to the graph to include the node itself in aggregation
        # Note: edge_weight needs to handle self-loops. For simplicity, we create a new mask.
        edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, fill_value=1.0, num_nodes=x.size(0))
        
        # Normalize edge weights (GCN style)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Transform node features
        x_trans = self.lin(x)
        
        # Start propagating messages
        h_local = self.propagate(edge_index, x=x_trans, norm=norm)
        
        # 2. Global Prototype Attention (ProtoAttn)
        if prototypes is not None and prototypes.size(0) > 0:
            # prototypes shape: (num_cwes, out_channels)
            # Compute attention scores: (N, num_cwes)
            d_k = torch.tensor(self.out_channels, dtype=torch.float32, device=x.device)
            scores = torch.matmul(x_trans, prototypes.t()) / torch.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1) # (N, num_cwes)
            
            # Compute prototype signal: (N, out_channels)
            h_global = torch.matmul(attn_weights, prototypes)
            
            # 3. Compute beta gate
            # [h_v || x_morph_v]
            beta_in = torch.cat([x_trans, x_morph], dim=-1)
            beta_v = self.beta_gate(beta_in) # (N, 1)
            
            # 4. Combine local and global
            h_out = (1 - self.lambda_k) * h_local + self.lambda_k * (beta_v * h_global)
        else:
            h_out = h_local
            
        return F.relu(h_out)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # norm has shape [E]
        return norm.view(-1, 1) * x_j
