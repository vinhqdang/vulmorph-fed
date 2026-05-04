import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GatedGraphConv, global_max_pool, global_mean_pool
import torch.nn.functional as F

class DevignBaseline(nn.Module):
    """
    Simulated Devign Baseline (GGNN + Conv Pooling).
    Original paper: Zhou et al., NeurIPS 2019.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # GGNN layer
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        
        # 1D Conv layer for graph pooling (simplified Devign pooling)
        self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim // 2, kernel_size=3, padding=1)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, data, **kwargs):
        x = self.embedding(data.x_lex)
        
        # Project x to hidden_dim if needed
        if x.size(-1) != self.ggnn.out_channels:
            x = F.linear(x, torch.randn(self.ggnn.out_channels, x.size(-1), device=x.device))
            
        h = self.ggnn(x, data.edge_index)
        
        # Convert to batch format for Conv1d: (Batch, Channels, Nodes)
        # Note: PyG's global pooling is easier, but Devign uses convolution over nodes.
        # We simplify it here using a combination of global pooling and linear to mimic capacity.
        h_pool = global_max_pool(h, data.batch)
        
        logits = self.classifier(h_pool)
        return logits, h_pool, None # Matches signature (logits, embeddings, mask)

class GATBaseline(nn.Module):
    """
    Standard GAT Baseline (simulating CPVD base model).
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gat1 = GATConv(embed_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data, **kwargs):
        x = self.embedding(data.x_lex)
        h = F.relu(self.gat1(x, data.edge_index))
        h = F.relu(self.gat2(h, data.edge_index))
        
        h_pool = global_mean_pool(h, data.batch)
        logits = self.classifier(h_pool)
        return logits, h_pool, None
