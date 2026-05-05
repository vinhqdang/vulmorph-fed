import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GatedGraphConv, global_max_pool, global_mean_pool
import torch.nn.functional as F


class DevignBaseline(nn.Module):
    """
    Devign Baseline (GGNN + global max pooling).
    Reference: Zhou et al., NeurIPS 2019.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Project token embeddings to GGNN input dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, **kwargs):
        x = self.embedding(data.x_lex)          # (N, embed_dim)
        x = self.input_proj(x)                   # (N, hidden_dim)
        h = self.ggnn(x, data.edge_index)        # (N, hidden_dim)
        h_pool = global_max_pool(h, data.batch)  # (B, hidden_dim)
        logits = self.classifier(h_pool)         # (B, 1)
        return logits, h_pool, None


class GATBaseline(nn.Module):
    """
    Standard 2-layer GAT Baseline (CPVD-style).
    Reference: Zheng et al., ASE 2021.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gat1 = GATConv(embed_dim, hidden_dim // heads, heads=heads)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, **kwargs):
        x = self.embedding(data.x_lex)
        h = F.elu(self.gat1(x, data.edge_index))   # (N, hidden_dim)
        h = F.elu(self.gat2(h, data.edge_index))   # (N, hidden_dim)
        h_pool = global_mean_pool(h, data.batch)    # (B, hidden_dim)
        logits = self.classifier(h_pool)
        return logits, h_pool, None
