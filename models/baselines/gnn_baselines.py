import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GatedGraphConv, global_max_pool, global_mean_pool
import torch.nn.functional as F

from data.morphology import NUM_MORPHOLOGY_TYPES


class _NodeInput(nn.Module):
    """
    Shared node-input block for all baselines.

    input_mode:
      - "lex"   : embeds raw lexical tokens (standard full-graph baseline)
      - "morph" : embeds only the abstract morphological type — i.e. the
                  baseline consumes the SAME VCSA-abstracted representation
                  as VulMorph-Fed, isolating the FL mechanism from the
                  representation (RQ2c).
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 input_mode: str = "lex",
                 num_morph_types: int = NUM_MORPHOLOGY_TYPES):
        super().__init__()
        assert input_mode in ("lex", "morph")
        self.input_mode = input_mode
        if input_mode == "lex":
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        else:
            from data.loaders.ast_graphs import NUM_AST_KINDS
            self.embedding = nn.Embedding(num_morph_types, embed_dim)
            self.kind_embedding = nn.Embedding(NUM_AST_KINDS, embed_dim)

    def forward(self, data):
        if self.input_mode == "lex":
            return self.embedding(data.x_lex)
        # Same VCSA-abstracted representation VulMorph consumes:
        # morphology type + AST grammar kind (both project-invariant).
        return self.embedding(data.x_morph) + self.kind_embedding(data.x_kind)


class DevignBaseline(nn.Module):
    """
    Devign Baseline (GGNN + global max pooling).
    Reference: Zhou et al., NeurIPS 2019.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 3, input_mode: str = "lex"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input = _NodeInput(vocab_size, embed_dim, input_mode)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, **kwargs):
        x = self.input(data)                     # (N, embed_dim)
        x = self.input_proj(x)                   # (N, hidden_dim)
        h = self.ggnn(x, data.edge_index)        # (N, hidden_dim)
        h_pool = global_max_pool(h, data.batch)  # (B, hidden_dim)
        logits = self.classifier(h_pool)         # (B, 1)
        return logits, h_pool, None


class GATBaseline(nn.Module):
    """
    Standard 2-layer GAT Baseline (CPVD-style).
    Reference: Zhang et al., IEEE TSE 2023.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 heads: int = 4, input_mode: str = "lex"):
        super().__init__()
        self.input = _NodeInput(vocab_size, embed_dim, input_mode)
        self.gat1 = GATConv(embed_dim, hidden_dim // heads, heads=heads)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, **kwargs):
        x = self.input(data)
        h = F.elu(self.gat1(x, data.edge_index))   # (N, hidden_dim)
        h = F.elu(self.gat2(h, data.edge_index))   # (N, hidden_dim)
        h_pool = global_mean_pool(h, data.batch)    # (B, hidden_dim)
        logits = self.classifier(h_pool)
        return logits, h_pool, None
