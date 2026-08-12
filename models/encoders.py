"""
Backbone encoders and node-feature modes for the representation study.

The experiment factorises two things that prior work conflates:

  * FEATURE MODE - what a node carries:
      lexical : embedding of the node's source text (project-specific)
      op      : embedding of its operation type in the taxonomy (invariant)
      kind    : embedding of its tree-sitter grammar kind (invariant)
      op_kind : sum of the two invariant embeddings (the proposed abstraction)

  * BACKBONE - how information propagates: GGNN, GAT or GIN.

Everything else is held fixed: depth, hidden width, readout, dropout and the
classification head are identical across every cell of the grid, so a
difference between cells is attributable to the factor that varies.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GATConv, GatedGraphConv, GINConv,
                                global_mean_pool, global_max_pool)

FEATURE_MODES = ("lexical", "op", "kind", "op_kind")
BACKBONES = ("ggnn", "gat", "gin")


class NodeFeatures(nn.Module):
    """Node-feature embedding under one of the FEATURE_MODES."""

    def __init__(self, mode: str, embed_dim: int, vocab_size: int,
                 num_op_types: int, num_kinds: int):
        super().__init__()
        assert mode in FEATURE_MODES, mode
        self.mode = mode
        if mode == "lexical":
            self.emb = nn.Embedding(vocab_size, embed_dim)
        elif mode == "op":
            self.emb = nn.Embedding(num_op_types, embed_dim)
        elif mode == "kind":
            self.emb = nn.Embedding(num_kinds, embed_dim)
        else:
            self.op_emb = nn.Embedding(num_op_types, embed_dim)
            self.kind_emb = nn.Embedding(num_kinds, embed_dim)

    def forward(self, data):
        if self.mode == "lexical":
            return self.emb(data.x_lex)
        if self.mode == "op":
            return self.emb(data.x_morph)
        if self.mode == "kind":
            return self.emb(data.x_kind)
        return self.op_emb(data.x_morph) + self.kind_emb(data.x_kind)


class RepresentationModel(nn.Module):
    """
    A detector = feature mode + backbone + a fixed head.

    The head is deliberately identical in every configuration (mean-max
    readout into a two-layer MLP) so that readout capacity cannot be confused
    with a representation or backbone effect.
    """

    def __init__(self, feature_mode: str, backbone: str,
                 vocab_size: int, num_op_types: int, num_kinds: int,
                 embed_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert backbone in BACKBONES, backbone
        self.feature_mode = feature_mode
        self.backbone_name = backbone
        self.num_layers = num_layers

        self.features = NodeFeatures(feature_mode, embed_dim, vocab_size,
                                     num_op_types, num_kinds)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        if backbone == "ggnn":
            self.backbone = GatedGraphConv(out_channels=hidden_dim,
                                           num_layers=num_layers)
        elif backbone == "gat":
            self.layers = nn.ModuleList(
                [GATConv(hidden_dim, hidden_dim, heads=1)
                 for _ in range(num_layers)])
        else:
            self.layers = nn.ModuleList([
                GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim)))
                for _ in range(num_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, prototypes: Optional[torch.Tensor] = None):
        x = self.input_proj(self.features(data))
        if self.backbone_name == "ggnn":
            h = self.backbone(x, data.edge_index)
        else:
            h = x
            for layer in self.layers:
                h = F.relu(layer(h, data.edge_index))
        h_mean = global_mean_pool(h, data.batch)
        h_max = global_max_pool(h, data.batch)
        logits = self.head(torch.cat([h_mean, h_max], dim=-1))
        # Signature matches the rest of the codebase: (logits, graph_emb, mask)
        return logits, h_mean, None
