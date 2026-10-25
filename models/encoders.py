"""
Backbone encoders and node-feature modes for the representation study.

The experiment factorises two things that prior work conflates:

  * FEATURE MODE - what a node carries:
      lexical : embedding of the node's source text (project-specific)
      op      : embedding of its operation type in the taxonomy (invariant)
      kind    : embedding of its tree-sitter grammar kind (invariant)
      op_kind : sum of the two invariant embeddings (the proposed abstraction)
      lex_kind  : lexical embedding + kind embedding (what Devign-style
                  detectors do: token text plus node type)
      anon      : alpha-renamed token (identifiers replaced by VAR_i, ...)
      anon_kind : alpha-renamed token + kind embedding
      kind_low  : capacity control, kind embedding squeezed through 9 dims
      op_wide   : capacity control, 9-row taxonomy table widened until it has
                  as many parameters as the kind table

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
from torch_geometric.utils import to_dense_batch

FEATURE_MODES = ("lexical", "op", "kind", "op_kind",
                 "lex_kind", "anon", "anon_kind", "kind_low", "op_wide")
BACKBONES = ("ggnn", "gat", "gin", "tf")
EDGE_SETS = {"all": (0, 1, 2, 3), "ast": (0, 1), "ast_sib": (0, 1, 2),
             "ast_du": (0, 1, 3)}


class NodeFeatures(nn.Module):
    """Node-feature embedding under one of the FEATURE_MODES."""

    def __init__(self, mode: str, embed_dim: int, vocab_size: int,
                 num_op_types: int, num_kinds: int, anon_size: int = 2048):
        super().__init__()
        assert mode in FEATURE_MODES, mode
        self.mode = mode
        # Per-fold map from corpus-wide token ids to compact training-only
        # ids; every token not seen in training maps to <UNK> = 0.
        self.register_buffer("lex_remap", torch.zeros(1, dtype=torch.long),
                             persistent=False)
        if mode in ("lexical", "lex_kind"):
            self.emb = nn.Embedding(vocab_size, embed_dim)
        if mode in ("anon", "anon_kind"):
            self.anon_emb = nn.Embedding(anon_size, embed_dim)
        if mode == "op":
            self.emb = nn.Embedding(num_op_types, embed_dim)
        if mode == "kind":
            self.emb = nn.Embedding(num_kinds, embed_dim)
        if mode in ("op_kind",):
            self.op_emb = nn.Embedding(num_op_types, embed_dim)
        if mode in ("op_kind", "lex_kind", "anon_kind"):
            self.kind_emb = nn.Embedding(num_kinds, embed_dim)
        if mode == "kind_low":
            self.emb = nn.Embedding(num_kinds, num_op_types)
            self.proj = nn.Linear(num_op_types, embed_dim, bias=False)
        if mode == "op_wide":
            width = max(embed_dim, round(num_kinds * embed_dim / num_op_types))
            self.emb = nn.Embedding(num_op_types, width)
            self.proj = nn.Linear(width, embed_dim, bias=False)

    def set_lex_remap(self, remap: torch.Tensor):
        self.lex_remap = remap.to(self.lex_remap.device)

    def forward(self, data):
        m = self.mode
        if m == "lexical":
            return self.emb(self.lex_remap[data.x_lex])
        if m == "lex_kind":
            return (self.emb(self.lex_remap[data.x_lex])
                    + self.kind_emb(data.x_kind))
        if m == "anon":
            return self.anon_emb(data.x_anon)
        if m == "anon_kind":
            return self.anon_emb(data.x_anon) + self.kind_emb(data.x_kind)
        if m == "op":
            return self.emb(data.x_morph)
        if m == "kind":
            return self.emb(data.x_kind)
        if m == "kind_low":
            return self.proj(self.emb(data.x_kind))
        if m == "op_wide":
            return self.proj(self.emb(data.x_morph))
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
                 num_layers: int = 2, dropout: float = 0.3,
                 edge_set: str = "all"):
        super().__init__()
        assert backbone in BACKBONES, backbone
        self.feature_mode = feature_mode
        self.backbone_name = backbone
        self.num_layers = num_layers
        self.edge_types = EDGE_SETS[edge_set]

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
        elif backbone == "tf":
            # Not message passing: a Transformer encoder over the pre-order
            # node sequence (edges unused), with learned position embeddings.
            self.pos = nn.Embedding(512, hidden_dim)
            self.tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_dim, 4, 2 * hidden_dim,
                                           dropout=0.1, batch_first=True),
                num_layers=num_layers, enable_nested_tensor=False)
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
        ei = data.edge_index
        if self.edge_types != EDGE_SETS["all"] and hasattr(data, "edge_type"):
            keep = torch.zeros_like(data.edge_type, dtype=torch.bool)
            for t in self.edge_types:
                keep |= data.edge_type == t
            ei = ei[:, keep]
        if self.backbone_name == "ggnn":
            h = self.backbone(x, ei)
        elif self.backbone_name == "tf":
            dense, mask = to_dense_batch(x, data.batch)
            pos = torch.arange(dense.size(1), device=x.device).clamp(max=511)
            dense = dense + self.pos(pos)[None]
            dense = self.tf(dense, src_key_padding_mask=~mask)
            h = dense[mask]
        else:
            h = x
            for layer in self.layers:
                h = F.relu(layer(h, ei))
        h_mean = global_mean_pool(h, data.batch)
        h_max = global_max_pool(h, data.batch)
        logits = self.head(torch.cat([h_mean, h_max], dim=-1))
        # Signature matches the rest of the codebase: (logits, graph_emb, mask)
        return logits, h_mean, None
