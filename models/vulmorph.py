import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATConv

from data.morphology import MorphologyEmbedding
from .vcsa import VCSA
from .mgmp import MGMPLayer


class VulMorph(nn.Module):
    """
    VulMorph-Fed Local Client Model.

    Integrates the three VulMorph-Fed innovations into one network:
      1. MorphologyEmbedding — project-invariant node features
      2. VCSA               — differentiable vulnerability-critical subgraph extraction
      3. MGMPLayer          — prototype-injected message passing

    Ablation flags allow each component to be disabled independently,
    replicating all variants in Section 5.5 of the research plan.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_cwes: int = 5,
        use_vcsa: bool = True,
        use_mgmp: bool = True,
        use_morphology: bool = True,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.use_vcsa = use_vcsa
        self.use_mgmp = use_mgmp
        self.use_morphology = use_morphology
        self.num_layers = num_layers

        # ── Node embeddings ──────────────────────────────────────────────
        self.lexical_embedding = nn.Embedding(vocab_size, embed_dim)
        self.morph_embedding = MorphologyEmbedding(embed_dim)

        morph_dim = embed_dim if use_morphology else 0
        self.node_dim = embed_dim + morph_dim          # concat of lex + morph (or lex only)

        # ── Component 1: VCSA ────────────────────────────────────────────
        if use_vcsa:
            self.vcsa = VCSA(node_dim=self.node_dim, hidden_dim=hidden_dim)

        # ── Component 3: MGMP or GAT fallback ───────────────────────────
        dims = [self.node_dim] + [hidden_dim] * num_layers
        if use_mgmp:
            self.gnn_layers = nn.ModuleList([
                MGMPLayer(
                    in_channels=dims[i],
                    out_channels=dims[i + 1],
                    num_cwes=num_cwes,
                    morph_dim=morph_dim,
                )
                for i in range(num_layers)
            ])
        else:
            self.gnn_layers = nn.ModuleList([
                GATConv(dims[i], dims[i + 1])
                for i in range(num_layers)
            ])

        # ── Classifier ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    # ------------------------------------------------------------------
    def forward(self, data, prototypes: torch.Tensor = None):
        """
        Args:
            data:       PyG Batch containing x_lex, x_morph, edge_index, batch.
            prototypes: Global prototype bank P* (num_cwes, hidden_dim) or None.

        Returns:
            logits:           (B, 1)  raw scores for BCEWithLogits.
            graph_embeddings: (B, hidden_dim)  used for L_SCL & prototype construction.
            edge_mask:        (E,)  VCSA soft mask (or all-ones tensor when ablated).
        """
        # 1. Node features
        x_lex = self.lexical_embedding(data.x_lex)            # (N, embed_dim)

        if self.use_morphology:
            x_morph = self.morph_embedding(data.x_morph)      # (N, embed_dim)
            x = torch.cat([x_lex, x_morph], dim=-1)           # (N, 2·embed_dim)
        else:
            x_morph = None
            x = x_lex

        # 2. VCSA edge masking
        if self.use_vcsa:
            edge_mask = self.vcsa(x, data.edge_index)          # (E,)
        else:
            edge_mask = torch.ones(
                data.edge_index.size(1), dtype=x.dtype, device=x.device
            )

        # 3. Message passing
        h = x
        morph_input = (
            x_morph
            if (x_morph is not None and self.use_mgmp)
            else torch.empty((x.size(0), 0), device=x.device)
        )

        for layer in self.gnn_layers:
            if self.use_mgmp:
                h = layer(h, data.edge_index, edge_mask, prototypes, morph_input)
            else:
                h = F.relu(layer(h, data.edge_index))

        # 4. Graph pooling
        graph_embeddings = global_mean_pool(h, data.batch)    # (B, hidden_dim)

        # 5. Classify
        logits = self.classifier(graph_embeddings)             # (B, 1)

        return logits, graph_embeddings, edge_mask
