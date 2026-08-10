import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class TransformerSeqBaseline(nn.Module):
    """
    Sequence-model baseline: a small Transformer encoder over the raw lexical
    token sequence (graph structure entirely ignored), with mean pooling and
    a linear classification head.

    This is the structure-free counterpart to the GNN baselines: it represents
    the token-sequence family (LineVul / VulFL-NLP style) at a scale trainable
    on the same hardware and data budget as the graph models. It is *not* a
    pre-trained code LM; we state this explicitly in the manuscript.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 2, num_heads: int = 4, max_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2, dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.max_len = max_len
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data, **kwargs):
        # (B, L) dense token sequences with padding mask
        x, mask = to_dense_batch(data.x_lex, data.batch, max_num_nodes=self.max_len)
        h = self.embedding(x)                                   # (B, L, d)
        pos = torch.arange(x.size(1), device=x.device)
        h = h + self.pos_embedding(pos).unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=~mask)         # (B, L, d)

        # Masked mean pooling
        mask_f = mask.unsqueeze(-1).float()
        h_pool = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)   # (B, d)

        logits = self.classifier(h_pool)
        return logits, h_pool, None
