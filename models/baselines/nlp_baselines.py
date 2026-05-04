import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class CodeBERTSimple(nn.Module):
    """
    Simplified NLP baseline (simulating LineVul/VulFL-NLP).
    Treats code as a sequence of tokens, ignoring graph structure.
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Simple LSTM to simulate sequential processing of CodeBERT
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, data, **kwargs):
        # data.x_lex has shape (total_nodes_in_batch,)
        # We need to process each graph's tokens as a sequence.
        # For simplicity in this PyG setup, we just embed and pool.
        # A true NLP baseline wouldn't use PyG batches.
        
        x = self.embedding(data.x_lex)
        
        # Ignore edges entirely
        
        # Pool to graph level (simulating [CLS] token pooling)
        h_pool = global_mean_pool(x, data.batch)
        
        # Pass through a dummy layer to match capacity
        h_proj = nn.functional.relu(nn.Linear(x.size(-1), self.lstm.hidden_size * 2, device=x.device)(h_pool))
        
        logits = self.classifier(h_proj)
        return logits, h_proj, None
