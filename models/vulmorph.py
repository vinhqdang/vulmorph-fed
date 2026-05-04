import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from data.morphology import MorphologyEmbedding, NUM_MORPHOLOGY_TYPES
from .vcsa import VCSA
from .mgmp import MGMPLayer

class VulMorph(nn.Module):
    """
    VulMorph-Fed Local Client Model.
    Combines Morphology Embedding, VCSA, MGMP, and a binary classifier.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_cwes: int = 5):
        super().__init__()
        
        # We start with lexical token embeddings, though the framework relies heavily on morphology
        self.lexical_embedding = nn.Embedding(vocab_size, embed_dim)
        self.morph_embedding = MorphologyEmbedding(embed_dim)
        
        # Node features are sum/concat of lexical and morphological. 
        # For true structural transfer, we could drop lexical entirely, but we keep it for full representation.
        self.node_dim = embed_dim * 2
        
        # Component 1: Vulnerability-Critical Subgraph Abstraction
        self.vcsa = VCSA(node_dim=self.node_dim, hidden_dim=hidden_dim)
        
        # Component 3: Morphology-Guided Message Passing
        # We use two MGMP layers
        self.mgmp1 = MGMPLayer(in_channels=self.node_dim, out_channels=hidden_dim, num_cwes=num_cwes, morph_dim=embed_dim)
        self.mgmp2 = MGMPLayer(in_channels=hidden_dim, out_channels=hidden_dim, num_cwes=num_cwes, morph_dim=embed_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1) # Binary classification (vuln or not)
        )

    def forward(self, data, prototypes: torch.Tensor = None):
        """
        Args:
            data: PyG Data object containing x_lex, x_morph, edge_index, batch
            prototypes: Global prototype bank P* of shape (num_cwes, hidden_dim)
        Returns:
            logits: Binary classification logits
            graph_embeddings: Used for L_SCL and prototype construction
            edge_mask: Soft edge masks from VCSA
        """
        x_lex = self.lexical_embedding(data.x_lex)
        x_morph = self.morph_embedding(data.x_morph)
        
        x = torch.cat([x_lex, x_morph], dim=-1)
        
        # 1. VCSA Edge Masking
        # Get soft mask for edges
        edge_mask = self.vcsa(x, data.edge_index)
        
        # Optional: apply threshold tau here to get G*, but we use soft mask as edge_weight for differentiability
        
        # 2. MGMP Layers
        h = self.mgmp1(x, data.edge_index, edge_mask, prototypes, x_morph)
        h = self.mgmp2(h, data.edge_index, edge_mask, prototypes, x_morph)
        
        # 3. Graph Pooling
        # Global mean pooling over nodes to get graph-level embeddings
        graph_embeddings = global_mean_pool(h, data.batch)
        
        # 4. Classification
        logits = self.classifier(graph_embeddings)
        
        return logits, graph_embeddings, edge_mask
