import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import copy
from typing import Dict, Optional

from models.vulmorph import VulMorph
from models.vcsa import structural_contrastive_loss
from utils.privacy import add_laplace_noise
from data.dataset import MockCPGDataset

class VulMorphClient:
    def __init__(self, client_id: int, dataset: MockCPGDataset, 
                 vocab_size: int, embed_dim: int, hidden_dim: int, num_cwes: int,
                 device: str = 'cpu'):
        self.client_id = client_id
        self.dataset = dataset
        self.num_cwes = num_cwes
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        
        # Initialize local model
        self.model = VulMorph(vocab_size=vocab_size, embed_dim=embed_dim, 
                              hidden_dim=hidden_dim, num_cwes=num_cwes).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # DataLoader
        self.train_loader = PyGDataLoader(self.dataset, batch_size=32, shuffle=True)

    def train_local(self, global_prototypes: Optional[torch.Tensor], epochs: int = 1, 
                    alpha: float = 0.1, gamma: float = 0.01):
        """
        Train the local model for a number of epochs.
        Args:
            global_prototypes: The P* prototype bank from the server (num_cwes, hidden_dim)
            epochs: Number of local training epochs E_local
            alpha: Weight for Structural Contrastive Loss (L_SCL)
            gamma: Weight for L1 sparsity loss on edge masks
        """
        self.model.train()
        
        if global_prototypes is not None:
            global_prototypes = global_prototypes.to(self.device)
            
        epoch_loss = 0.0
        
        for epoch in range(epochs):
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                logits, graph_embeddings, edge_mask = self.model(batch, prototypes=global_prototypes)
                
                # BCE Loss for vulnerability detection
                loss_bce = self.bce_loss(logits.squeeze(-1), batch.y)
                
                # Structural Contrastive Loss for VCSA
                loss_scl = structural_contrastive_loss(graph_embeddings, batch.y, batch.cwe)
                
                # L1 Sparsity Loss for edge masks
                loss_l1 = torch.norm(edge_mask, p=1) / batch.num_edges
                
                # Total loss
                loss_total = loss_bce + alpha * loss_scl + gamma * loss_l1
                
                loss_total.backward()
                self.optimizer.step()
                
                epoch_loss += loss_total.item()
                
        return epoch_loss / max(1, len(self.train_loader) * epochs)

    def compute_local_prototypes(self) -> torch.Tensor:
        """
        Compute CWE-conditioned local prototypes p_{c,k}
        Returns:
            prototypes: Tensor of shape (num_cwes, hidden_dim)
        """
        self.model.eval()
        
        # We need a sum and a count for each CWE
        proto_sums = torch.zeros((self.num_cwes, self.hidden_dim), device=self.device)
        proto_counts = torch.zeros(self.num_cwes, device=self.device)
        
        with torch.no_grad():
            for batch in self.train_loader:
                batch = batch.to(self.device)
                # Forward pass (without global prototypes) just to get embeddings
                _, graph_embeddings, _ = self.model(batch, prototypes=None)
                
                # For each sample in the batch
                for i in range(batch.num_graphs):
                    if batch.y[i] == 1: # Only construct prototypes from vulnerable samples
                        cwe_idx = batch.cwe[i].item()
                        if cwe_idx >= 0 and cwe_idx < self.num_cwes:
                            proto_sums[cwe_idx] += graph_embeddings[i]
                            proto_counts[cwe_idx] += 1
                            
        # Average
        prototypes = torch.zeros((self.num_cwes, self.hidden_dim), device=self.device)
        for c in range(self.num_cwes):
            if proto_counts[c] > 0:
                prototypes[c] = proto_sums[c] / proto_counts[c]
                
        return prototypes

    def get_noisy_prototypes(self, epsilon: float, delta_f: float) -> torch.Tensor:
        """
        Construct local prototypes and apply Laplace DP.
        """
        clean_prototypes = self.compute_local_prototypes()
        noisy_prototypes = add_laplace_noise(clean_prototypes, epsilon=epsilon, delta_f=delta_f)
        return noisy_prototypes
