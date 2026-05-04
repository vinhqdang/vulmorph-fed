import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Optional

from models.vulmorph import VulMorph
from models.vcsa import structural_contrastive_loss
from utils.privacy import add_laplace_noise
from data.dataset import StructuredCPGDataset as MockCPGDataset


class VulMorphClient:
    """
    VulMorph-Fed Federated Client.

    Responsibilities (per round):
      1. Train local model with L_total = L_BCE + α·L_SCL + γ·||ε||_1
      2. Construct CWE-conditioned local prototypes p_{c,k}
      3. Apply Laplace differential privacy → upload p̃_{c,k}
      4. Receive updated global prototype bank P* and update MGMP

    Reference: Section 3.5, Client Phase steps 1–5 and 9–10.
    """

    def __init__(
        self,
        client_id: int,
        dataset: MockCPGDataset,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_cwes: int,
        device: str = 'cpu',
        batch_size: int = 32,
        lr: float = 1e-3,
        use_dp: bool = True,
        **model_kwargs,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.num_cwes = num_cwes
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.use_dp = use_dp

        self.model = VulMorph(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_cwes=num_cwes,
            **model_kwargs,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.train_loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def train_local(
        self,
        global_prototypes: Optional[torch.Tensor],
        epochs: int = 1,
        alpha: float = 0.1,
        gamma: float = 0.01,
    ) -> float:
        """
        Train the local model for `epochs` passes over the local dataset.

        Args:
            global_prototypes: P* from the server (num_cwes, hidden_dim) or None.
            epochs:  Number of local epochs E_local.
            alpha:   Weight for L_SCL.
            gamma:   Weight for L1 edge-sparsity loss.
        Returns:
            Average total loss over all mini-batches.
        """
        self.model.train()

        if global_prototypes is not None:
            global_prototypes = global_prototypes.to(self.device)

        total_loss = 0.0
        steps = 0

        for _ in range(epochs):
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                logits, graph_emb, edge_mask = self.model(
                    batch, prototypes=global_prototypes
                )

                # Primary detection loss
                loss_bce = self.bce_loss(logits.squeeze(-1), batch.y)

                # Structural contrastive loss for cross-project alignment
                loss_scl = structural_contrastive_loss(
                    graph_emb, batch.y, batch.cwe
                )

                # L1 sparsity on edge masks
                if edge_mask is not None and edge_mask.numel() > 0:
                    loss_l1 = edge_mask.sum() / max(edge_mask.numel(), 1)
                else:
                    loss_l1 = torch.tensor(0.0, device=self.device)

                loss = loss_bce + alpha * loss_scl + gamma * loss_l1
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

        return total_loss / max(steps, 1)

    # ------------------------------------------------------------------
    # Prototype construction
    # ------------------------------------------------------------------

    def compute_local_prototypes(self) -> torch.Tensor:
        """
        Compute CWE-conditioned local prototypes p_{c,k} from vulnerable samples.

        Returns:
            Tensor (num_cwes, hidden_dim).  Zero rows = no local data for that CWE.
        """
        self.model.eval()

        proto_sums = torch.zeros(
            (self.num_cwes, self.hidden_dim), device=self.device
        )
        proto_counts = torch.zeros(self.num_cwes, device=self.device)

        with torch.no_grad():
            for batch in self.train_loader:
                batch = batch.to(self.device)
                _, graph_emb, _ = self.model(batch, prototypes=None)

                for i in range(batch.num_graphs):
                    if batch.y[i] == 1:
                        c = batch.cwe[i].item()
                        if 0 <= c < self.num_cwes:
                            proto_sums[c] += graph_emb[i].detach()
                            proto_counts[c] += 1

        protos = torch.zeros_like(proto_sums)
        for c in range(self.num_cwes):
            if proto_counts[c] > 0:
                protos[c] = proto_sums[c] / proto_counts[c]

        return protos

    def get_noisy_prototypes(
        self, epsilon: float, delta_f: float
    ) -> torch.Tensor:
        """
        Build local prototypes and optionally apply Laplace DP.

        Args:
            epsilon: Privacy budget ε (float('inf') disables DP).
            delta_f: Global L2-sensitivity Δf of the prototype function.
        """
        protos = self.compute_local_prototypes()
        if self.use_dp:
            protos = add_laplace_noise(protos, epsilon=epsilon, delta_f=delta_f)
        return protos
