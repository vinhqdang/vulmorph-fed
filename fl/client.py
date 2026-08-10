import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader as PyGDataLoader
from typing import Optional

from models.vulmorph import VulMorph
from models.vcsa import structural_contrastive_loss
from utils.privacy import add_calibrated_laplace_noise, clip_l1
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
        # Prototype bank has one slot per CWE bucket PLUS one benign slot;
        # every record (benign or vulnerable) contributes to exactly one row,
        # so the parallel-composition DP argument covers the whole bank.
        self.bank_slots = num_cwes + 1
        self.benign_slot = num_cwes
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

        # Class-weighted BCE: pos_weight = (#benign / #vulnerable) computed on
        # the local training split. Prevents the degenerate all-negative
        # classifier under class imbalance.
        n_pos = sum(1 for i in range(len(dataset)) if float(dataset[i].y[0]) == 1.0)
        n_neg = len(dataset) - n_pos
        pos_weight = torch.tensor(
            [n_neg / max(1, n_pos)], device=self.device
        ).clamp(max=20.0)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
        mu: float = 0.5,
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

                # Prototype-alignment regulariser (FedProto-style): pull each
                # sample embedding toward its own class prototype in the
                # GLOBAL bank, aligning client embedding spaces across rounds.
                loss_proto = torch.tensor(0.0, device=self.device)
                if global_prototypes is not None and mu > 0:
                    slots = torch.where(
                        batch.y.long() == 1,
                        batch.cwe.clamp(min=0, max=self.num_cwes - 1),
                        torch.full_like(batch.cwe, self.benign_slot),
                    )
                    targets = global_prototypes[slots]              # (B, d)
                    live = targets.norm(dim=1) > 1e-6               # skip empty rows
                    if live.any():
                        loss_proto = ((graph_emb[live] - targets[live]) ** 2
                                      ).sum(dim=1).mean()

                loss = (loss_bce + alpha * loss_scl + gamma * loss_l1
                        + mu * loss_proto)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                steps += 1

        return total_loss / max(steps, 1)

    # ------------------------------------------------------------------
    # Prototype construction
    # ------------------------------------------------------------------

    def compute_local_prototypes(self, clip_radius: float = 1.0):
        """
        Compute CWE-conditioned local prototypes p_{c,k} from vulnerable
        samples. Each per-sample embedding is first projected onto the L1
        ball of radius R = clip_radius, which bounds the sensitivity of the
        prototype mean at 2R / N_{c,k} (see utils/privacy.py).

        Returns:
            (prototypes, counts): (num_cwes, hidden_dim) tensor whose zero
            rows mean "no local data for that CWE", and the per-CWE sample
            counts (num_cwes,).
        """
        self.model.eval()

        proto_sums = torch.zeros(
            (self.bank_slots, self.hidden_dim), device=self.device
        )
        proto_counts = torch.zeros(self.bank_slots, device=self.device)

        with torch.no_grad():
            for batch in self.train_loader:
                batch = batch.to(self.device)
                _, graph_emb, _ = self.model(batch, prototypes=None)
                graph_emb = clip_l1(graph_emb.detach(), clip_radius)

                for i in range(batch.num_graphs):
                    if batch.y[i] == 1:
                        c = batch.cwe[i].item()
                        if not (0 <= c < self.num_cwes):
                            continue
                    else:
                        c = self.benign_slot
                    proto_sums[c] += graph_emb[i]
                    proto_counts[c] += 1

        protos = torch.zeros_like(proto_sums)
        for c in range(self.bank_slots):
            if proto_counts[c] > 0:
                protos[c] = proto_sums[c] / proto_counts[c]

        return protos, proto_counts

    def get_noisy_prototypes(
        self, epsilon: float, delta_f: float = 1.0
    ) -> torch.Tensor:
        """
        Build local prototypes and apply the calibrated Laplace mechanism.

        Args:
            epsilon: Per-round privacy budget ε (float('inf') disables DP).
            delta_f: L1 clipping radius R applied to per-sample embeddings
                     (the per-class noise scale is 2R / (N_{c,k} · ε)).
        """
        protos, counts = self.compute_local_prototypes(clip_radius=delta_f)
        if self.use_dp:
            protos = add_calibrated_laplace_noise(
                protos, counts, epsilon=epsilon, clip_radius=delta_f)
        return protos
