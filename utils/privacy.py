"""
Differential-privacy mechanisms and accounting for VulMorph-Fed.

Two mechanisms are applied, and the manuscript composes both:

  1. DP-SGD during local encoder training (`DPSGDAccountant`, `dp_sgd_step`)
     — per-example gradient clipping plus Gaussian noise, so that the encoder
     theta_k is itself a differentially private function of the client's data.
     Without this, the prototype guarantee below is only conditional on the
     encoder and the end-to-end pipeline has no finite epsilon.

  2. The per-class Laplace mechanism on the released prototype bank
     (`add_calibrated_laplace_noise`).

`total_epsilon` reports the end-to-end budget over both mechanisms and all
rounds.
"""

import math

import numpy as np
import torch


# ── Composition / accounting ─────────────────────────────────────────────────

def composed_epsilon(epsilon_per_round: float, rounds: int,
                     rows_touched: int = 2) -> float:
    """
    End-to-end budget of the prototype releases under sequential composition.

    Each round releases one Laplace-perturbed bank per client. Under the
    replacement neighbouring relation a single record change can affect two
    rows of the bank (it leaves one slot and joins another), so parallel
    composition across slots does not apply and each round costs
    `rows_touched` * epsilon. Over T rounds the total is T * rows_touched * eps.
    """
    if epsilon_per_round == float('inf'):
        return float('inf')
    return epsilon_per_round * rounds * rows_touched


_RDP_ORDERS = ([1 + x / 10.0 for x in range(1, 100)]
               + list(range(12, 64)) + [128, 256, 512])


def _log_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """
    RDP of the subsampled Gaussian mechanism at integer order alpha
    (Mironov et al., 2019, "Renyi Differential Privacy of the Sampled
    Gaussian Mechanism"), via the binomial expansion

        A_alpha = sum_{i=0}^{alpha} C(alpha,i) (1-q)^{alpha-i} q^i
                  exp( (i^2 - i) / (2 sigma^2) ),

    evaluated in log space for stability; RDP(alpha) = log(A_alpha)/(alpha-1).
    """
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * sigma ** 2)

    terms = []
    for i in range(alpha + 1):
        t = (_log_comb(alpha, i)
             + i * math.log(q)
             + (alpha - i) * math.log1p(-q)
             + (i * i - i) / (2.0 * sigma ** 2))
        terms.append(t)
    m = max(terms)
    log_a = m + math.log(sum(math.exp(t - m) for t in terms))
    return log_a / (alpha - 1)


def gaussian_epsilon(noise_multiplier: float, steps: int, sample_rate: float,
                     delta: float = 1e-5) -> float:
    """
    Privacy cost of `steps` applications of the subsampled Gaussian mechanism,
    accounted with Renyi DP (Abadi et al., CCS 2016; Mironov et al., 2019).

    RDP composes additively, so `steps` applications cost steps * RDP(alpha) at
    every order; we then convert to (eps, delta)-DP with the standard bound

        eps = rdp + log((alpha-1)/alpha) - (log(delta) + log(alpha))/(alpha-1)

    minimised over a grid of orders. This is the accountant used by TF-Privacy
    and Opacus, and is far tighter than advanced composition — using the
    latter would overstate our privacy cost by more than an order of magnitude.
    """
    if noise_multiplier <= 0:
        return float('inf')
    if steps <= 0:
        return 0.0
    q = min(max(sample_rate, 0.0), 1.0)

    best = float('inf')
    for alpha in _RDP_ORDERS:
        a_int = int(math.ceil(alpha))
        if a_int <= 1:
            continue
        rdp = steps * _rdp_subsampled_gaussian(q, noise_multiplier, a_int)
        eps = (rdp + math.log((a_int - 1) / a_int)
               - (math.log(delta) + math.log(a_int)) / (a_int - 1))
        best = min(best, eps)
    return max(best, 0.0)


def total_epsilon(proto_epsilon: float, rounds: int,
                  sgd_epsilon: float = 0.0, rows_touched: int = 2) -> float:
    """End-to-end budget: DP-SGD training + all prototype releases."""
    proto = composed_epsilon(proto_epsilon, rounds, rows_touched)
    if proto == float('inf') or sgd_epsilon == float('inf'):
        return float('inf')
    return proto + sgd_epsilon


# ── Clipping ─────────────────────────────────────────────────────────────────

def clip_l1(x: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Project each row of x onto the L1 ball of the given radius:
    x_i <- x_i * min(1, radius / ||x_i||_1).

    Clipping the per-sample embedding norm bounds the sensitivity of the
    prototype mean, so the Laplace mechanism below yields a real epsilon-DP
    guarantee rather than a heuristic one.
    """
    norms = x.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (radius / norms).clamp(max=1.0)
    return x * scale


# ── DP-SGD for local encoder training ────────────────────────────────────────

def dp_sgd_step(model, per_example_losses: torch.Tensor,
                optimizer, max_grad_norm: float = 1.0,
                noise_multiplier: float = 1.0) -> None:
    """
    One DP-SGD update (Abadi et al., CCS 2016) using microbatching.

    Each example's gradient is computed separately, clipped to `max_grad_norm`
    in L2, summed, perturbed with Gaussian noise of scale
    `noise_multiplier * max_grad_norm`, and averaged over the batch. This makes
    the resulting parameters a differentially private function of the batch.

    Microbatching (one backward pass per example) is slower than vectorised
    per-sample gradients, but it is exact, backend-agnostic, and adequate at
    the batch sizes used here.

    Args:
        model:               module whose parameters are being trained
        per_example_losses:  (B,) tensor of unreduced per-example losses
        optimizer:           optimiser stepping `model`'s parameters
        max_grad_norm:       L2 clipping bound C
        noise_multiplier:    sigma; noise std is sigma * C
    """
    params = [p for p in model.parameters() if p.requires_grad]
    accum = [torch.zeros_like(p) for p in params]
    B = per_example_losses.size(0)
    if B == 0:
        return

    for i in range(B):
        model.zero_grad(set_to_none=True)
        per_example_losses[i].backward(retain_graph=(i < B - 1))
        total_sq = 0.0
        for p in params:
            if p.grad is not None:
                total_sq += float(p.grad.detach().pow(2).sum())
        coef = min(1.0, max_grad_norm / (math.sqrt(total_sq) + 1e-12))
        for a, p in zip(accum, params):
            if p.grad is not None:
                a.add_(p.grad.detach() * coef)

    model.zero_grad(set_to_none=True)
    std = noise_multiplier * max_grad_norm
    for a, p in zip(accum, params):
        if std > 0:
            a.add_(torch.normal(0.0, std, size=a.shape, device=a.device,
                                dtype=a.dtype))
        p.grad = a / B
    optimizer.step()


class DPSGDAccountant:
    """Tracks DP-SGD steps so the training budget can be reported."""

    def __init__(self, noise_multiplier: float, sample_rate: float,
                 delta: float = 1e-5):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self.steps = 0

    def step(self, n: int = 1) -> None:
        self.steps += n

    def epsilon(self) -> float:
        return gaussian_epsilon(self.noise_multiplier, self.steps,
                                self.sample_rate, self.delta)


# ── Laplace mechanism on the prototype bank ──────────────────────────────────

def add_calibrated_laplace_noise(
    prototypes: torch.Tensor,
    counts: torch.Tensor,
    epsilon: float,
    clip_radius: float,
) -> torch.Tensor:
    """
    Per-class Laplace mechanism for the class-conditioned prototype bank.

    Each prototype p_c is the mean of N_c per-sample embeddings whose L1 norm
    has been clipped to `clip_radius` R, so replacing one record changes p_c by
    at most Delta_1 = 2R / N_c in L1 norm and the row release is epsilon-DP.
    A record change can move a record between two slots, so the bank release
    costs 2*epsilon per round (see `composed_epsilon`).

    Args:
        prototypes:  (slots, hidden_dim) prototype bank.
        counts:      (slots,) number of samples contributing to each row.
        epsilon:     Per-round privacy budget (inf disables noise).
        clip_radius: L1 clipping bound R applied to per-sample embeddings.
    """
    if epsilon == float('inf') or epsilon <= 0:
        return prototypes

    noisy = prototypes.clone()
    for c in range(prototypes.size(0)):
        n_c = float(counts[c])
        if n_c <= 0:
            continue    # empty rows are all-zero and carry no information
        b = (2.0 * clip_radius / n_c) / epsilon
        noise = np.random.laplace(loc=0.0, scale=b, size=prototypes.size(1))
        noisy[c] = noisy[c] + torch.tensor(
            noise, dtype=prototypes.dtype, device=prototypes.device)
    return noisy


def add_laplace_noise(prototypes: torch.Tensor, epsilon: float,
                      delta_f: float) -> torch.Tensor:
    """Uncalibrated Laplace mechanism, retained for the w/o-calibration ablation."""
    if epsilon == float('inf') or epsilon <= 0:
        return prototypes
    b = delta_f / epsilon
    noise = np.random.laplace(loc=0.0, scale=b, size=prototypes.shape)
    return prototypes + torch.tensor(noise, dtype=prototypes.dtype,
                                     device=prototypes.device)
