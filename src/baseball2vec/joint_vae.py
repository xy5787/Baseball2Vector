"""Joint Encoder VAE with Cholesky full covariance for Baseball2Vec."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .tools import ALIGN_ANCHORS, TOOL_NAMES


class JointVAE(nn.Module):
    """Joint encoder → 5D latent (Cholesky full covariance) → group-specific decoders.

    The encoder sees all raw stats jointly so that cross-tool interactions are
    captured. The full covariance Σ = LL^T encodes tool tradeoffs in its
    off-diagonal entries.

    Args:
        input_dim: Total number of input features (all tools concatenated).
        latent_dim: Latent space dimension (5 = number of tools).
        group_dims: Dict mapping tool name → number of features in that group.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        group_dims: dict[str, int],
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.group_names = list(group_dims.keys())

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
        )
        self.fc_mu = nn.Linear(32, latent_dim)

        # Cholesky factor: lower-triangular entries of the 5×5 matrix (15 params)
        self.n_tril = latent_dim * (latent_dim + 1) // 2
        self.fc_L = nn.Linear(32, self.n_tril)
        self.tril_indices = torch.tril_indices(latent_dim, latent_dim)

        self.decoders = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.LeakyReLU(0.1),
                    nn.Linear(32, 16),
                    nn.LeakyReLU(0.1),
                    nn.Linear(16, dim),
                )
                for name, dim in group_dims.items()
            }
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode *x* into (μ, L) where Σ = LL^T.

        Args:
            x: Input tensor of shape [batch, input_dim].

        Returns:
            Tuple of (μ [batch, latent_dim], L [batch, latent_dim, latent_dim]).
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)

        L_flat = self.fc_L(h)
        batch_size = x.size(0)
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=x.device)
        L[:, self.tril_indices[0], self.tril_indices[1]] = L_flat

        # Ensure positive diagonal (positive-definite covariance)
        diag = torch.arange(self.latent_dim)
        L[:, diag, diag] = torch.exp(L[:, diag, diag]) + 1e-6

        return mu, L

    def reparameterize(self, mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Sample z = μ + L @ ε, ε ~ N(0, I).

        Args:
            mu: Mean tensor [batch, latent_dim].
            L: Cholesky factor [batch, latent_dim, latent_dim].

        Returns:
            Sampled latent [batch, latent_dim].
        """
        eps = torch.randn_like(mu)
        return mu + torch.bmm(L, eps.unsqueeze(-1)).squeeze(-1)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reconstruct each tool group from latent *z*.

        Args:
            z: Latent tensor [batch, latent_dim].

        Returns:
            Dict mapping tool name → reconstructed features.
        """
        return {name: self.decoders[name](z) for name in self.group_names}

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, L = self.encode(x)
        z = self.reparameterize(mu, L)
        return self.decode(z), mu, L


def kl_divergence_full_cov(mu: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """KL(N(μ, LL^T) || N(0, I)), summed over the batch.

    KL = 0.5 * [tr(LL^T) + μ^T μ - k - 2 Σ_i log(L_ii)]

    Args:
        mu: Mean [batch, k].
        L: Cholesky factor [batch, k, k].

    Returns:
        Scalar KL divergence summed over the batch.
    """
    k = mu.size(1)
    trace_term = (L ** 2).sum(dim=(1, 2))
    mu_sq = (mu ** 2).sum(dim=1)
    diag = torch.arange(k)
    log_det = 2.0 * torch.log(L[:, diag, diag]).sum(dim=1)
    return 0.5 * (trace_term + mu_sq - k - log_det).sum()


def alignment_loss(
    mu: torch.Tensor,
    X_aligned: pd.DataFrame,
    tool_names: list[str] = TOOL_NAMES,
    anchor_dict: dict[str, str] = ALIGN_ANCHORS,
) -> torch.Tensor:
    """Penalize misalignment between latent dims and their anchor statistics.

    Maximizes Pearson correlation between z_i and the anchor stat for tool i.

    Args:
        mu: Latent means [batch, 5].
        X_aligned: Direction-corrected Z-scored DataFrame (for anchor values).
        tool_names: Ordered tool names matching latent dimensions.
        anchor_dict: Maps tool name → anchor column name.

    Returns:
        Scalar alignment loss (sum of (1 - r) over tools).
    """
    loss = torch.tensor(0.0)
    for i, tool in enumerate(tool_names):
        anchor = anchor_dict.get(tool)
        if anchor and anchor in X_aligned.columns:
            anchor_vals = torch.FloatTensor(X_aligned[anchor].values)
            z_i = mu[:, i]
            vx = z_i - z_i.mean()
            vy = anchor_vals - anchor_vals.mean()
            corr = torch.sum(vx * vy) / (
                torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
            )
            loss = loss + (1.0 - corr)
    return loss


def soft_disentangle_loss(mu: torch.Tensor, threshold: float = 0.7) -> torch.Tensor:
    """Penalize off-diagonal latent correlations that exceed *threshold*.

    Allows modest tool correlations (tradeoffs) but prevents dimension collapse.

    Args:
        mu: Latent means [batch, 5].
        threshold: Off-diagonal correlation magnitude above which penalty applies.

    Returns:
        Scalar disentanglement penalty.
    """
    mu_c = mu - mu.mean(dim=0, keepdim=True)
    cov = (mu_c.T @ mu_c) / (mu.size(0) - 1)
    std = torch.sqrt(torch.diag(cov) + 1e-8)
    corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
    mask = 1.0 - torch.eye(mu.size(1), device=mu.device)
    return torch.relu((corr * mask).abs() - threshold).mean()


def train(
    X_aligned: pd.DataFrame,
    all_features: list[str],
    final_groups: dict[str, list[str]],
    group_dims: dict[str, int],
    pa_series: pd.Series,
    tool_names: list[str] = TOOL_NAMES,
    n_epochs: int = 4000,
    lr: float = 0.003,
    kl_warmup: int = 1000,
    align_weight: float = 500.0,
    disent_weight: float = 50.0,
) -> tuple[JointVAE, np.ndarray, np.ndarray, np.ndarray]:
    """Train the JointVAE and return extracted tool scores and covariance.

    Args:
        X_aligned: Direction-corrected Z-scored DataFrame.
        all_features: All input feature column names (flattened across groups).
        final_groups: Tool → column mapping.
        group_dims: Tool → number of features.
        pa_series: PA values for loss weighting.
        tool_names: Ordered tool names.
        n_epochs: Training epochs.
        lr: Adam learning rate.
        kl_warmup: Epochs over which KL weight linearly ramps from 0 to 1.
        align_weight: Coefficient for alignment loss.
        disent_weight: Coefficient for soft disentanglement loss.

    Returns:
        Tuple of:
            - Trained JointVAE model (eval mode).
            - vae_mu: [N, 5] latent means (tool scores).
            - vae_sigma: [N, 5] per-tool uncertainty (diagonal of Σ).
            - mean_corr: [5, 5] league-average tool correlation matrix.
    """
    X_tensor = torch.FloatTensor(X_aligned[all_features].values)
    group_targets = {
        g: torch.FloatTensor(X_aligned[cols].values)
        for g, cols in final_groups.items()
    }

    pa_tensor = torch.FloatTensor(pa_series.values).unsqueeze(1)
    loss_weights = torch.log(pa_tensor + 1)
    loss_weights = loss_weights / loss_weights.mean()

    model = JointVAE(len(all_features), 5, group_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"  Training JointVAE ({n_epochs} epochs)...")
    for epoch in range(n_epochs):
        model.train()
        recon_groups, mu, L = model(X_tensor)

        recon_loss = sum(
            ((recon_groups[g] - group_targets[g]) ** 2)
            .mean(dim=1, keepdim=True)
            .mul(loss_weights)
            .sum()
            for g in tool_names
        )

        kl_weight = min(1.0, epoch / kl_warmup)
        kl = kl_divergence_full_cov(mu, L)
        align = alignment_loss(mu, X_aligned, tool_names)
        disent = soft_disentangle_loss(mu)

        total_loss = recon_loss + kl_weight * kl + align_weight * align + disent_weight * disent

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            n = len(pa_series)
            print(
                f"    Epoch {epoch+1}/{n_epochs} | "
                f"Recon: {recon_loss.item()/n:.4f} | "
                f"KL: {kl.item()/n:.4f} (w={kl_weight:.2f}) | "
                f"Align: {align.item():.4f} | "
                f"Disent: {disent.item():.4f}"
            )

    model.eval()
    with torch.no_grad():
        _, mu_final, L_final = model(X_tensor)
        vae_mu = mu_final.numpy()

        Sigma = torch.bmm(L_final, L_final.transpose(1, 2))
        vae_sigma = torch.sqrt(torch.diagonal(Sigma, dim1=1, dim2=2)).numpy()

        mean_Sigma = Sigma.mean(dim=0).numpy()
        mean_std = np.sqrt(np.diag(mean_Sigma))
        mean_corr = mean_Sigma / (np.outer(mean_std, mean_std) + 1e-8)

    return model, vae_mu, vae_sigma, mean_corr
