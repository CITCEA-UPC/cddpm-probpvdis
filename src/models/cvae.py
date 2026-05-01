"""
cvae.py — Conditional Variational Autoencoder (cVAE) benchmark model.

Used as a probabilistic baseline in the paper. The cVAE encodes the joint
distribution of (PV profile, conditions) into a latent space and decodes
new samples conditioned on the same inputs.

Architecture:
  Encoder: [pv, cond_flat] -> (mu, logvar) via MLP with LayerNorm
  Decoder: [z, cond_flat]  -> pv_hat via MLP with LayerNorm + Softplus output

Training uses a beta-annealed ELBO loss combining MSE reconstruction and
a ramp-rate penalty term for temporal smoothness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CVAE(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        pv_dim: int = 48,
        z_dim: int = 16,
        hidden: int = 256,
    ):
        """
        Args:
            cond_dim: Flattened condition dimension (n_conditions * ts_length).
            pv_dim:   Time series length T.
            z_dim:    Latent space dimensionality.
            hidden:   MLP hidden size.
        """
        super().__init__()
        self.pv_dim = pv_dim
        self.cond_dim = cond_dim
        self.z_dim = z_dim

        # Encoder: [pv || cond] -> (mu, logvar)
        self.enc = nn.Sequential(
            nn.Linear(pv_dim + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)

        # Decoder: [z || cond] -> pv_hat
        self.dec = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, pv_dim),
        )

    def encode(self, pv: torch.Tensor, cond: torch.Tensor) -> tuple:
        h = self.enc(torch.cat([pv, cond], dim=1))
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.dec(torch.cat([z, cond], dim=1)))

    def forward(self, pv: torch.Tensor, cond: torch.Tensor) -> tuple:
        mu, logvar = self.encode(pv, cond)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_features_flat(
    conditions_dict: dict[str, torch.Tensor],
    order: list[str],
) -> torch.Tensor:
    """
    Flatten and concatenate condition tensors into a single feature vector.

    Args:
        conditions_dict: {key: Tensor [B, T]}.
        order:           Keys in the desired concatenation order.

    Returns:
        Tensor [B, n_conditions * T].
    """
    return torch.cat([conditions_dict[k] for k in order], dim=1)


def _ramp_mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """First-order difference MSE for temporal smoothness regularisation."""
    return F.mse_loss(y_hat[:, 1:] - y_hat[:, :-1], y[:, 1:] - y[:, :-1])


def _cvae_loss(
    pv_hat: torch.Tensor,
    pv: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    alpha: float = 0.8,
) -> tuple:
    recon = alpha * F.mse_loss(pv_hat, pv) + (1 - alpha) * _ramp_mse(pv_hat, pv)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach(), kl.detach()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cvae(
    pv_train: torch.Tensor,
    cond_train_flat: torch.Tensor,
    pv_dim: int = 48,
    z_dim: int = 16,
    hidden: int = 256,
    n_epochs: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 5e-3,
    beta_max: float = 0.01,
    warmup_frac: float = 0.7,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CVAE:
    """
    Train the cVAE model.

    Args:
        pv_train:        Normalised PV training tensor [N, T].
        cond_train_flat: Flattened condition tensor [N, n_cond * T].
        pv_dim:          Time series length T.
        z_dim:           Latent dimensionality.
        hidden:          MLP hidden size.
        n_epochs:        Training epochs.
        batch_size:      Mini-batch size.
        learning_rate:   Adam learning rate.
        beta_max:        Maximum KL weight (reached after warmup).
        warmup_frac:     Fraction of epochs over which beta is annealed.
        device:          Compute device.

    Returns:
        Trained CVAE model.
    """
    model = CVAE(
        cond_dim=cond_train_flat.shape[1], pv_dim=pv_dim, z_dim=z_dim, hidden=hidden
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pv_train = pv_train.to(device)
    cond_train_flat = cond_train_flat.to(device)
    N = pv_train.shape[0]
    warmup_steps = int(n_epochs * warmup_frac)

    for ep in range(n_epochs):
        beta = beta_max * min(1.0, ep / max(1, warmup_steps))
        idx = torch.randint(0, N, (batch_size,), device=device)
        pv_hat, mu, logvar = model(pv_train[idx], cond_train_flat[idx])
        loss, recon, kl = _cvae_loss(pv_hat, pv_train[idx], mu, logvar, beta)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 50 == 0:
            print(
                f"Epoch {ep:4d} | Loss: {loss.item():.4f} "
                f"(recon={recon.item():.4f}, kl={kl.item():.4f}, beta={beta:.4f})"
            )

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_cvae(
    model: CVAE,
    cond_one_flat: torch.Tensor,
    n_samples: int = 50,
    z_temp: float = 2.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Generate PV samples by decoding from the prior.

    Args:
        model:         Trained CVAE.
        cond_one_flat: Flattened condition for a single day [n_cond * T].
        n_samples:     Number of profiles to generate.
        z_temp:        Temperature scaling for latent noise (> 1 = more diversity).
        device:        Compute device.

    Returns:
        Tensor [n_samples, T] on CPU.
    """
    model.eval()
    c = cond_one_flat.unsqueeze(0).repeat(n_samples, 1).to(device)
    z = z_temp * torch.randn(n_samples, model.z_dim, device=device)
    return model.decode(z, c).cpu()
