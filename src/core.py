"""
core.py — Forward diffusion and reverse sampling for the cDDPM model.

Training:  train_model(...)  → trains a UNet1D to predict noise at each diffusion step.
Inference: sample(...)       → runs the reverse denoising chain to generate PV profiles.
"""

import torch
import torch.nn as nn

from models.unet_v2 import UNet1D
from utils import get_noise_schedule, q_sample, generate_real_pv_ts, reverse_step


def train_model(
    pv_train: torch.Tensor,
    conditions_train: dict[str, torch.Tensor],
    diffusion_steps: int,
    in_ch: int,
    base_channels: int = 128,
    time_embed_dim: int = 32,
    n_epochs: int = 2000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    noise_schedule: str = "linear",
    unet_depth: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[UNet1D, list]:
    """
    Train the conditional diffusion model.

    Args:
        pv_train:          Normalised PV training tensor [N, T].
        conditions_train:  Dict of condition tensors, each [N, T].
        diffusion_steps:   Number of forward diffusion steps K.
        in_ch:             Number of input channels (1 PV + n_conditions).
        base_channels:     Base channel width of the UNet.
        time_embed_dim:    Sinusoidal timestep embedding dimension.
        n_epochs:          Training epochs.
        batch_size:        Mini-batch size.
        learning_rate:     Adam learning rate.
        noise_schedule:    'linear' or 'cosine'.
        unet_depth:        Encoder/decoder levels (ts_length must be divisible by 2**depth).
        device:            Compute device.

    Returns:
        model:   Trained UNet1D.
        losses:  Per-epoch training loss values.
    """
    required_divisor = 2 ** unet_depth
    ts_length = pv_train.shape[1]
    if ts_length % required_divisor != 0:
        raise ValueError(
            f"ts_length={ts_length} must be divisible by 2^{unet_depth}={required_divisor}"
        )
    for key, cond in conditions_train.items():
        if cond.shape != pv_train.shape:
            raise ValueError(
                f"Condition '{key}' has shape {cond.shape}, expected {pv_train.shape}"
            )

    model = UNet1D(
        in_ch=in_ch,
        base_channels=base_channels,
        time_embed_dim=time_embed_dim,
        diffusion_steps=diffusion_steps,
        depth=unet_depth,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    schedule = get_noise_schedule(diffusion_steps, noise_schedule)
    alphas_cumprod = torch.tensor(
        schedule["alphas_cumprod"], dtype=torch.float32, device=device
    )

    losses = []
    for epoch in range(n_epochs):
        x_0, *conditions = generate_real_pv_ts(
            pv_train, *conditions_train.values(), n_samples=batch_size
        )
        x_0 = x_0.to(device)

        t = torch.randint(0, diffusion_steps, (batch_size,))
        noise = torch.randn_like(x_0).to(device)
        x_t = q_sample(x_0, t, noise, alphas_cumprod)

        x_t, t, noise = x_t.to(device), t.to(device), noise.to(device)
        conditions = [c.to(device) for c in conditions]

        pred_noise = model(x_t, t, conditions)
        loss = loss_fn(pred_noise, noise)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

    return model, losses


def sample(
    model: UNet1D,
    conditions_dict: dict[str, torch.Tensor],
    n_samples: int = 50,
    ts_length: int = 48,
    diffusion_steps: int = 100,
    noise_schedule: str = "linear",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Run the reverse diffusion process to generate PV samples.

    Args:
        model:            Trained UNet1D.
        conditions_dict:  Dict of condition tensors for a single day, each [T].
        n_samples:        Number of PV profiles to generate.
        ts_length:        Time series length T (48 for 30-min data).
        diffusion_steps:  Must match the value used during training.
        noise_schedule:   Must match the value used during training.
        device:           Compute device.

    Returns:
        Tensor of shape [n_samples, T] with generated (clamped >= 0) PV profiles.
    """
    schedule = get_noise_schedule(diffusion_steps, noise_schedule)
    betas = torch.tensor(schedule["betas"], dtype=torch.float32, device=device)
    alphas = torch.tensor(schedule["alphas"], dtype=torch.float32, device=device)
    alphas_cumprod = torch.tensor(
        schedule["alphas_cumprod"], dtype=torch.float32, device=device
    )

    conditions = [
        c.unsqueeze(0).repeat(n_samples, 1).to(device)
        for c in conditions_dict.values()
    ]

    x_t = torch.randn((n_samples, ts_length), device=device)

    with torch.no_grad():
        for t in reversed(range(diffusion_steps)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
            pred_noise = model(x_t, t_batch, conditions)
            x_t = reverse_step(
                x_t, pred_noise, alphas[t], betas[t], alphas_cumprod[t], t
            )

    return torch.clamp(x_t, 0.0)
