import torch
import torch.nn as nn
import numpy as np
# from typing import List

from models.unet_v2 import UNet1D
from utils import get_noise_schedule, q_sample, generate_real_pv_ts, reverse_step


def train_model(pv_train: torch.Tensor, conditions_train: dict[str, torch.Tensor], diffusion_steps: int, in_ch: int,
                base_channels: int = 64, time_embed_dim: int = 32, n_epochs: int = 2000, batch_size: int = 64,
                learning_rate: float = 1e-3, noise_schedule: str = 'linear', unet_depth: int = 2,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> UNet1D:

    # Check 1: time series length must be compatible with UNet depth
    required_divisor = 2 ** unet_depth
    ts_length = pv_train.shape[1]
    if ts_length % required_divisor != 0:
        raise ValueError(f"ts_length={ts_length} must be divisible by 2^{unet_depth} = {required_divisor}")

    # Check 2: all condition tensors must match pv_train shape
    for key, cond in conditions_train.items():
        if cond.shape != pv_train.shape:
            raise ValueError(f"Condition '{key}' has shape {cond.shape}, expected {pv_train.shape}")

    model = UNet1D(in_ch=in_ch, base_channels=base_channels, time_embed_dim=time_embed_dim,
                   diffusion_steps=diffusion_steps, depth=unet_depth).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    schedule = get_noise_schedule(diffusion_steps, noise_schedule)
    alphas_cumprod = torch.tensor(schedule['alphas_cumprod'], dtype=torch.float32, device=device)

    losses = []
    for epoch in range(n_epochs):
        x_0, *conditions = generate_real_pv_ts(pv_train, *conditions_train.values(),
                                               n_samples=batch_size)
        x_0 = x_0.to(device)

        t = torch.randint(0, diffusion_steps, (batch_size,))  # pick a random timestep for each sample
        noise = torch.randn_like(x_0).to(device)  # returns a tensor same size filled with random numbers N(0, 1)
        x_t = q_sample(x_0, t, noise, alphas_cumprod)

        # Ensure all tensors are on the same device
        x_t, t, noise = x_t.to(device), t.to(device), noise.to(device)
        conditions = [c.to(device) for c in conditions]

        pred_noise = model(x_t, t, conditions)
        loss = loss_fn(pred_noise, noise)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model, losses


def sample(model, conditions_dict: dict[str, torch.Tensor], n_samples: int = 1, ts_length: int = 48,
           diffusion_steps: int = 200, noise_schedule: str = 'linear',
           device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    """
    Run the reverse diffusion process to generate PV samples.
    """
    schedule = get_noise_schedule(diffusion_steps, noise_schedule)
    betas = torch.tensor(schedule['betas'], dtype=torch.float32, device=device)
    alphas = torch.tensor(schedule['alphas'], dtype=torch.float32, device=device)
    alphas_cumprod = torch.tensor(schedule['alphas_cumprod'], dtype=torch.float32, device=device)

    # Repeat conditions n_samples times
    conditions = [c.unsqueeze(0).repeat(n_samples, 1).to(device) for c in conditions_dict.values()]

    gi_cond = None
    nc_cond = None
    nc_lag1_cond = None
    if "gi" in conditions_dict:
        gi_cond = conditions_dict['gi'].unsqueeze(0).repeat(n_samples, 1).to(device)
    if "nc" in conditions_dict:
        nc_cond = conditions_dict['nc'].unsqueeze(0).repeat(n_samples, 1).to(device)


    # Initial noise
    x_t = torch.randn((n_samples, ts_length), device=device)  # [n_samples, L]

    with torch.no_grad():  # Disable gradient calculation during inference
        for t in reversed(range(diffusion_steps)):
            t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)

            pred_noise = model(x_t, t_batch, conditions)

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]

            # Reverse sampling step
            x_t = reverse_step(x_t, pred_noise, alpha_t, beta_t, alpha_cumprod_t, t)

    return torch.clamp(x_t, 0.0)


