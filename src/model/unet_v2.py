import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_sinusoidal_embedding(timesteps, dim):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: Tensor of shape [B]
    :param dim: Embedding dimension
    :return: [B, dim] sinusoidal embeddings
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]  # [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, dim]
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, t):
        emb = get_sinusoidal_embedding(t, dim=self.embed_dim)
        return self.mlp(emb)  # [B, out_dim]


class ModulatedConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = nn.Linear(time_emb_dim, out_ch * 2)

    def forward(self, x, t_emb):
        h = self.conv(x)
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        return F.relu(h * (1 + scale) + shift)


class UNet1D(nn.Module):
    def __init__(self, in_ch: int, base_channels: int, time_embed_dim: int, diffusion_steps: int, depth: int=2):
        super().__init__()
        self.in_ch = in_ch
        self.base_channels = base_channels
        self.time_embed_dim = time_embed_dim
        self.diffusion_steps = diffusion_steps
        self.depth = depth

        self.time_embedding = TimestepEmbedding(time_embed_dim, base_channels)

        # Initial convolution layer
        self.init_conv = nn.Conv1d(in_ch, base_channels, kernel_size=3, padding=1)

        # Variable depth encoder
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2)
        in_channels = base_channels
        self.preconv_channels = []
        for _ in range(depth):
            out_channels = in_channels * 2
            self.enc_blocks.append(ModulatedConv1D(in_channels, out_channels, base_channels))
            self.preconv_channels.append(in_channels)
            in_channels = out_channels

        # Bottleneck
        self.bot1 = ModulatedConv1D(in_channels, in_channels*2, base_channels)
        self.bot2 = ModulatedConv1D(in_channels * 2, in_channels, base_channels)

        # Variable depth decoder
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for preconv_ch in reversed(self.preconv_channels):
            self.up_blocks.append(nn.ConvTranspose1d(in_channels, preconv_ch, kernel_size=2, stride=2))
            self.dec_blocks.append(ModulatedConv1D(in_channels + preconv_ch, preconv_ch, base_channels))
            in_channels = preconv_ch

        # Final output layer
        self.final_conv = nn.Conv1d(in_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, conditions: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the UNet model.
        :param x: Input tensor of shape [B, L].
        :param t: Diffusion timestep tensor of shape [B].
        :param conditions: List of condition tensors (e.g., cond_gi, cond_nc, cond_nc_lag1), each of shape [B, L].
        :return: Output tensor of shape [B, L].
        """
        # Stack conditionals dynamically
        cond = torch.cat([c.unsqueeze(1) for c in conditions], dim=1)  # [B, N_cond, L]
        x = x.unsqueeze(1)  # [B, 1, L]
        x = torch.cat(tensors=[x, cond], dim=1)  # [B, in_ch, L]

        t_emb = self.time_embedding(t)  # [B, base_channels]
        x = self.init_conv(x)  # [B, base_channels, L]

        # # Inject timestep embedding after init_conv
        # context = t_emb.unsqueeze(-1).repeat(1, 1, x.size(2))  # [B, base_channels, L]
        # x = x + context

        # Encoder
        skips = []
        for enc in self.enc_blocks:
            x = F.relu(enc(x, t_emb))  # [B, 2*C, L]
            skips.append(x)  # Store skip connections
            x = self.pool(x)  # Downsample [B, C, L/2]

        # Bottleneck
        x = F.relu(self.bot1(x, t_emb))
        x = F.relu(self.bot2(x, t_emb))

        # Decoder
        for up, dec in zip(self.up_blocks, self.dec_blocks):
            skip = skips.pop() # Get the corresponding skip connection
            x = up(x)          # Upsample [B, C, L*2]
            x = torch.cat([x, skip], dim=1)  # Concatenate skip connection [B, C + skip_ch, L*2]
            x = F.relu(dec(x, t_emb))  # [B, skip_ch, L*2]

        out = self.final_conv(x)

        return out.squeeze(1)                       # [B, L]

