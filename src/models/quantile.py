"""
quantile.py — Quantile Regression with a 1D CNN (QR) benchmark model.

Used as a probabilistic baseline in the paper. The model directly predicts
Q quantile curves (Q=21, from 0 to 1 in steps of 0.05) using a pinball loss.
Quantile curves are treated as "samples" in the shared evaluation pipeline.

Architecture:
  Two Conv1d layers -> Conv1d head outputting Q trajectories [B, Q, T].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = torch.linspace(0.0, 1.0, 21, dtype=torch.float32)  # 0.00, 0.05, ..., 1.00


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QuantileCNN(nn.Module):
    def __init__(self, in_ch: int, base: int = 64, out_q: int = 21):
        """
        Args:
            in_ch:  Number of input channels (= number of conditions).
            base:   Number of Conv1d feature channels.
            out_q:  Number of quantiles to predict.
        """
        super().__init__()
        self.out_q = out_q
        self.feat = nn.Sequential(
            nn.Conv1d(in_ch, base, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(base, base, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(base, out_q, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Condition tensor [B, C, T].

        Returns:
            Quantile predictions [B, Q, T].
        """
        return self.head(self.feat(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_features_seq(
    conditions_dict: dict[str, torch.Tensor],
    order: list[str],
) -> torch.Tensor:
    """
    Stack condition tensors into a channel-first sequence tensor.

    Args:
        conditions_dict: {key: Tensor [B, T]}.
        order:           Keys in the desired channel order.

    Returns:
        Tensor [B, n_conditions, T].
    """
    return torch.cat([conditions_dict[k].unsqueeze(1) for k in order], dim=1)


def _pinball_loss(
    pred_q: torch.Tensor,
    y: torch.Tensor,
    quantiles: torch.Tensor = QUANTILES,
) -> torch.Tensor:
    """
    Pinball (quantile) loss, ignoring boundary quantiles q=0 and q=1.

    Args:
        pred_q:    Predicted quantiles [B, Q, T].
        y:         Ground truth [B, T].
        quantiles: Quantile levels [Q].

    Returns:
        Scalar loss.
    """
    q = quantiles.to(pred_q.device).view(1, -1, 1)
    # Exclude boundary quantiles
    q_mid = q[:, 1:-1, :]
    pred_mid = pred_q[:, 1:-1, :]
    e = y.unsqueeze(1) - pred_mid
    return torch.maximum(q_mid * e, (q_mid - 1.0) * e).mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_quantile_regressor(
    pv_train: torch.Tensor,
    cond_train: dict[str, torch.Tensor],
    cond_order: list[str],
    n_epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    base: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> QuantileCNN:
    """
    Train the quantile regression model.

    Args:
        pv_train:    Normalised PV training tensor [N, T].
        cond_train:  Dict of condition tensors, each [N, T].
        cond_order:  Keys defining channel order.
        n_epochs:    Training epochs.
        batch_size:  Mini-batch size.
        lr:          Adam learning rate.
        base:        CNN base channel width.
        device:      Compute device.

    Returns:
        Trained QuantileCNN model.
    """
    x_train = build_features_seq(cond_train, cond_order).to(device)  # [N, C, T]
    y_train = pv_train.to(device)                                     # [N, T]

    model = QuantileCNN(in_ch=x_train.shape[1], base=base, out_q=len(QUANTILES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    N = x_train.shape[0]
    for ep in range(n_epochs):
        idx = torch.randint(0, N, (batch_size,), device=device)
        loss = _pinball_loss(model(x_train[idx]), y_train[idx])

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 50 == 0:
            print(f"Epoch {ep:4d} | Pinball loss: {loss.item():.4f}")

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_quantiles(
    model: QuantileCNN,
    cond_one: dict[str, torch.Tensor],
    cond_order: list[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Predict quantile curves for a single day.

    The returned tensor has shape [Q, T] and is passed directly into the shared
    evaluation pipeline (refine_samples / compute_pv_stats) treating Q=21 as
    the number of "samples".

    Args:
        model:      Trained QuantileCNN.
        cond_one:   Dict of condition tensors for a single day, each [T].
        cond_order: Keys defining channel order.
        device:     Compute device.

    Returns:
        Tensor [Q, T] on CPU.
    """
    model.eval()
    x = torch.cat(
        [cond_one[k].unsqueeze(0).unsqueeze(0).to(device) for k in cond_order], dim=1
    )  # [1, C, T]
    return model(x).squeeze(0).cpu()  # [Q, T]
