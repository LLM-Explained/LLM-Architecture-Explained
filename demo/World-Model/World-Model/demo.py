from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# ------------------------------------------------------------
# Core LeWorldModel backbone
# ------------------------------------------------------------
# Simplified from the official LeWM design:
#   z_t   = encoder(o_t)
#   ẑ_t+1 = predictor(z_t, a_t)
#
# Training objective:
#   L = L_pred + λ * SIGReg(Z)
#
# Planning:
#   encode start / goal observations
#   optimize action sequence in latent space with CEM
# ------------------------------------------------------------


class ConvPatchEncoder(nn.Module):
    """
    Foundational image encoder backbone:
    raw pixel observation -> compact latent embedding.

    In the official LeWorldModel stack, the encoder is the first major component
    and maps frame observations into a low-dimensional latent state. Here we use
    a compact conv-based encoder to keep the demo runnable and concise.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4,
                      stride=2, padding=1),  # 64 -> 32
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2,
                      padding=1),           # 32 -> 16
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2,
                      padding=1),          # 16 -> 8
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        z = self.proj(x)
        return z


class ActionEncoder(nn.Module):
    """
    Minimal action encoder.
    """

    def __init__(self, action_dim: int, action_emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, action_emb_dim),
            nn.GELU(),
            nn.Linear(action_emb_dim, action_emb_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


class LatentPredictor(nn.Module):
    """
    Simplified latent dynamics predictor:
    predicts z_{t+1} from z_t and a_t.

    The official LeWM uses a dedicated predictor over latent states conditioned
    on actions. Here we keep the same structural role with a compact MLP block.
    """

    def __init__(self, latent_dim: int = 192, action_emb_dim: int = 64, hidden_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, a_emb], dim=-1)
        return self.net(x)


class LeWorldModel(nn.Module):
    """
    Core LeWorldModel backbone:
      encoder + action encoder + predictor
    """

    def __init__(self, obs_channels: int = 3, action_dim: int = 4, latent_dim: int = 192, action_emb_dim: int = 64):
        super().__init__()
        self.encoder = ConvPatchEncoder(
            in_channels=obs_channels, latent_dim=latent_dim)
        self.action_encoder = ActionEncoder(
            action_dim=action_dim, action_emb_dim=action_emb_dim)
        self.predictor = LatentPredictor(
            latent_dim=latent_dim, action_emb_dim=action_emb_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_next_latent(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_encoder(action)
        return self.predictor(z_t, a_emb)

    def forward(self, obs_t: torch.Tensor, action_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_t = self.encode(obs_t)
        z_hat_tp1 = self.predict_next_latent(z_t, action_t)
        return z_t, z_hat_tp1


# ------------------------------------------------------------
# SIGReg-style Gaussian regularizer backbone
# ------------------------------------------------------------
# The official LeWM objective uses:
#   L = L_pred + λ * SIGReg(Z)
#
# This demo uses a compact Gaussian regularization scaffold that captures the
# major intent:
#   - zero-mean latent dimensions
#   - unit-variance latent dimensions
#   - decorrelated latent dimensions
# ------------------------------------------------------------

def sigreg_backbone(z: torch.Tensor) -> torch.Tensor:
    """
    Foundational SIGReg-style regularizer.
    z: [batch, latent_dim]
    """
    z = z - z.mean(dim=0, keepdim=True)

    # mean and variance shaping
    mean_loss = z.mean(dim=0).pow(2).mean()
    var = z.var(dim=0, unbiased=False)
    var_loss = (var - 1.0).pow(2).mean()

    # decorrelation
    cov = (z.T @ z) / z.size(0)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).mean()

    return mean_loss + var_loss + cov_loss


@dataclass
class TrainStats:
    loss: float
    pred_loss: float
    reg_loss: float


def training_step(
    model: LeWorldModel,
    optimizer: torch.optim.Optimizer,
    obs_t: torch.Tensor,
    action_t: torch.Tensor,
    obs_tp1: torch.Tensor,
    reg_weight: float = 0.1,
) -> TrainStats:
    """
    Core training step:
      1. encode o_t
      2. predict ẑ_{t+1} from z_t and a_t
      3. encode o_{t+1}
      4. compute next-embedding prediction loss
      5. add Gaussian latent regularizer
      6. optimize end-to-end
    """
    model.train()
    optimizer.zero_grad()

    z_t = model.encode(obs_t)
    z_hat_tp1 = model.predict_next_latent(z_t, action_t)
    z_tp1 = model.encode(obs_tp1)

    pred_loss = F.mse_loss(z_hat_tp1, z_tp1)
    reg_loss = sigreg_backbone(torch.cat([z_t, z_tp1], dim=0))
    loss = pred_loss + reg_weight * reg_loss

    loss.backward()
    optimizer.step()

    return TrainStats(
        loss=float(loss.item()),
        pred_loss=float(pred_loss.item()),
        reg_loss=float(reg_loss.item()),
    )


# ------------------------------------------------------------
# Latent-space planning with CEM
# ------------------------------------------------------------
# The LeWM website describes planning by:
#   - encoding start / goal observations
#   - rolling out candidate action sequences through the predictor
#   - selecting those ending closest to the goal latent
#   - optimizing with Cross-Entropy Method (CEM)
# ------------------------------------------------------------

@torch.no_grad()
def rollout_latents(
    model: LeWorldModel,
    z_start: torch.Tensor,
    action_seq: torch.Tensor,
) -> torch.Tensor:
    """
    Roll out latent trajectory under a candidate action sequence.

    z_start: [1, latent_dim]
    action_seq: [horizon, action_dim]
    returns final latent: [1, latent_dim]
    """
    z = z_start
    for t in range(action_seq.size(0)):
        a_t = action_seq[t:t+1]
        z = model.predict_next_latent(z, a_t)
    return z


@torch.no_grad()
def cem_plan(
    model: LeWorldModel,
    obs_start: torch.Tensor,
    obs_goal: torch.Tensor,
    horizon: int = 8,
    action_dim: int = 4,
    num_samples: int = 128,
    num_elites: int = 16,
    num_iters: int = 5,
    action_std_init: float = 1.0,
) -> Optional[torch.Tensor]:
    """
    Minimal CEM planner in latent space.
    Returns the best action sequence [horizon, action_dim].
    """
    model.eval()

    z_start = model.encode(obs_start)   # [1, latent_dim]
    z_goal = model.encode(obs_goal)     # [1, latent_dim]

    mean = torch.zeros(horizon, action_dim, device=obs_start.device)
    std = torch.ones(horizon, action_dim,
                     device=obs_start.device) * action_std_init

    best_seq = None
    best_score = float("inf")

    for _ in range(num_iters):
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            num_samples, horizon, action_dim, device=obs_start.device
        )

        scores = []
        for i in range(num_samples):
            z_final = rollout_latents(model, z_start, samples[i])
            score = F.mse_loss(z_final, z_goal).item()
            scores.append(score)

        scores_t = torch.tensor(scores, device=obs_start.device)
        elite_idx = torch.topk(scores_t, k=num_elites, largest=False).indices
        elites = samples[elite_idx]

        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp(min=1e-3)

        if scores_t[elite_idx[0]].item() < best_score:
            best_score = scores_t[elite_idx[0]].item()
            best_seq = samples[elite_idx[0]].clone()

    return best_seq


# ------------------------------------------------------------
# Minimal runnable example
# ------------------------------------------------------------

def make_dummy_batch(batch_size: int = 16, obs_size: int = 64, action_dim: int = 4):
    obs_t = torch.randn(batch_size, 3, obs_size, obs_size)
    action_t = torch.randn(batch_size, action_dim)
    obs_tp1 = torch.randn(batch_size, 3, obs_size, obs_size)
    return obs_t, action_t, obs_tp1


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LeWorldModel(obs_channels=3, action_dim=4,
                         latent_dim=192, action_emb_dim=64).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("=== Simplified LeWorldModel backbone demo ===\n")

    # ----------------------------------------
    # 1) training loop backbone
    # ----------------------------------------
    print("Training:")
    for step in range(5):
        obs_t, action_t, obs_tp1 = make_dummy_batch(
            batch_size=16, obs_size=64, action_dim=4)
        obs_t = obs_t.to(device)
        action_t = action_t.to(device)
        obs_tp1 = obs_tp1.to(device)

        stats = training_step(
            model=model,
            optimizer=optimizer,
            obs_t=obs_t,
            action_t=action_t,
            obs_tp1=obs_tp1,
            reg_weight=0.1,
        )
        print(
            f"step={step:02d} "
            f"loss={stats.loss:.4f} "
            f"pred={stats.pred_loss:.4f} "
            f"sigreg={stats.reg_loss:.4f}"
        )

    # ----------------------------------------
    # 2) latent planning backbone
    # ----------------------------------------
    print("\nPlanning:")
    obs_start = torch.randn(1, 3, 64, 64, device=device)
    obs_goal = torch.randn(1, 3, 64, 64, device=device)

    action_seq = cem_plan(
        model=model,
        obs_start=obs_start,
        obs_goal=obs_goal,
        horizon=6,
        action_dim=4,
        num_samples=64,
        num_elites=8,
        num_iters=4,
        action_std_init=1.0,
    )

    print(f"planned action sequence shape: {tuple(action_seq.shape)}")
    print("first action vector:", action_seq[0].cpu())

    # ----------------------------------------
    # 3) inspect latent geometry
    # ----------------------------------------
    print("\nLatent inspection:")
    z = model.encode(torch.randn(8, 3, 64, 64, device=device))
    print("latent batch shape:", tuple(z.shape))
    print("latent mean abs   :", z.mean(dim=0).abs().mean().item())
    print("latent std mean   :", z.std(dim=0).mean().item())


if __name__ == "__main__":
    main()
