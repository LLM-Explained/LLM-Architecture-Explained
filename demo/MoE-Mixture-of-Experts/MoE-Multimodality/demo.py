from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# ------------------------------------------------------------
# Simplified multimodal MoE backbone
# ------------------------------------------------------------
# Core architectural idea implemented here:
#   1) separate text and vision encoders
#   2) fuse multimodal representation
#   3) route fused representation through sparse experts
#   4) allow modality-sensitive specialization
#
# This is a simplified version of the architecture idea implied by
# "vision-language scaling asymmetry + MoE" rather than the full paper stack.
# ------------------------------------------------------------


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 128, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        # text_tokens: [B, T]
        x = self.embed(text_tokens).mean(dim=1)
        return self.proj(x)


class VisionEncoder(nn.Module):
    def __init__(self, image_dim: int = 64, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        # image_feat: [B, image_dim]
        return self.net(image_feat)


class Expert(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalMoE(nn.Module):
    """
    Simplified multimodal MoE architecture:
      text encoder + vision encoder -> fused state -> gated experts -> task head
    """

    def __init__(
        self,
        vocab_size: int = 128,
        text_len: int = 12,
        image_dim: int = 64,
        d_model: int = 64,
        num_experts: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size=vocab_size, d_model=d_model)
        self.vision_encoder = VisionEncoder(
            image_dim=image_dim, d_model=d_model)

        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
        )

        self.router = nn.Linear(2 * d_model, num_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model=2 * d_model, hidden_dim=2 * d_model) for _ in range(num_experts)])

        self.head = nn.Linear(2 * d_model, num_classes)

    def forward(self, text_tokens: torch.Tensor, image_feat: torch.Tensor):
        z_text = self.text_encoder(text_tokens)
        z_vision = self.vision_encoder(image_feat)

        z = torch.cat([z_text, z_vision], dim=-1)
        z = self.fuse(z)

        router_logits = self.router(z)                        # [B, E]
        router_weights = torch.softmax(router_logits, dim=-1)

        expert_outs = torch.stack(
            [expert(z) for expert in self.experts], dim=1)   # [B, E, D]
        mixed = (router_weights.unsqueeze(-1) *
                 expert_outs).sum(dim=1)            # [B, D]

        logits = self.head(mixed)
        return logits, router_weights


# ------------------------------------------------------------
# Synthetic training data to emulate asymmetry
# ------------------------------------------------------------
# We create:
#   - text-heavy examples where labels depend more on text
#   - vision-heavy examples where labels depend more on image
#
# This lets the router learn modality-sensitive specialization.
# ------------------------------------------------------------

@dataclass
class Batch:
    text_tokens: torch.Tensor
    image_feat: torch.Tensor
    labels: torch.Tensor
    mode: str


def make_batch(batch_size: int = 64, text_len: int = 12, vocab_size: int = 128, image_dim: int = 64, mode: str = "text-heavy") -> Batch:
    text_tokens = torch.randint(0, vocab_size, (batch_size, text_len))
    image_feat = torch.randn(batch_size, image_dim)

    # create labels from different dominant modalities
    text_signal = (text_tokens[:, :3].float().mean(
        dim=1) > (vocab_size / 2)).long()
    vision_signal = (image_feat[:, :8].mean(dim=1) > 0.0).long()

    if mode == "text-heavy":
        labels = text_signal
    elif mode == "vision-heavy":
        labels = vision_signal
    else:
        labels = ((text_signal + vision_signal) > 0).long()

    return Batch(text_tokens=text_tokens, image_feat=image_feat, labels=labels, mode=mode)


def train_step(model: MultimodalMoE, optimizer, batch: Batch):
    model.train()
    optimizer.zero_grad()

    logits, router_weights = model(batch.text_tokens, batch.image_feat)
    ce_loss = F.cross_entropy(logits, batch.labels)

    # light load-balancing regularizer so experts don't collapse immediately
    mean_usage = router_weights.mean(dim=0)
    balance_loss = ((mean_usage - 1.0 / router_weights.size(1)) ** 2).mean()

    loss = ce_loss + 0.05 * balance_loss
    loss.backward()
    optimizer.step()

    return float(loss.item()), router_weights.detach()


@torch.no_grad()
def evaluate_router_bias(model: MultimodalMoE):
    model.eval()

    text_batch = make_batch(mode="text-heavy")
    vision_batch = make_batch(mode="vision-heavy")

    _, text_router = model(text_batch.text_tokens, text_batch.image_feat)
    _, vision_router = model(vision_batch.text_tokens, vision_batch.image_feat)

    return text_router.mean(dim=0), vision_router.mean(dim=0)


def main() -> None:
    model = MultimodalMoE(
        vocab_size=128,
        text_len=12,
        image_dim=64,
        d_model=64,
        num_experts=4,
        num_classes=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("=== Simplified multimodal MoE backbone demo ===\n")

    # alternate text-heavy and vision-heavy updates
    for step in range(20):
        mode = "text-heavy" if step % 2 == 0 else "vision-heavy"
        batch = make_batch(mode=mode)
        loss, _ = train_step(model, optimizer, batch)

        if step % 5 == 0 or step == 19:
            text_usage, vision_usage = evaluate_router_bias(model)
            print(f"step={step:02d} loss={loss:.4f}")
            print(
                f"  avg expert usage on text-heavy   : {text_usage.tolist()}")
            print(
                f"  avg expert usage on vision-heavy : {vision_usage.tolist()}")
            print()

    print("Interpretation:")
    print("- Separate encoders create modality-specific representations.")
    print("- The router learns different expert preferences for text-heavy vs vision-heavy inputs.")
    print("- This is the core backbone of the idea that MoE can support modality-sensitive specialization.")
    print("- It is a simplified architecture demo, not the full multimodal pretraining stack from the paper.")


if __name__ == "__main__":
    main()
