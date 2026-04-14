from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass


torch.manual_seed(0)


# ------------------------------------------------------------
# Attention Residuals backbone demo
# ------------------------------------------------------------
# Major architecture pieces implemented:
#   1) standard residual accumulation across depth
#   2) full AttnRes depth-wise aggregation
#   3) block-style AttnRes over grouped layer representations
#
# This is a simplified implementation of the Kimi / Moonshot AI idea:
# replace fixed additive residual accumulation with learned,
# input-dependent attention over earlier layer outputs.
# ------------------------------------------------------------


@dataclass
class DepthMemory:
    values: list[torch.Tensor]


class SimpleSubLayer(nn.Module):
    """
    Lightweight stand-in for a Transformer sublayer.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class StandardResidualStack(nn.Module):
    def __init__(self, d_model: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [SimpleSubLayer(d_model) for _ in range(depth)])

    def forward(self, x0: torch.Tensor):
        xs = [x0]
        x = x0
        for layer in self.layers:
            x = x + layer(x)
            xs.append(x)
        return x, xs


class FullAttnResStack(nn.Module):
    """
    Full Attention Residuals:
      h_l = sum_i alpha_{i->l} * v_i
    """

    def __init__(self, d_model: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [SimpleSubLayer(d_model) for _ in range(depth)])
        self.depth_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model) * 0.02) for _ in range(depth)]
        )
        self.depth_norm = nn.LayerNorm(d_model)

    def depth_aggregate(self, values: list[torch.Tensor], query_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # values: list of [B, T, D]
        V = torch.stack(values, dim=0)                  # [L, B, T, D]
        K = self.depth_norm(V)                          # [L, B, T, D]
        logits = torch.einsum("d, l b t d -> l b t",
                              query_vec, K)   # [L, B, T]
        alpha = logits.softmax(dim=0)                  # attention over depth
        h = torch.einsum("l b t, l b t d -> b t d", alpha, V)
        return h, alpha

    def forward(self, x0: torch.Tensor):
        values = [x0]
        all_alpha = []

        for i, layer in enumerate(self.layers):
            h, alpha = self.depth_aggregate(values, self.depth_queries[i])
            x = h + layer(h)
            values.append(x)
            all_alpha.append(alpha.detach())

        return values[-1], values, all_alpha


class BlockAttnResStack(nn.Module):
    """
    Simplified Block AttnRes:
    accumulate within block using standard residuals,
    attend only over block-level representations + current partial block.
    """

    def __init__(self, d_model: int, depth: int, block_size: int = 2):
        super().__init__()
        assert depth % block_size == 0
        self.layers = nn.ModuleList(
            [SimpleSubLayer(d_model) for _ in range(depth)])
        self.block_size = block_size
        self.depth_norm = nn.LayerNorm(d_model)
        self.block_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(d_model) * 0.02) for _ in range(depth)]
        )

    def block_aggregate(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        query_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        V = torch.stack(blocks + [partial_block], dim=0)   # [N+1, B, T, D]
        K = self.depth_norm(V)
        logits = torch.einsum("d, l b t d -> l b t", query_vec, K)
        alpha = logits.softmax(dim=0)
        h = torch.einsum("l b t, l b t d -> b t d", alpha, V)
        return h, alpha

    def forward(self, x0: torch.Tensor):
        blocks = [x0]
        partial = x0
        all_alpha = []

        for i, layer in enumerate(self.layers):
            h, alpha = self.block_aggregate(
                blocks, partial, self.block_queries[i])
            out = layer(h)
            partial = partial + out
            all_alpha.append(alpha.detach())

            if (i + 1) % self.block_size == 0:
                blocks.append(partial)

        return partial, blocks, all_alpha


def summarize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    alpha shape:
      full attnres   -> [L_prev, B, T]
      block attnres  -> [N_blocks+1, B, T]
    Return mean attention over batch+tokens.
    """
    return alpha.mean(dim=(1, 2))


def main():
    batch = 2
    seq = 5
    d_model = 16
    depth = 4

    x0 = torch.randn(batch, seq, d_model)

    print("=== Attention Residuals backbone demo ===\n")

    # ------------------------------
    # Standard residual stack
    # ------------------------------
    std_model = StandardResidualStack(d_model=d_model, depth=depth)
    std_out, std_values = std_model(x0)

    print("Standard residual stack:")
    for i, x in enumerate(std_values):
        print(
            f"  depth {i}: mean hidden norm = {x.norm(dim=-1).mean().item():.4f}")
    print()

    # ------------------------------
    # Full AttnRes stack
    # ------------------------------
    full_model = FullAttnResStack(d_model=d_model, depth=depth)
    full_out, full_values, full_alpha = full_model(x0)

    print("Full AttnRes stack:")
    for i, x in enumerate(full_values):
        print(
            f"  depth {i}: mean hidden norm = {x.norm(dim=-1).mean().item():.4f}")
    print("  depth-attention weights by layer:")
    for i, alpha in enumerate(full_alpha):
        print(f"    layer {i}: {summarize_alpha(alpha).tolist()}")
    print()

    # ------------------------------
    # Block AttnRes stack
    # ------------------------------
    block_model = BlockAttnResStack(d_model=d_model, depth=depth, block_size=2)
    block_out, blocks, block_alpha = block_model(x0)

    print("Block AttnRes stack:")
    for i, x in enumerate(blocks):
        print(
            f"  block rep {i}: mean hidden norm = {x.norm(dim=-1).mean().item():.4f}")
    print("  block-attention weights by layer:")
    for i, alpha in enumerate(block_alpha):
        print(f"    layer {i}: {summarize_alpha(alpha).tolist()}")
    print()

    print("Interpretation:")
    print("- Standard residuals accumulate all previous transformations with fixed additive weights.")
    print("- Full AttnRes replaces that with learned, input-dependent attention over depth.")
    print("- Block AttnRes keeps the same idea but attends over block summaries for lower overhead.")
    print("- The depth-attention weights show that different layers can prefer different earlier representations.")


if __name__ == "__main__":
    main()
