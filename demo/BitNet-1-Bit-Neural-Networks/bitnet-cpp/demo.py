from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# -----------------------------
# Packing / unpacking ternary
# -----------------------------
# Encode ternary values {-1, 0, +1} as digits {0, 1, 2}
# and pack 5 values into 1 byte (3^5 = 243 <= 256).
# This gives 1.6 bits / weight in storage, close to the
# theoretical log2(3) ~= 1.58 bits / weight.

VALUES_PER_BYTE = 5
POW3 = [1, 3, 9, 27, 81]


def ternary_encode(t: torch.Tensor) -> torch.Tensor:
    """
    Map {-1,0,+1} -> {0,1,2}
    """
    return (t + 1).to(torch.int16)


def ternary_decode(codes: torch.Tensor) -> torch.Tensor:
    """
    Map {0,1,2} -> {-1,0,+1}
    """
    return (codes.to(torch.int16) - 1).to(torch.float32)


def pack_ternary_row(row_codes: torch.Tensor) -> torch.Tensor:
    """
    row_codes: shape [N], values in {0,1,2}
    returns uint8 packed bytes
    """
    n = row_codes.numel()
    padded = int(math.ceil(n / VALUES_PER_BYTE) * VALUES_PER_BYTE)

    if padded > n:
        pad = torch.ones(padded - n, dtype=row_codes.dtype,
                         device=row_codes.device)
        # pad with 1 -> ternary zero
        row_codes = torch.cat([row_codes, pad], dim=0)

    row_codes = row_codes.view(-1, VALUES_PER_BYTE)
    packed = torch.zeros(row_codes.size(
        0), dtype=torch.uint8, device=row_codes.device)

    for i in range(VALUES_PER_BYTE):
        packed += (row_codes[:, i] * POW3[i]).to(torch.uint8)

    return packed


def unpack_ternary_row(packed: torch.Tensor, original_len: int) -> torch.Tensor:
    """
    packed: uint8 bytes
    returns codes in {0,1,2} of length original_len
    """
    out = []
    for byte in packed.tolist():
        x = int(byte)
        for i in range(VALUES_PER_BYTE):
            out.append(x % 3)
            x //= 3
    return torch.tensor(out[:original_len], dtype=torch.int16)


# -----------------------------
# BitNet-style ternary quantization
# -----------------------------
def ternarize_with_scale(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-output-channel scale.
    w: [out_features, in_features]
    Returns:
      ternary values in {-1,0,1}
      per-row scale [out_features, 1]
    """
    # Simple, educational approximation of BitNet-style scaling
    scale = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
    wt = torch.round(w / scale).clamp(-1, 1)
    return wt.to(torch.int8), scale.to(torch.float32)


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: torch.Tensor) -> torch.Tensor:
        wt, scale = ternarize_with_scale(w)
        return wt.float() * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Straight-through estimator
        return grad_output


@dataclass
class PackedTernaryWeight:
    packed_rows: list[torch.Tensor]
    scales: torch.Tensor
    in_features: int
    out_features: int

    @classmethod
    def from_float_weight(cls, w: torch.Tensor) -> "PackedTernaryWeight":
        wt, scales = ternarize_with_scale(w.detach())
        packed_rows = []
        for r in range(wt.size(0)):
            codes = ternary_encode(wt[r].to(torch.int16))
            packed_rows.append(pack_ternary_row(codes))
        return cls(
            packed_rows=packed_rows,
            scales=scales.squeeze(1).clone(),
            in_features=w.size(1),
            out_features=w.size(0),
        )

    def unpack_to_float(self, device: torch.device | None = None) -> torch.Tensor:
        rows = []
        for r in range(self.out_features):
            codes = unpack_ternary_row(self.packed_rows[r], self.in_features)
            ternary = ternary_decode(codes)
            rows.append(ternary * self.scales[r])
        w = torch.stack(rows, dim=0)
        if device is not None:
            w = w.to(device)
        return w


# -----------------------------
# Minimal BitLinear
# -----------------------------
class BitLinearTrain(nn.Module):
    """
    Training-time fake BitNet layer:
    - keep full-precision master weights
    - use STE ternary projection in forward
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(
            out_features, in_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wq = STEQuantize.apply(self.weight)
        y = x @ wq.t()
        if self.bias is not None:
            y = y + self.bias
        return y

    def export_packed(self) -> tuple[PackedTernaryWeight, torch.Tensor | None]:
        return PackedTernaryWeight.from_float_weight(self.weight), (
            self.bias.detach().clone() if self.bias is not None else None
        )


class BitLinearInfer(nn.Module):
    """
    Inference-time packed BitNet layer:
    - store packed ternary codes + shared scales
    - unpack on the fly for simplicity
    """

    def __init__(self, packed_weight: PackedTernaryWeight, bias: torch.Tensor | None):
        super().__init__()
        self.packed_weight = packed_weight
        if bias is not None:
            self.register_buffer("bias", bias.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.packed_weight.unpack_to_float(device=x.device)
        y = x @ w.t()
        if self.bias is not None:
            y = y + self.bias
        return y


class TinyBitNetTrain(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 32, out_dim: int = 2):
        super().__init__()
        self.fc1 = BitLinearTrain(in_dim, hidden_dim)
        self.fc2 = BitLinearTrain(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def export_infer_model(self) -> "TinyBitNetInfer":
        pw1, b1 = self.fc1.export_packed()
        pw2, b2 = self.fc2.export_packed()
        return TinyBitNetInfer(pw1, b1, pw2, b2)


class TinyBitNetInfer(nn.Module):
    def __init__(
        self,
        pw1: PackedTernaryWeight,
        b1: torch.Tensor | None,
        pw2: PackedTernaryWeight,
        b2: torch.Tensor | None,
    ):
        super().__init__()
        self.fc1 = BitLinearInfer(pw1, b1)
        self.fc2 = BitLinearInfer(pw2, b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# Tiny dataset + training loop
# -----------------------------
def make_xor_data(n: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 2) * 2 - 1
    y = ((x[:, 0] * x[:, 1]) < 0).long()
    return x, y


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1)
        return (pred == y).float().mean().item()


def estimate_storage_bits(packed: PackedTernaryWeight) -> int:
    code_bits = sum(len(row) * 8 for row in packed.packed_rows)
    scale_bits = packed.scales.numel() * 32  # FP32 per output channel
    return code_bits + scale_bits


def main() -> None:
    x_train, y_train = make_xor_data(2048)
    x_test, y_test = make_xor_data(512)

    model = TinyBitNetTrain()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=1e-3)

    for step in range(300):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 299:
            acc = accuracy(model, x_test, y_test)
            print(
                f"train step={step:03d} loss={loss.item():.4f} test_acc={acc:.4f}")

    infer_model = model.export_infer_model()
    infer_acc = accuracy(infer_model, x_test, y_test)

    print(f"\ninference accuracy with packed ternary weights: {infer_acc:.4f}")

    # Show approximate storage
    pw1 = infer_model.fc1.packed_weight
    dense_bits_fc1 = pw1.out_features * pw1.in_features * 32
    packed_bits_fc1 = estimate_storage_bits(pw1)

    print("\nFirst layer storage comparison:")
    print(f"dense FP32 bits : {dense_bits_fc1}")
    print(f"packed ternary+scale bits : {packed_bits_fc1}")
    print(f"effective bits/weight incl shared scales: "
          f"{packed_bits_fc1 / (pw1.out_features * pw1.in_features):.3f}")

    # Show example unpacked ternary weights
    print("\nExample unpacked first 4 rows of first layer:")
    print(pw1.unpack_to_float()[:4])


if __name__ == "__main__":
    main()
