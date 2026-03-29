from __future__ import annotations

import numpy as np


def set_seed(seed: int = 0) -> None:
    np.random.seed(seed)


def fake_fp8_quantize_per_token(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Toy per-token symmetric quantization.
    This is NOT real FP8 encoding.
    """
    scale = np.max(np.abs(x), axis=1, keepdims=True) + 1e-8
    q = np.round(np.clip(x / scale, -1.0, 1.0) * 127).astype(np.int8)
    return q, scale


def fake_fp8_dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) / 127.0 * scale


def naive_quantize_all(kv: np.ndarray) -> np.ndarray:
    q, s = fake_fp8_quantize_per_token(kv)
    return fake_fp8_dequantize(q, s)


def rope_aware_quantize(kv: np.ndarray, rope_dims: int) -> np.ndarray:
    """
    Keep the RoPE-sensitive prefix in high precision.
    Quantize the rest per token.
    """
    rope = kv[:, :rope_dims].copy()
    content = kv[:, rope_dims:]
    q, s = fake_fp8_quantize_per_token(content)
    content_hat = fake_fp8_dequantize(q, s)
    return np.concatenate([rope, content_hat], axis=1)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def make_toy_mla_kv(num_tokens: int = 128, dim: int = 64, rope_dims: int = 16) -> np.ndarray:
    """
    Construct a toy KV tensor where the RoPE-like part is higher frequency / more fragile.
    """
    x = np.random.randn(num_tokens, dim).astype(np.float32) * 0.5

    # Make the first rope_dims more oscillatory / sensitive
    t = np.linspace(0, 8 * np.pi, num_tokens, dtype=np.float32)
    for i in range(rope_dims):
        x[:, i] += 0.1 * np.sin((i + 1) * t)

    return x


def toy_pv_output(query: np.ndarray, kv: np.ndarray) -> np.ndarray:
    """
    Toy downstream consumer standing in for a PV-like stage.
    """
    return query @ kv.T


def main() -> None:
    set_seed(0)

    num_tokens = 128
    dim = 64
    rope_dims = 16

    kv = make_toy_mla_kv(num_tokens=num_tokens, dim=dim, rope_dims=rope_dims)
    query = np.random.randn(8, dim).astype(np.float32)

    kv_naive = naive_quantize_all(kv)
    kv_rope_aware = rope_aware_quantize(kv, rope_dims=rope_dims)

    print("=== SnapMLA-inspired toy demo ===\n")
    print(f"KV shape                 : {kv.shape}")
    print(f"RoPE-sensitive dims kept : {rope_dims}")
    print()

    print("Reconstruction MSE")
    print(f"  naive all-quantized    : {mse(kv, kv_naive):.8f}")
    print(f"  rope-aware quantized   : {mse(kv, kv_rope_aware):.8f}")
    print()

    out_ref = toy_pv_output(query, kv)
    out_naive = toy_pv_output(query, kv_naive)
    out_rope = toy_pv_output(query, kv_rope_aware)

    print("Downstream output MSE (toy PV stage)")
    print(f"  naive all-quantized    : {mse(out_ref, out_naive):.8f}")
    print(f"  rope-aware quantized   : {mse(out_ref, out_rope):.8f}")
    print()

    print("Interpretation:")
    print("- Quantizing everything uniformly is simple but can over-damage sensitive substructure.")
    print("- Keeping RoPE-sensitive dimensions in higher precision can reduce downstream error.")
    print("- Real SnapMLA also addresses PV pipeline reconstruction and end-to-end dataflow.")


if __name__ == "__main__":
    main()
