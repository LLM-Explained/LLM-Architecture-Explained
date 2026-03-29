# SnapMLA-Inspired Toy Demo

This is a minimal runnable demo inspired by the core intuition behind SnapMLA:

- not all MLA KV-cache substructures are equally quantization-friendly
- RoPE-sensitive parts may need higher precision
- per-token quantization can align better with autoregressive decoding

## What this demo shows

The demo compares two toy strategies on synthetic KV-like tensors:

1. **naive all-quantized**
   - quantize the whole KV tensor uniformly

2. **RoPE-aware quantized**
   - keep a small RoPE-sensitive prefix in higher precision
   - quantize the rest per token

It then compares:

- reconstruction MSE
- downstream toy PV-stage output MSE

## What this demo is

A tiny educational simulation of the first major SnapMLA idea.

## What this demo is NOT

This is not the real SnapMLA implementation.

It does **not** implement:

- real FP8 formats
- MLA kernels
- PV GEMM pipeline reconstruction
- Hopper-specific kernel/dataflow optimization
- SGLang integration
- benchmark-grade performance testing

It only demonstrates the architectural intuition in the smallest runnable form.

## Requirements

- Python 3.9+
- NumPy

## Install

```bash
pip install numpy
```
