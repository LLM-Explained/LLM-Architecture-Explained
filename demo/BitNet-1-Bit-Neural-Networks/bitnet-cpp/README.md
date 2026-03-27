# Tiny BitNet Demo (bitnet.cpp-inspired)

This is a minimal runnable demo inspired by the current public BitNet stack:

- BitNet b1.58 / BitNet b1.58 2B4T
- bitnet.cpp as the official inference framework for 1-bit / ternary LLMs

## What this demo shows

This demo keeps the three most important ideas:

1. ternary weights in `{-1, 0, +1}`
2. shared scaling factors (instead of one float per weight)
3. packed storage for ternary codes

It also uses a training-time straight-through estimator (STE) so the toy network can learn.

## What this demo does NOT implement

This is **not** a reproduction of bitnet.cpp kernels.

It does **not** implement:

- TL / ternary lookup-table kernels
- I2_S kernels
- the actual 2B4T model
- llama.cpp / bitnet.cpp runtime integration
- GPU / CPU optimized mpGEMM kernels

Instead, it is the smallest educational demo that stays faithful to the core representation idea.

## Why this is closer to BitNet than a normal toy quantization demo

A lot of toy “BitNet” examples just ternarize a float tensor in the forward pass and still store everything as dense FP32.

This demo is better because it actually:

- packs ternary codes
- keeps scales shared per output channel
- reconstructs weight values from `scale * ternary_code`

That is much closer to the real storage story.

## Requirements

- Python 3.9+
- PyTorch

## Install

```bash
pip install torch
