# Toy dLLM Demo

This is a minimal runnable demo illustrating the intuition behind diffusion large language models (dLLMs).

## What this demo shows

It compares:

- a toy autoregressive generation loop
- a toy diffusion-style denoising loop
- a toy denoising loop with stability-aware skipping

The point is to show why dLLMs create a different inference optimization surface than AR models.

## What this demo is

A tiny educational example of:

- iterative denoising
- multi-position parallel updates
- stable-token skipping intuition

## What this demo is NOT

This is not a real dLLM.

It does **not** implement:

- diffusion training
- bidirectional transformer inference
- LLaDA / Dream / dLLM framework
- real caching
- real early-skipping kernels
- any benchmark-quality speed measurements

It only demonstrates the generation and optimization intuition in the smallest runnable form.

## Requirements

- Python 3.9+

No extra dependencies are required.

## Run

```bash
python demo.py
```
