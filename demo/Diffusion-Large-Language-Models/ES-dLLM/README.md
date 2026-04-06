# ES-dLLM Early-Skipping Demo

This repository provides a foundational prototype for the main systems intuition behind ES-dLLM:

stable, low-importance tokens in diffusion language model inference do not always need full recomputation in early layers.

## What this demo shows

The demo compares two denoising modes:

- **vanilla denoising**
  - every token is recomputed every iteration

- **early-skipping denoising**
  - stable, high-confidence tokens are skipped
  - compute is focused on tokens that are still changing

It reports:

- denoising traces
- total work units
- number of skipped tokens

## What this code is

A compact research scaffold for the key ES-dLLM systems idea:
dynamic compute allocation based on token stability and confidence.

## Scope

This code is not the full ES-dLLM method.

It does not implement:

- real diffusion language models
- actual hidden states, K/V tensors, or confidence estimators
- real transformer layers
- benchmark-grade inference measurements

Instead, it serves as a core backbone for understanding why early-skipping can reduce redundant computation in iterative denoising.

## Requirements

- Python 3.9+

No extra dependencies are required.

## Run

```bash
python demo.py
```
