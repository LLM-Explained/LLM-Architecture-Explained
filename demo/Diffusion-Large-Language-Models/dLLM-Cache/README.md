# dLLM-Cache-Inspired Toy Demo

This is a minimal runnable demo illustrating the core intuition behind dLLM-Cache:

diffusion language models have reusable structure that ordinary AR caching misses.

## What this demo shows

It compares two toy denoising pipelines:

1. **full recomputation**
   - every position is reconsidered every denoising step

2. **cached partial update**
   - positions that remain stable across adjacent steps are skipped
   - only unstable positions are updated

The demo tracks:

- denoising steps
- a toy "work units" count
- how much repeated computation can be avoided

## What this demo is

A tiny educational simulation of the dLLM-Cache idea:
reuse in dLLMs comes from static prompt structure and partial response stability, not from ordinary AR prefix growth.

## What this demo is NOT

This is not the real dLLM-Cache implementation.

It does **not** implement:

- bidirectional transformer inference
- actual key/value caches
- feature-similarity-guided updates
- real prompt-cache intervals
- latency benchmarks
- LLaDA or Dream inference

It only demonstrates the systems intuition in the smallest runnable form.

## Requirements

- Python 3.9+

No extra dependencies are required.

## Run

```bash
python demo.py
```
