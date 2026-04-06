from __future__ import annotations

import random
from typing import List, Tuple


MASK = "_"
TARGET = list("DIFFUSION CACHE")


def init_sequence(length: int) -> List[str]:
    return [MASK] * length


def detect_stable_positions(prev_seq: List[str], cur_seq: List[str]) -> List[bool]:
    return [p == c and c != MASK for p, c in zip(prev_seq, cur_seq)]


def denoise_step_full(cur_seq: List[str], target: List[str], fill_prob: float = 0.30) -> Tuple[List[str], int]:
    """
    Full recomputation toy:
    every position is considered each denoising step.
    Returns (next_seq, work_units)
    """
    nxt = cur_seq[:]
    work_units = len(cur_seq)

    for i, ch in enumerate(cur_seq):
        if ch == MASK and random.random() < fill_prob:
            nxt[i] = target[i]
    return nxt, work_units


def denoise_step_cached(
    prev_seq: List[str],
    cur_seq: List[str],
    target: List[str],
    fill_prob: float = 0.30,
) -> Tuple[List[str], int]:
    """
    dLLM-Cache-inspired toy:
    - assume prompt is static and cached
    - only unstable response positions need update
    Returns (next_seq, work_units)
    """
    stable = detect_stable_positions(prev_seq, cur_seq)
    nxt = cur_seq[:]
    work_units = 0

    for i, ch in enumerate(cur_seq):
        if stable[i]:
            continue
        work_units += 1
        if ch == MASK and random.random() < fill_prob:
            nxt[i] = target[i]
    return nxt, work_units


def run_full_denoising(target: List[str], max_steps: int = 12) -> Tuple[List[List[str]], int]:
    seq = init_sequence(len(target))
    history = [seq[:]]
    total_work = 0

    for _ in range(max_steps):
        seq, work = denoise_step_full(seq, target)
        total_work += work
        history.append(seq[:])
        if seq == target:
            break

    return history, total_work


def run_cached_denoising(target: List[str], max_steps: int = 12) -> Tuple[List[List[str]], int]:
    prev = init_sequence(len(target))
    seq = init_sequence(len(target))
    history = [seq[:]]
    total_work = 0

    for _ in range(max_steps):
        nxt, work = denoise_step_cached(prev, seq, target)
        total_work += work
        history.append(nxt[:])
        prev, seq = seq, nxt
        if seq == target:
            break

    return history, total_work


def main() -> None:
    random.seed(0)

    full_hist, full_work = run_full_denoising(TARGET)
    cached_hist, cached_work = run_cached_denoising(TARGET)

    print("=== dLLM-Cache-inspired toy demo ===\n")

    print("Full recomputation path:")
    for i, seq in enumerate(full_hist):
        print(f"step {i:2d}: {''.join(seq)}")
    print(f"steps taken : {len(full_hist)-1}")
    print(f"work units  : {full_work}\n")

    print("Cached partial-update path:")
    for i, seq in enumerate(cached_hist):
        print(f"step {i:2d}: {''.join(seq)}")
    print(f"steps taken : {len(cached_hist)-1}")
    print(f"work units  : {cached_work}\n")

    print("Interpretation:")
    print("- Full recomputation revisits every position every step.")
    print("- Cached partial updates skip positions that stayed stable across adjacent steps.")
    print("- This is a toy stand-in for the idea that dLLMs have reusable structure beyond AR-style prefix caching.")


if __name__ == "__main__":
    main()
