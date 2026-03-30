from __future__ import annotations

import random
from typing import List


TARGET = list("DIFFUSION LLM")
MASK = "_"


def init_masked_sequence(length: int) -> List[str]:
    return [MASK] * length


def autoregressive_generate(target: List[str]) -> List[str]:
    """
    Toy AR generation:
    fill one token per step from left to right.
    """
    out = []
    for ch in target:
        out.append(ch)
    return out


def denoise_step(current: List[str], target: List[str], fill_prob: float = 0.35) -> List[str]:
    """
    Toy diffusion-like denoising:
    some masked positions get resolved in parallel each step.
    """
    next_tokens = current[:]
    for i, ch in enumerate(current):
        if ch == MASK and random.random() < fill_prob:
            next_tokens[i] = target[i]
    return next_tokens


def detect_stable_tokens(prev_tokens: List[str], cur_tokens: List[str]) -> List[bool]:
    return [p == c and c != MASK for p, c in zip(prev_tokens, cur_tokens)]


def update_only_unstable_positions(current: List[str], target: List[str], stable: List[bool], fill_prob: float = 0.35) -> List[str]:
    """
    Toy dLLM optimization intuition:
    skip positions already stable; only spend work on unstable ones.
    """
    next_tokens = current[:]
    for i, ch in enumerate(current):
        if stable[i]:
            continue
        if ch == MASK and random.random() < fill_prob:
            next_tokens[i] = target[i]
    return next_tokens


def toy_diffusion_generate(target: List[str], use_stability_skip: bool, max_steps: int = 12):
    tokens = init_masked_sequence(len(target))
    prev = tokens[:]
    history = [tokens[:]]

    for step in range(max_steps):
        if use_stability_skip:
            stable = detect_stable_tokens(prev, tokens)
            next_tokens = update_only_unstable_positions(
                tokens, target, stable)
        else:
            next_tokens = denoise_step(tokens, target)

        history.append(next_tokens[:])
        prev, tokens = tokens, next_tokens

        if tokens == target:
            break

    return history


def main():
    random.seed(0)

    print("=== Toy AR vs dLLM demo ===\n")

    ar = autoregressive_generate(TARGET)
    print("Autoregressive result:")
    print("".join(ar))
    print(f"AR decode steps (toy): {len(TARGET)}\n")

    hist_plain = toy_diffusion_generate(TARGET, use_stability_skip=False)
    hist_skip = toy_diffusion_generate(TARGET, use_stability_skip=True)

    print("Toy dLLM denoising without stability skip:")
    for i, seq in enumerate(hist_plain):
        print(f"step {i:2d}: {''.join(seq)}")
    print(f"steps taken: {len(hist_plain)-1}\n")

    print("Toy dLLM denoising with stability-aware skipping:")
    for i, seq in enumerate(hist_skip):
        print(f"step {i:2d}: {''.join(seq)}")
    print(f"steps taken: {len(hist_skip)-1}\n")

    print("Interpretation:")
    print("- AR fills one token per step.")
    print("- dLLM can fill multiple positions in parallel.")
    print("- Stability-aware skipping represents the idea that many positions do not need full recomputation every denoising step.")
    print("- This is only a toy illustration, not a real diffusion language model.")


if __name__ == "__main__":
    main()
