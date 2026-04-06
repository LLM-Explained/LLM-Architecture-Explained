from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TokenState:
    value: str
    confidence: float
    hidden_delta: float
    skipped: bool = False


TARGET = list("EARLY SKIPPING")
MASK = "_"


def init_tokens():
    return [TokenState(value=MASK, confidence=0.0, hidden_delta=1.0) for _ in TARGET]


def denoise_step(tokens: list[TokenState], use_skipping: bool, conf_threshold: float = 0.85, delta_threshold: float = 0.08):
    next_tokens = []
    work_units = 0

    for i, tok in enumerate(tokens):
        stable = tok.hidden_delta < delta_threshold

        if use_skipping and stable and tok.confidence > conf_threshold and tok.value != MASK:
            next_tokens.append(TokenState(
                value=tok.value,
                confidence=tok.confidence,
                hidden_delta=tok.hidden_delta * 0.8,
                skipped=True,
            ))
            continue

        work_units += 1

        new_value = tok.value
        if tok.value == MASK and random.random() < 0.35:
            new_value = TARGET[i]

        new_conf = min(1.0, tok.confidence + random.uniform(0.08, 0.22))
        new_delta = max(0.01, tok.hidden_delta * random.uniform(0.45, 0.85))

        next_tokens.append(TokenState(
            value=new_value,
            confidence=new_conf,
            hidden_delta=new_delta,
            skipped=False,
        ))

    return next_tokens, work_units


def seq_string(tokens: list[TokenState]) -> str:
    return "".join(tok.value for tok in tokens)


def run(use_skipping: bool, max_steps: int = 12):
    tokens = init_tokens()
    history = [seq_string(tokens)]
    total_work = 0
    total_skipped = 0

    for _ in range(max_steps):
        tokens, work = denoise_step(tokens, use_skipping=use_skipping)
        total_work += work
        total_skipped += sum(int(tok.skipped) for tok in tokens)
        history.append(seq_string(tokens))

        if seq_string(tokens) == "".join(TARGET):
            break

    return history, total_work, total_skipped


def main():
    random.seed(0)

    hist_full, work_full, skipped_full = run(use_skipping=False)
    hist_skip, work_skip, skipped_skip = run(use_skipping=True)

    print("=== ES-dLLM-inspired early-skipping demo ===\n")

    print("Vanilla denoising:")
    for i, seq in enumerate(hist_full):
        print(f"step {i:2d}: {seq}")
    print(f"steps taken : {len(hist_full)-1}")
    print(f"work units  : {work_full}")
    print(f"skipped     : {skipped_full}\n")

    print("Early-skipping denoising:")
    for i, seq in enumerate(hist_skip):
        print(f"step {i:2d}: {seq}")
    print(f"steps taken : {len(hist_skip)-1}")
    print(f"work units  : {work_skip}")
    print(f"skipped     : {skipped_skip}\n")

    print("Interpretation:")
    print("- Vanilla mode recomputes every token every step.")
    print("- Early-skipping avoids recomputing stable, high-confidence tokens.")
    print("- This is a core backbone for the systems intuition behind ES-dLLM.")


if __name__ == "__main__":
    main()
