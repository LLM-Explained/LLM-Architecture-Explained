# Simplified LeWorldModel Backbone

This repository provides a **core backbone** for the main architectural and training ideas behind **LeWorldModel (LeWM)**.

It is designed as a compact, runnable scaffold that preserves the major modeling decisions from the official LeWM stack while stripping away dataset, environment, and infrastructure layers that are not essential for understanding the core method.

## What this code covers

This backbone includes the main elements that define LeWM:

- **encoder**
  - maps raw pixel observations into a compact latent state

- **predictor**
  - predicts the next latent state from the current latent state and the action

- **two-term training objective**
  - next-embedding prediction loss
  - Gaussian latent regularization (**SIGReg-style backbone**)

- **latent-space planning**
  - Cross-Entropy Method (CEM) over candidate action sequences
  - planning directly in latent space using the learned dynamics model

## Architectural summary

The backbone follows the same high-level decomposition as the official LeWM work:

```text
observation o_t  -> encoder   -> z_t
(z_t, action a_t) -> predictor -> z_hat_{t+1}
```
