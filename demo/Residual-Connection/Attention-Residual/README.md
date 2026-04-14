# Attention Residuals Backbone

This repository provides a **core backbone** for the main architectural idea behind **Attention Residuals (AttnRes)** from the Moonshot AI / Kimi team:

replace fixed additive residual accumulation with learned, input-dependent attention over earlier layer representations.

## What this code covers

This scaffold implements the architecture idea itself in three forms:

- **standard residual accumulation**
- **full AttnRes**
- **block-style AttnRes**

It is designed to make the key structural difference explicit.

## Core equations

Standard residual accumulation:

$$
\mathbf{h}_l = \mathbf{h}_{l-1} + F_l(\mathbf{h}_{l-1})
$$
