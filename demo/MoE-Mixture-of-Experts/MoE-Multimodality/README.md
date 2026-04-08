# Multimodal MoE Backbone for Modality Scaling Asymmetry

This repository provides a **core backbone** for the architectural idea behind recent multimodal scaling work:

vision and language may scale differently, and **Mixture-of-Experts (MoE)** can help reconcile that asymmetry in a unified model.

## What this code covers

This backbone includes the major architectural choices needed to express the idea:

- **separate text encoder**
- **separate vision encoder**
- **multimodal fusion**
- **expert router**
- **sparse expert mixing**
- **task head**
- **light expert-balancing regularization**

It is designed to show how modality-sensitive routing and specialization can emerge in a compact, runnable setting.

## Architectural summary

The simplified architecture is:

```text
text -> text encoder ----\
                          -> fused state -> router -> experts -> head
image -> vision encoder -/
```
