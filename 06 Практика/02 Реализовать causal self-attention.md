---
title: Реализовать causal self-attention
type: practice
status: canonical
last_updated: 2026-07-16
---

# Реализовать causal self-attention

## Проверки реализации

- shape `B × T × C` сохраняется;
- верхний треугольник weights равен нулю;
- каждая строка softmax суммируется в 1;
- output совпадает с PyTorch SDPA на малом tensor;
- изменение будущего токена не меняет прошлые outputs.

Опоры:

- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

