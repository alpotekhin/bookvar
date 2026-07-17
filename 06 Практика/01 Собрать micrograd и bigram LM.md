---
title: Собрать micrograd и bigram LM
type: practice
status: canonical
last_updated: 2026-07-16
---

# Собрать micrograd и bigram LM

## Цель

Понять backprop и next-token objective до PyTorch-абстракций.

1. Пройти [micrograd](https://github.com/karpathy/micrograd).
2. Проверить gradients finite differences.
3. Реализовать character bigram table.
4. Обучить через negative log-likelihood.
5. Сгенерировать samples и сравнить train/validation loss.

Связано: [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation]].

