---
title: Реализовать causal self-attention
type: practice
status: canonical
last_updated: 2026-07-18
---

# Реализовать causal self-attention

Причинная маска не просто делает матрицу внимания треугольной. Она гарантирует,
что представление позиции не зависит от ещё не сгенерированного продолжения.
Начните со [встроенной интерактивной лаборатории](/practice/causal-self-attention/):
увеличьте оценку запрещённой будущей связи, отключите маску и выполните пять
проверок. После этого воспроизведите тот же механизм в PyTorch.

Минимальная последовательность операций:

```python
scores = (q @ k.transpose(-2, -1)) / head_dim**0.5
scores = scores.masked_fill(~causal_mask, float("-inf"))
weights = scores.softmax(dim=-1)
output = weights @ v
```

Важно маскировать оценки **до** softmax. Если сначала нормировать строку, а
затем обнулить запрещённые элементы, сумма оставшихся весов перестанет быть
равной единице.

## Проверки реализации

- форма `B × T × C` сохраняется;
- верхний треугольник матрицы весов равен нулю;
- каждая строка после softmax суммируется в единицу;
- результат совпадает с PyTorch SDPA на небольшом тензоре;
- изменение будущего токена не меняет выходы предыдущих позиций.

## Материалы

- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
