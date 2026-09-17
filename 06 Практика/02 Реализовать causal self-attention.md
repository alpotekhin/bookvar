---
title: Реализовать causal self-attention
type: practice
status: canonical
last_updated: 2026-09-15
---

# Реализовать causal self-attention

Причинная маска не просто делает матрицу внимания треугольной. Она гарантирует,
что представление позиции не зависит от ещё не сгенерированного продолжения.
Начните со [встроенной интерактивной лаборатории](/practice/causal-self-attention/):
увеличьте оценку запрещённой будущей связи, отключите маску и выполните пять
проверок. После этого воспроизведите тот же механизм в PyTorch.

## Полностью воспроизводимый пример

Входы ниже — уже вычисленные проекции Q, K и V. Их форма
`[B, H, T, D]`: два примера, три головы, пять позиций и четыре признака
в каждой голове. Здесь длины Q и K равны: это полная обработка
последовательности без KV-кэша. Прямоугольная маска при декодировании с кэшем
требует отдельного учёта абсолютных позиций.

```python
# bookvar: causal-attention
import torch
from torch.nn import functional as F

torch.manual_seed(7)
B, H, T, D = 2, 3, 5, 4
q, k, v = [torch.randn(B, H, T, D, dtype=torch.float64)
           for _ in range(3)]
causal_mask = torch.ones(T, T, dtype=torch.bool).tril()

def attention(q, k, v):
    scores = (q @ k.transpose(-2, -1)) / D**0.5
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights

output, weights = attention(q, k, v)
reference = F.scaled_dot_product_attention(
    q, k, v, is_causal=True, dropout_p=0.0
)
torch.testing.assert_close(output, reference, atol=1e-10, rtol=1e-10)
assert torch.count_nonzero(weights.triu(1)) == 0
torch.testing.assert_close(weights.sum(dim=-1), torch.ones(B, H, T,
                                                         dtype=q.dtype))

# Меняем только последнюю позицию: предыдущие выходы не должны измениться.
q2, k2, v2 = [x.clone() for x in (q, k, v)]
q2[:, :, -1] += 100
k2[:, :, -1] -= 100
v2[:, :, -1] += 100
changed, _ = attention(q2, k2, v2)
torch.testing.assert_close(changed[:, :, :-1], output[:, :, :-1])

# Возвращаем порядок осей и объединяем головы; выходной проекции здесь нет.
merged = output.transpose(1, 2).contiguous().view(B, T, H * D)
assert merged.shape == (B, T, H * D)
print("OK: mask, SDPA, future independence, shapes")
```

Код проверен на CPU в PyTorch 2.8.0 с `float64`. Это проверка формулы, не
измерение скорости FlashAttention. Для GPU и пониженной точности нужны
соответствующие допуски. В [API SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)
dropout управляется аргументом `dropout_p`; вызов `eval()` у внешнего модуля
сам по себе не меняет переданное в функцию значение. Поэтому при сравнении
явно указан ноль.

Важно маскировать оценки **до** softmax. Если сначала нормировать строку, а
затем обнулить запрещённые элементы, сумма оставшихся весов перестанет быть
равной единице.

## Проверки реализации

- после объединения голов получается `B × T × (H·D)`; в полном слое
  выходная проекция возвращает ширину модели `C`;
- верхний треугольник матрицы весов равен нулю;
- каждая строка после softmax суммируется в единицу;
- результат совпадает с PyTorch SDPA на небольшом тензоре;
- изменение будущего токена не меняет выходы предыдущих позиций.

## Материалы

- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)
