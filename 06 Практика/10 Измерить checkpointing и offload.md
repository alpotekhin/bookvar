---
title: Измерить activation checkpointing и offload
type: practice
status: canonical
last_updated: 2026-09-15
---

# Измерить activation checkpointing и offload

Используйте [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/practice_part1.ipynb|EDLS Week 4, practice part 1]] вместе с [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week04_large_models/lecture.pdf|полной лекцией]].

Для одного Transformer block и небольшой stack измерьте baseline, selective
activation checkpointing и offload. В каждом случае сохраните:

- peak allocated и reserved GPU memory;
- step time и tokens/s;
- число повторно выполненных forward operators;
- объём и длительность D2H/H2D copies;
- численное совпадение loss и gradients в заданном tolerance.

Перед сравнением восстановите одинаковые параметры, состояние оптимизатора,
входной пакет и состояния генераторов случайных чисел. При dropout повторный
прямой проход должен использовать ту же маску, что и исходный: иначе
сравниваются производные разных случайных вычислений. В
[`torch.utils.checkpoint`](https://docs.pytorch.org/docs/2.8/checkpoint.html)
это обеспечивает `preserve_rng_state=True`; укажите также
`use_reentrant=False`. Не переносите тензоры на новое устройство внутри
checkpoint-функции: сохранение RNG не охватывает произвольные новые устройства.

Ниже независимый CPU-тест на PyTorch 2.8.0. Параметры модели здесь не нужны:
сравнивается градиент по одному и тому же входу. В последнем запуске
сохранение RNG намеренно отключено.

```python
# bookvar: checkpoint-rng
import torch
from torch.utils.checkpoint import checkpoint

def run(use_checkpoint, preserve=True):
    x = torch.linspace(-1, 1, 32, requires_grad=True)
    torch.manual_seed(19)
    def block(z):
        return torch.nn.functional.dropout(z, p=0.5, training=True).square()
    y = (checkpoint(block, x, use_reentrant=False,
                    preserve_rng_state=preserve)
         if use_checkpoint else block(x))
    loss = y.sum()
    loss.backward()
    return loss.detach(), x.grad.clone()

plain_loss, plain_grad = run(False)
saved_loss, saved_grad = run(True)
changed_loss, changed_grad = run(True, preserve=False)
torch.testing.assert_close(saved_loss, plain_loss)
torch.testing.assert_close(saved_grad, plain_grad)
torch.testing.assert_close(changed_loss, plain_loss)
assert not torch.equal(changed_grad, plain_grad)
print("OK: same forward loss; RNG preservation restores the gradient")
```

Совпадение прямой функции потерь недостаточно: без сохранения RNG ошибка
проявляется именно в повторных вычислениях при обратном проходе. В полном
эксперименте сравните также градиенты параметров и результат шага оптимизатора.

Постройте Pareto-график «peak memory — step time». Затем увеличивайте sequence
length, пока baseline не перестанет помещаться, и покажите, какой режим
расширяет допустимую длину. Не называйте offload бесплатным: на timeline должно
быть видно, перекрылись ли copies с compute или попали на критический путь.
