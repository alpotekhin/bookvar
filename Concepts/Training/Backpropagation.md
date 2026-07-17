---
title: "Backpropagation"
aliases: [Backpropagation, Backprop, обратное распространение ошибки]
type: concept
status: legacy
category: Training
papers: []
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 03 — Backprop|CS224N L03]]"
sources:
  - "[Rumelhart, Hinton & Williams — Learning representations by back-propagating errors (1986)](https://www.nature.com/articles/323533a0)"
  - "[Karpathy — Yes you should understand backprop (2016)](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)"
  - "[PyTorch Autograd docs](https://pytorch.org/docs/stable/notes/autograd.html)"
---

# Backpropagation — обратное распространение ошибки

## Зачем это нужно: градиенты для обучения сетей

Обучение нейросети — это минимизация loss $L(\theta)$ по параметрам $\theta$ градиентным спуском:

$$\theta \leftarrow \theta - \eta \nabla_\theta L$$

Для сети с миллиардами параметров нужно вычислить $\nabla_\theta L$ эффективно. Наивный подход (численное дифференцирование, $(L(\theta + \epsilon) - L(\theta))/\epsilon$ для каждого параметра) требует $O(|\theta|)$ forward pass'ов — непрактично.

**Backpropagation** (Rumelhart, Hinton, Williams, 1986) вычисляет **все градиенты за один backward pass**, используя chain rule. Стоимость backward pass $\approx 2\times$ стоимости forward pass — независимо от числа параметров.

## Chain rule — математическая основа

Если $L = f(g(h(x)))$, то:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

Для нейросети это означает: градиент loss по любому параметру — произведение локальных якобианов на пути от параметра к loss.

### Вычислительный граф

Сеть представляется как **DAG** (directed acyclic graph), где узлы — операции, рёбра — тензоры. Forward pass проходит граф от входа к loss, backward — в обратном порядке, применяя chain rule на каждом узле.

```
x ─► [W·x] ─► [+b] ─► [ReLU] ─► [loss] ─► L
     ▲        ▲        
     │        │        
     W        b        
```

## Forward pass + Backward pass

### Forward pass

Вычисляем выход сети и loss, **сохраняя промежуточные активации** (нужны для backward).

Пример: двухслойный MLP
```
z1 = W1 @ x + b1        # linear
a1 = relu(z1)           # activation
z2 = W2 @ a1 + b2       # linear
L  = cross_entropy(z2, y)
```

### Backward pass

Начинаем с $\frac{\partial L}{\partial L} = 1$ и идём в обратном порядке, применяя chain rule:

```
dz2 = softmax(z2) - y                   # from CE loss
dW2 = dz2 @ a1.T                        # ∂L/∂W2 = dz2 · a1ᵀ
db2 = dz2                               # ∂L/∂b2
da1 = W2.T @ dz2                        # ∂L/∂a1
dz1 = da1 * (z1 > 0)                    # ∂L/∂z1 через ReLU'
dW1 = dz1 @ x.T
db1 = dz1
```

**Ключевой принцип:** каждый узел знает свою локальную производную и передаёт градиент назад. Глобальный алгоритм — лишь композиция локальных правил.

## Автоматическое дифференцирование

Современные фреймворки (PyTorch, JAX, TensorFlow) реализуют backprop автоматически через **autodiff**. Есть два режима:

| Режим | Стоимость для $f: \mathbb{R}^n \to \mathbb{R}^m$ |
|-------|--------------------------------------------------|
| Forward-mode | $O(n)$ forward passes |
| Reverse-mode | $O(m)$ backward passes |

Для нейросетей $m = 1$ (скалярный loss) и $n \gg 1$ (миллиарды параметров) — **reverse-mode** несоизмеримо эффективнее. Backprop — это именно reverse-mode autodiff.

### PyTorch autograd

PyTorch строит **dynamic computational graph** во время forward pass:

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)  # [2.0, 4.0]
```

Каждый тензор с `requires_grad=True` хранит `.grad_fn` — ссылку на операцию, его создавшую. `y.backward()` запускает обход графа от $y$ к листьям, заполняя `.grad` у каждого листа.

**Важные детали:**
- Градиенты **накапливаются** в `.grad` — нужно вызывать `optimizer.zero_grad()` перед каждым шагом
- Операции с `torch.no_grad()` не строят граф — экономия памяти на инференсе
- `detach()` отрезает тензор от графа — градиент не потечёт дальше

## Backprop для ключевых операций

### Linear layer $y = Wx + b$

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T, \quad \frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial y}, \quad \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}$$

### Attention

Для [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] $\text{Attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$:

Backward проходит через softmax, два matmul и scaling. Полный вывод даёт градиенты по $Q, K, V$. FlashAttention (Dao 2022) переписывает этот backward для работы в SRAM GPU — 2-4× ускорение без потери точности.

### LSTM

В [[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]] backward проходит через **BPTT** (Backpropagation Through Time). Ключевая формула:

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

Если forget gate $f_t \approx 1$, градиент по cell state проходит без затухания — именно это решает vanishing gradients в vanilla RNN.

## Численная стабильность

Реальные сети обучаются десятки тысяч шагов, и численные проблемы копятся.

### Gradient clipping

Если норма градиента превышает threshold — масштабируем:

$$g \leftarrow g \cdot \frac{\text{threshold}}{\|g\|} \quad \text{если} \quad \|g\| > \text{threshold}$$

Стандарт для обучения LLM — `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`. Предотвращает **exploding gradients**, особенно в RNN и при нестабильном loss.

### Mixed precision

Хранение весов в FP32, вычисления в FP16/BF16. Экономия памяти 2×, ускорение 2-4× на современных GPU (A100, H100).

**Проблема FP16:** узкий диапазон ($\sim 6 \times 10^{-5}$ до $65504$). Маленькие градиенты обнуляются, большие — overflow в $\inf$.

**Решение — loss scaling:**
1. Умножаем loss на большой масштаб $S$ (например, $2^{16}$) перед backward
2. Градиенты становятся больше в $S$ раз — не обнуляются в FP16
3. Перед optimizer step делим градиенты на $S$
4. Если есть $\inf$/$\text{NaN}$ — пропускаем шаг, уменьшаем $S$

PyTorch реализует это через `torch.cuda.amp.GradScaler`. **BF16** (Brain Float 16) имеет тот же диапазон, что FP32, и не требует loss scaling — стандарт для обучения современных LLM.

### Gradient checkpointing

Backward pass требует хранить все активации forward pass — $O(L)$ памяти для $L$ слоёв. **Gradient checkpointing** (Chen et al., 2016) жертвует compute ради memory:

- Храним активации только в избранных точках (checkpoints)
- Между ними пересчитываем forward во время backward
- Memory $O(\sqrt{L})$, compute $\sim 1.33\times$

Критично для обучения больших моделей — без checkpointing 70B модель не влезает в H100.

### NaN debugging

Распространённая проблема: после нескольких шагов loss становится NaN. Причины:
- Деление на 0 (log(0), sqrt(0))
- Overflow в FP16 без loss scaling
- Слишком большой learning rate → exploding gradient → Inf → NaN
- Незамаскированные паддинги в attention

Инструменты: `torch.autograd.set_detect_anomaly(True)` — PyTorch укажет операцию, породившую NaN. Стоит использовать только при отладке (замедляет работу).

## Stop-gradient

Иногда нужно **запретить** градиенту течь через часть графа:

- **Detach target** в self-supervised learning: target net не получает градиентов (BYOL, DINO)
- **REINFORCE / policy gradient**: сэмплирование дискретных токенов — градиент не определён, используется log-prob trick
- **KL penalty** в RLHF: reference model заморожена, градиенты не текут

В PyTorch: `x.detach()` или `with torch.no_grad():`.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, обучаемая backprop
- [[02 Areas/ML & DL/Concepts/Architectures/LSTM|LSTM]] — BPTT как частный случай backprop
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — backward через attention (FlashAttention)
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — основной потребитель backprop compute
- [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] — трюк, сокращающий compute backward pass
