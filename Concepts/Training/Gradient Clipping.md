---
title: "Gradient Clipping"
aliases: [Gradient Clipping, Клиппинг градиентов, Обрезка градиентов]
type: concept
category: Training
papers: []
courses: []
sources:
  - "[Pascanu et al. — On the difficulty of training RNNs (2013)](https://arxiv.org/abs/1211.5063)"
  - "[PyTorch docs — torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)"
---

# Gradient Clipping — обрезка градиентов

## Зачем это нужно: проблема взрывающихся градиентов

При обучении глубоких сетей (особенно [[RNN]], [[LSTM]], [[Attention Is All You Need|Transformer]]) градиенты могут **взрываться** — их норма становится огромной ($10^6$ и выше). Один такой шаг SGD/Adam «выбрасывает» веса далеко от разумной области, и обучение расходится: loss становится `NaN` или `Inf`, и восстановить его невозможно.

Причина — цепное правило: градиент через $T$ слоёв содержит произведение $T$ якобианов. Если их спектральные нормы $>1$, произведение растёт экспоненциально по глубине. Для RNN с длиной последовательности 1000 это особенно критично.

**Gradient clipping** — простейшая и самая эффективная регуляризация против этого: перед шагом оптимизатора ограничиваем норму (или значение) градиентов сверху.

## Clip by Value vs Clip by Norm

### Clip by value

Каждая компонента градиента обрезается поэлементно:

$$g_i \leftarrow \text{clip}(g_i, -c, c)$$

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**Минус:** меняет **направление** градиента (не только длину). Если одна компонента огромна, а остальные малы — после clip направление искажается. Используется редко.

### Clip by norm (стандарт)

Обрезается **L2-норма** всего вектора градиента:

$$g \leftarrow g \cdot \min\left(1, \frac{\text{max\_norm}}{\|g\|_2}\right)$$

Направление градиента сохраняется, меняется только длина. Это эквивалентно «шагу в том же направлении, но ограниченной длины».

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Global vs per-parameter norm:** по умолчанию норма считается **глобально** (конкатенация всех градиентов в один вектор). Альтернатива — обрезать каждый тензор отдельно, но это менее стабильно.

## Стандартные значения

| Тип модели | `max_norm` |
|------------|-----------|
| RNN/LSTM | 0.25 – 5.0 (часто 1.0) |
| Transformer (pre-training LLM) | 1.0 |
| Transformer (fine-tuning) | 1.0 |
| RLHF/PPO | 0.5 – 1.0 |
| Diffusion models | 1.0 |

**Правило большого пальца:** `max_norm=1.0` работает в 90% случаев. Если градиенты часто превышают порог — либо lr слишком большой, либо инициализация плохая, либо есть bug в loss.

## Где ставить clip в training loop

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad()
```

Важно: clip **после** `backward()` (градиенты уже посчитаны) и **до** `optimizer.step()` (оптимизатор использует обрезанные).

### С [[Mixed Precision Training|Mixed Precision]]

При FP16 градиенты масштабируются через loss scaling. Clip нужно делать **после unscale**:

```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                 # важно — unscale до clip
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

## Почему критично для RNN

В RNN градиент через $T$ шагов времени содержит произведение $T$ якобианов рекуррентной матрицы $W_h$. Если спектральный радиус $\rho(W_h) > 1$, то норма градиента растёт как $\rho^T$ — экспоненциально по длине последовательности.

Pascanu et al. (2013) показали, что **gradient clipping + careful initialization** — необходимое условие для обучения vanilla RNN на длинных последовательностях. Без clip обучение расходится практически всегда.

[[LSTM]] и [[GRU]] частично решают проблему через gating (градиент течёт через аддитивный путь cell state), но clip всё равно используется как safety net.

## Почему важно для LLM

У трансформеров формально нет рекурренции, но:

1. **Длинный контекст** — backprop через 4K-128K токенов создаёт длинные пути.
2. **Residual stream** — градиенты суммируются через все слои, могут накапливать большие значения.
3. **Adam с большими learning rate** — даже редкий spike в градиенте даёт разрушительный шаг.
4. **Разреженные обновления** — при MoE или sparse experts отдельные эксперты могут получать огромные градиенты.

Почти все публичные LLM (GPT, LLaMA, DeepSeek, Qwen) используют `max_norm=1.0`. В [[PPO]]/[[GRPO]] clipping ещё важнее — policy gradient estimator по своей природе шумный.

## Gradient spikes как индикатор проблем

Мониторинг `grad_norm` (до clip) — один из ключевых сигналов здоровья обучения:

- **Стабильные значения $0.1 – 1.0$** — норма, обучение здоровое.
- **Периодические spikes до $10 – 100$** — возможно, bad batches или instability, clip спасает.
- **Постоянные spikes $> 100$** — серьёзная проблема: bad init, слишком большой lr, bug в loss.
- **`NaN`** — уже поздно, нужно rollback к чекпоинту и пересмотреть hyperparams.

В логах `wandb`/`tensorboard` стоит всегда писать `grad_norm` до clip.

## Адаптивные варианты

- **Adaptive gradient clipping (AGC)** — норма ограничивается относительно нормы параметров: $\|g\| \leq \lambda \|w\|$. Используется в NFNet (без batch norm).
- **Percentile-based clipping** — `max_norm` подбирается автоматически как 90-й перцентиль исторических grad_norm.
- **Z-loss** (PaLM, Gemini) — регуляризация логитов softmax, уменьшает необходимость агрессивного clip в attention.

## Gradient clipping vs другие стабилизаторы

| Техника | Что решает |
|---------|-----------|
| Gradient clipping | Взрывающиеся градиенты |
| Residual connections | Исчезающие градиенты |
| [[Layer Normalization]] | Ковариативный сдвиг активаций |
| Warmup learning rate | Нестабильность в начале обучения |
| Weight decay | Рост норм весов |

Все они обычно используются **вместе** — это комплементарные механизмы.

## Related concepts

- [[Vanishing Gradient]] — обратная проблема (исчезание, а не взрыв)
- [[Mixed Precision Training]] — clip должен быть после unscale
- [[Layer Normalization]] — стабилизирует активации и косвенно градиенты
- [[PPO]], [[GRPO]] — RL-алгоритмы, где clip особенно критичен
- [[RNN]], [[LSTM]] — архитектуры, для которых clip исторически придуман
