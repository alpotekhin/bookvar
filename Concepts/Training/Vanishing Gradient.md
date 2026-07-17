---
title: "Vanishing Gradient"
aliases: [Vanishing Gradient, Исчезающие градиенты, Исчезновение градиента]
type: concept
category: Training
papers: []
courses: []
sources:
  - "[Hochreiter — Untersuchungen zu dynamischen neuronalen Netzen (1991)](https://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)"
  - "[Bengio et al. — Learning Long-Term Dependencies with Gradient Descent is Difficult (1994)](https://www.researchgate.net/publication/5583935)"
  - "[He et al. — Deep Residual Learning (2015)](https://arxiv.org/abs/1512.03385)"
---

# Vanishing Gradient — исчезающие градиенты

## В чём проблема

При обучении глубоких сетей через backpropagation градиенты распространяются от выхода к входу по цепному правилу. Если на каждом слое градиент **уменьшается** (множитель $< 1$), то после $L$ слоёв он становится экспоненциально малым:

$$\frac{\partial L}{\partial W^{(1)}} \propto \prod_{l=2}^{L} \frac{\partial h^{(l)}}{\partial h^{(l-1)}}$$

Если каждый множитель $\sim 0.5$, то через 20 слоёв градиент $\sim 10^{-6}$, через 50 слоёв — $\sim 10^{-15}$ (ниже точности FP32). Нижние слои **перестают обучаться**: веса фактически заморожены, хотя loss может уменьшаться за счёт верхних слоёв.

Симптомы:
- Loss падает, но медленно и выходит на плато.
- Норма градиента на нижних слоях на порядки меньше, чем на верхних.
- Модель «забывает» далёкие в прошлом зависимости ([[RNN]] на длинных последовательностях).

Это **обратная проблема** [[Gradient Clipping|взрывающихся градиентов]] — градиенты не взрываются, а исчезают.

## Математическая причина

Рассмотрим MLP с сигмоидной активацией: $h^{(l)} = \sigma(W^{(l)} h^{(l-1)})$.

Производная:

$$\frac{\partial h^{(l)}}{\partial h^{(l-1)}} = \text{diag}(\sigma'(z^{(l)})) \cdot W^{(l)}$$

Производная сигмоиды: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$ (максимум в нуле). То есть уже от активации приходит множитель $\leq 0.25$. Если $\|W\| \sim 1$, итоговый множитель $\leq 0.25$, и после 10 слоёв градиент $\leq 10^{-6}$.

Для `tanh`: $\tanh'(z) \leq 1$, немного лучше, но для насыщенных входов тоже близко к нулю.

### Для RNN

В RNN градиент через $T$ шагов времени содержит произведение $T$ одинаковых матриц $W_h$:

$$\frac{\partial h_T}{\partial h_0} \propto \prod_{t=1}^{T} W_h^\top \cdot \text{diag}(\sigma'(z_t))$$

Если спектральный радиус $\rho(W_h) < 1$ → градиент экспоненциально затухает по $T$. Vanilla RNN не может учить зависимости длиннее ~10-20 шагов именно поэтому.

## Последствия

- **Глубокие MLP** (20+ слоёв) до эпохи [[ResNet|residual connections]] практически не обучались — проигрывали более мелким сетям.
- **RNN** забывают информацию из начала последовательности.
- **Transformer** без layer norm и residual connections плохо обучается при глубине >6 слоёв.
- **RL с длинным credit assignment** — один из источников нестабильности [[PPO]]/[[GRPO]].

## Решения

### 1. Gated architectures — [[LSTM]] и GRU

[[LSTM]] решает проблему через **cell state** $c_t$, который обновляется аддитивно:

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

Градиент через cell state течёт по **аддитивному пути** без мультипликативного затухания (когда forget gate $f_t \approx 1$). Это называется **constant error carousel** — Хохрайтер специально придумал LSTM чтобы сохранить градиент на длинных последовательностях.

### 2. Residual connections (ResNet, Transformer)

$$h^{(l+1)} = h^{(l)} + F(h^{(l)})$$

Градиент:

$$\frac{\partial h^{(l+1)}}{\partial h^{(l)}} = I + \frac{\partial F}{\partial h^{(l)}}$$

Единичный член $I$ гарантирует, что градиент как минимум равен единице — даже если $\frac{\partial F}{\partial h} \to 0$. Это позволило обучать 100+ слойные CNN (ResNet) и глубокие трансформеры.

Residual stream в трансформере — основной канал протекания градиента сквозь все слои.

### 3. Batch / Layer Normalization

[[Layer Normalization]] и Batch Norm нормализуют активации, не давая им уходить в зоны насыщения (где $\sigma'(z) \approx 0$). Побочный эффект — стабилизируют градиенты.

В Transformer слой нормализации стоит **перед** (Pre-LN) или **после** (Post-LN) residual блока. Pre-LN (современный стандарт, LLaMA, GPT-2+) лучше предотвращает vanishing/exploding в глубине.

### 4. ReLU и его варианты

$$\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

**Почему ReLU помогает:**
- Производная либо 0, либо **ровно 1** — нет затухания в множителе активации.
- Градиент через активные нейроны передаётся без искажения.

**Минус:** «dying ReLU» — нейроны с отрицательным предактивационным значением получают нулевой градиент навсегда. Решается вариантами: Leaky ReLU, PReLU, ELU, GELU, SiLU/Swish.

Современные LLM используют **GELU** или **SiLU/SwiGLU** — гладкие варианты ReLU с ненулевым градиентом везде.

### 5. Careful initialization

- **Xavier/Glorot init** — для tanh/sigmoid: $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$.
- **He/Kaiming init** — для ReLU: $\text{Var}(W) = \frac{2}{n_{in}}$.

Правильная инициализация сохраняет дисперсию активаций и градиентов постоянной по глубине — градиенты не затухают и не взрываются с первого шага.

### 6. Skip connections (не только residual)

- **DenseNet** — конкатенация всех предыдущих слоёв.
- **U-Net** — skip connections между encoder и decoder.
- **Highway Networks** — gated residuals (предшественник ResNet).

Все они создают прямой путь для градиента.

## Современная ситуация

В 2024-2026 глубокие модели (LLaMA-70B = 80 слоёв, DeepSeek-V3 = 61 слой) обучаются без видимых проблем с vanishing gradient благодаря комбинации:

1. **Residual connections** в каждом блоке.
2. **Pre-LN** (RMSNorm в LLaMA).
3. **SwiGLU/GELU** активации.
4. **He/scaled init** с учётом глубины.
5. **BF16** training (больший диапазон, чем FP16).
6. **Warmup LR** schedule.

Проблема vanishing gradient **не исчезла**, но стала архитектурно решённой. Она по-прежнему возникает в:
- Vanilla RNN (поэтому их почти не используют).
- Очень глубоких сетях без residuals.
- RL с длинным credit assignment.
- Диффузионных моделях с большим числом шагов.

## Related concepts

- [[Gradient Clipping]] — обратная проблема (взрыв градиентов)
- [[LSTM]] — архитектура, явно спроектированная против vanishing
- [[Layer Normalization]] — стабилизирует активации и градиенты
- [[Attention Is All You Need|Transformer]] — residual connections обязательны
- [[Mixed Precision Training]] — underflow в FP16 усугубляет vanishing
