---
title: "Dropout"
aliases: [Dropout]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Attention Is All You Need|Attention Is All You Need]]"
  - "[[02 Areas/ML & DL/Papers/BERT|BERT]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/Lecture 04 — Backpropagation|CS224N L04]]"
sources:
  - "[Srivastava et al. — Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)](https://jmlr.org/papers/v15/srivastava14a.html)"
---

# Dropout

## Проблема: переобучение в больших сетях

Нейронные сети с миллионами параметров склонны к **переобучению** (overfitting) — запоминанию training data вместо обобщения. Классические методы регуляризации (L2, early stopping) помогают, но недостаточно для очень глубоких сетей.

**Dropout** (Srivastava et al., 2014) — простой, но мощный метод регуляризации: **случайное обнуление нейронов во время обучения**.

## Механизм работы

### Training time

На каждом forward pass каждый нейрон **с вероятностью $p$ обнуляется** (dropout rate). Для слоя $\mathbf{h}$:

$$\mathbf{m} \sim \text{Bernoulli}(1 - p)$$
$$\tilde{\mathbf{h}} = \mathbf{m} \odot \mathbf{h}$$

где $\mathbf{m}$ — бинарная маска, $\odot$ — поэлементное умножение. На каждом батче маска генерируется заново — разные нейроны отключаются каждый раз.

### Inference time

При инференсе все нейроны активны, но выход масштабируется на $(1 - p)$:

$$\hat{\mathbf{h}} = (1 - p) \cdot \mathbf{h}$$

Это необходимо для сохранения **expected value** — при обучении в среднем $(1 - p)$ нейронов активны, поэтому при инференсе нужно уменьшить значения на тот же фактор.

### Inverted Dropout (стандартная реализация)

На практике используется **inverted dropout** — масштабирование происходит **при обучении**, а не при инференсе:

$$\tilde{\mathbf{h}} = \frac{\mathbf{m} \odot \mathbf{h}}{1 - p}$$

Преимущество: при инференсе никаких изменений не нужно — forward pass идентичен обычному. Именно так реализован `nn.Dropout` в PyTorch:

```python
# PyTorch автоматически масштабирует на 1/(1-p) при training
# и ничего не делает при eval
self.dropout = nn.Dropout(p=0.1)
```

## Почему Dropout работает: неявный ансамбль

Ключевая интуиция: dropout при обучении **неявно создаёт ансамбль** из экспоненциально большого числа подсетей.

Сеть с $n$ нейронами и dropout имеет $2^n$ возможных подсетей (каждый нейрон либо включен, либо выключен). На каждом батче обучается одна случайная подсеть. При инференсе все нейроны активны — это аппроксимация **среднего предсказания всех подсетей**.

Это объясняет регуляризирующий эффект: отдельные нейроны не могут полагаться на конкретные «партнёрские» нейроны, потому что те могут быть отключены. Каждый нейрон вынужден учить **устойчивые признаки**, полезные независимо от контекста.

## Dropout в Transformer

В архитектуре [[02 Areas/ML & DL/Papers/Attention Is All You Need|Transformer]] dropout применяется в **трёх местах**:

### 1. Attention Dropout

После вычисления attention weights (softmax), но до умножения на values:

$$\text{Attention}(Q, K, V) = \text{dropout}\left(\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)\right) V$$

Это заставляет модель не полагаться на одну позицию — внимание распределяется более равномерно.

### 2. Residual Dropout

После каждого подслоя (self-attention, FFN), но перед residual connection:

$$\text{output} = \text{LayerNorm}(x + \text{dropout}(\text{sublayer}(x)))$$

### 3. Embedding Dropout

После embedding layer (token embeddings + positional embeddings):

$$\text{emb} = \text{dropout}(E_{token} + E_{pos})$$

### Типичные значения

| Контекст | Dropout rate |
|----------|-------------|
| Оригинальная статья (FC layers) | 0.5 |
| Transformer (Vaswani et al.) | 0.1 |
| BERT-base | 0.1 |
| BERT-large | 0.1 |
| GPT-2 | 0.1 |
| Практика для Transformers | 0.1 - 0.3 |

Transformer'ы используют значительно меньший dropout (0.1) по сравнению с оригинальной рекомендацией (0.5), потому что residual connections, layer normalization и большой объём данных уже обеспечивают достаточную регуляризацию.

## Dropout в эпоху LLM: почти не используется

Парадоксальный факт: **современные LLM при pre-training не используют dropout**. LLaMA, Mistral, Qwen, Falcon — все устанавливают $p = 0$.

### Почему dropout не нужен при pre-training

1. **Data scale решает проблему переобучения.** При обучении на триллионах токенов модель видит каждый пример ~1 раз (single epoch или меньше). Переобучение возникает при многократном показе данных — здесь этого нет.

2. **Dropout вредит throughput.** Генерация и применение масок — дополнительный compute. При масштабах pre-training (тысячи GPU, месяцы обучения) даже 5% overhead — значительная стоимость.

3. **Dropout мешает параллелизму.** Tensor parallelism и pipeline parallelism требуют детерминированного поведения для корректной синхронизации. Случайные маски усложняют это.

4. **Эмпирические результаты.** Chinchilla scaling laws показали, что при оптимальном соотношении модель/данные dropout не улучшает результат.

### Dropout при fine-tuning

При fine-tuning (SFT, LoRA) dropout **снова полезен**, потому что:
- Датасет маленький (тысячи-десятки тысяч примеров)
- Модель обучается несколько эпох
- Риск переобучения высок

Типичные значения при [[02 Areas/ML & DL/Concepts/Training/SFT|SFT]]:
- Attention dropout: 0.05 - 0.1
- В LoRA: `lora_dropout=0.05` (dropout в low-rank matrices)

## Варианты и расширения

### DropConnect

Вместо обнуления активаций обнуляются **веса**:

$$\tilde{W} = \mathbf{m} \odot W$$

Более fine-grained, но дороже вычислительно.

### Spatial Dropout (2D Dropout)

Для CNN: обнуляется весь feature map (канал), а не отдельные пиксели. Это обеспечивает пространственную корреляцию в маске.

### DropPath (Stochastic Depth)

Обнуляется **весь residual branch** с некоторой вероятностью:

$$\text{output} = x + m \cdot \text{sublayer}(x), \quad m \sim \text{Bernoulli}(1 - p)$$

Используется в Vision Transformers (ViT, Swin) и некоторых LLM. По сути — dropout на уровне целого слоя, а не нейрона.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — этап, где современные LLM dropout не используют
- [[02 Areas/ML & DL/Concepts/Training/SFT|SFT]] — этап, где dropout полезен
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — dropout как регуляризация при дообучении
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] — attention dropout внутри механизма внимания
