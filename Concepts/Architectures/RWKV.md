---
title: "RWKV"
aliases: [RWKV, Receptance Weighted Key Value, RWKV-4, RWKV-LM]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/RWKV|RWKV]]"
courses: []
sources:
  - "[Hugging Face — Introducing RWKV](https://huggingface.co/blog/rwkv)"
  - "[Full Stack Deep Learning — RWKV, Explained](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/)"
  - "[RWKV Wiki — Architecture History](https://wiki.rwkv.com/basic/architecture.html)"
  - "[Zhai et al. — Attention Free Transformer (AFT)](https://arxiv.org/abs/2105.14103)"
---

# RWKV

## Зачем это нужно: RNN vs. Transformer — вечный компромисс

| Свойство | RNN | Transformer |
|----------|-----|-------------|
| Training | $O(n)$ по памяти, но **не параллелизуется** | $O(n^2)$ compute, но полная параллелизация |
| Inference | $O(1)$ на шаг, $O(d)$ по памяти | $O(n)$ на шаг (KV-cache), $O(n)$ по памяти |
| Длинные seq | Линейная сложность | Квадратичная сложность |
| Качество | Vanishing gradients, теряет дальний контекст | Прямой доступ к любому токену |

До 2023 считалось, что нужно **выбирать**: или параллельное обучение Transformer (быстро), или эффективный инференс RNN (дёшево). RWKV (Peng et al., 2023, EleutherAI) показал, что **можно совместить оба**: обучать как Transformer (параллельно по времени), а инференс делать как RNN ($O(1)$ на шаг).

RWKV — **крупнейшая dense RNN** когда-либо обученная (14B параметров, 330B токенов на The Pile). Сопоставима по качеству с Transformer-моделями аналогичного размера.

## Ключевые элементы: R, W, K, V

Название RWKV — аббревиатура четырёх фундаментальных компонентов:

- **R (Receptance)** — вектор-рецептор прошлой информации. Sigmoid gate, определяющий, сколько прошлого контекста пропустить
- **W (Weight)** — trainable positional decay vector. Channel-wise экспоненциальное затухание по relative position
- **K (Key)** — аналог Key в стандартном attention
- **V (Value)** — аналог Value в стандартном attention

## Архитектура: два sub-блока

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rwkv/rwkv-architecture.png]]
*Элементы RWKV блока (слева) и полная архитектура RWKV-LM (справа). Каждый residual block содержит Time-Mixing и Channel-Mixing sub-блоки (источник: оригинальная статья)*

Модель — стек одинаковых residual-блоков. Каждый блок содержит два sub-блока:

### Time-Mixing (аналог self-attention)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rwkv/rwkv-lm-architecture.png]]
*RWKV для language modeling: стек блоков с LayerNorm, Time-Mixing, Channel-Mixing и финальным LM Head (источник: оригинальная статья)*

**Token Shift** — первая ключевая идея. Все линейные проекции получают **смесь текущего и предыдущего токена**:

$$r_t = W_r \cdot (\mu_r \odot x_t + (1 - \mu_r) \odot x_{t-1})$$
$$k_t = W_k \cdot (\mu_k \odot x_t + (1 - \mu_k) \odot x_{t-1})$$
$$v_t = W_v \cdot (\mu_v \odot x_t + (1 - \mu_v) \odot x_{t-1})$$

где $\mu_r, \mu_k, \mu_v$ — learnable interpolation coefficients. Простейший трюк: `nn.ZeroPad2d((0,0,1,-1))` в PyTorch. Это даёт модели доступ к **биграмной** информации без attention.

**WKV Operator** — сердце RWKV. Вычисляет взвешенную сумму values с channel-wise exponential decay:

$$\text{wkv}_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} \odot v_i + e^{u+k_t} \odot v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w + k_i} + e^{u+k_t}}$$

где $w \in (\mathbb{R}_{\geq 0})^d$ — trainable decay (non-negative, чтобы $e^{-w} \leq 1$), $u$ — отдельный вектор для текущего токена.

Это модификация Attention Free Transformer (AFT, Zhai et al., 2021). Отличие от AFT: вместо pairwise learned biases $w_{t,i}$ — channel-wise decay $-(t-i)w$, зависящий только от **relative position**.

**Output Gating:**

$$o_t = W_o \cdot (\sigma(r_t) \odot \text{wkv}_t)$$

Sigmoid от receptance $r_t$ — финальный gate, определяющий, какая информация из WKV попадёт в выход.

### Channel-Mixing (аналог FFN)

$$o'_t = \sigma(r'_t) \odot (W'_v \cdot \max(k'_t, 0)^2)$$

Squared ReLU activation (So et al., 2021). Receptance gate $\sigma(r'_t)$ контролирует пропускание. Token shift применяется и здесь.

## Два режима: параллельный и рекуррентный

### Time-Parallel Mode (обучение)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rwkv/rwkv-formula.png]]
*RWKV attention формула: WKV оператор, комбинирующий exponential decay с key-value взвешиванием (источник: Hugging Face)*

Сложность одного слоя: $O(BTd^2)$ — доминируют матричные умножения $W_{\lambda}$ (аналогично $W_Q, W_K, W_V, W_O$ в Transformer). WKV update — $O(BTd)$, что **пренебрежимо** по сравнению с линейными проекциями.

Матричные умножения параллелизуются так же, как в Transformer. WKV вычисление time-dependent, но параллелизуется по batch и channel dimensions.

### Time-Sequential Mode (инференс)

При авторегрессивном декодировании RWKV работает как классическая RNN:

$$s_t = e^{-w} \odot s_{t-1} + e^{k_t} \odot v_t$$
$$\text{wkv}_t = \frac{s_t}{...}$$

Каждый шаг: $O(d)$ по compute, $O(d)$ по памяти. **Не нужен KV-cache** — всё состояние в одном vector.

### Сравнение сложности

| Модель | Training time | Training space | Inference time/step | Inference space |
|--------|--------------|----------------|-------------------|----------------|
| Transformer | $O(T^2 d)$ | $O(T^2 + Td)$ | $O(Td)$ (KV-cache) | $O(Td)$ |
| Linear Transformer | $O(Td^2)$ | $O(Td + d^2)$ | $O(d^2)$ | $O(d^2)$ |
| **RWKV** | **$O(Td)$** | **$O(d)$** | **$O(d)$** | **$O(d)$** |

RWKV — **минимальная сложность** среди всех сравниваемых архитектур.

## Scaling Laws: RNN масштабируется как Transformer

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rwkv/rwkv-scaling-comparison.png]]
*RWKV показывает сопоставимую производительность с Transformer-моделями (Pythia, OPT, BLOOM) при одинаковом compute (источник: оригинальная статья)*

45 моделей RWKV обучены для проверки scaling laws. Результат: RWKV следует **тем же log-log linear scaling laws**, что и Transformers ($r^2 = 0.994$). Это опровергает утверждение Kaplan et al. (2020), что LSTM не следует scaling laws Transformer.

### Модели

| Модель | Layers | Dim | Params | FLOP/token |
|--------|--------|-----|--------|-----------|
| 169M | 12 | 768 | 169M | 261M |
| 430M | 24 | 1024 | 430M | 757M |
| 1.5B | 24 | 2048 | 1.5B | 2.8B |
| 3B | 32 | 2560 | 3.0B | 5.7B |
| 7B | 32 | 4096 | 7.4B | 14.4B |
| **14B** | **40** | **5120** | **14.2B** | **27.8B** |

Все модели обучены на 330B токенов (1 epoch The Pile).

### Zero-Shot сравнение с Transformers

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rwkv/rwkv-eval.png]]
*Zero-shot performance RWKV vs. Transformer-моделей на NLP бенчмарках (источник: Hugging Face)*

RWKV показывает сопоставимые результаты с Pythia, OPT и BLOOM на FLOP-matched основе. На ARC-Challenge и HellaSwag — на уровне или чуть ниже Transformer. На LAMBADA — иногда превосходит.

## Инференс: линейный скейлинг

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/rwkv/rwkv-inference-time.png]]
*Cumulative time на text generation: RWKV линеен, Transformer — суперлинеен из-за растущего KV-cache (источник: оригинальная статья)*

Практически: **int8 RWKV-14B** работает на последовательностях **любой длины** при **3GB VRAM**. Для Transformer той же размерности потребовалось бы 10x+ больше памяти при длинных контекстах.

## Дополнительные оптимизации

### Custom CUDA Kernel
WKV вычисление — единственная не-параллелизуемая часть. Custom CUDA kernel объединяет все операции в один kernel call, избегая накладных расходов PyTorch scheduler.

### Small Init Embedding
Embedding матрица инициализируется **маленькими значениями** + дополнительный LayerNorm. Это ускоряет сходимость: модель быстро уходит от шумного начального состояния. Критически важно для обучения глубоких архитектур с post-LN.

### Custom Initialization
Большинство весов инициализированы нулями. Линейные слои без bias. Принципы из He et al. (2016) и AlphaFold (Jumper et al., 2021) — identity-like mapping с нарушением симметрии для установления чёткого information flow.

## Ограничения: цена линейного attention

### Сжатие контекста в один вектор
Всё прошлое compressed в state vector размера $d$. В отличие от Transformer, который хранит **полный KV-cache** (все прошлые токены), RWKV **необратимо сжимает** информацию. Это фундаментально ограничивает **exact recall** из длинного контекста.

### Чувствительность к prompt engineering
Экстремальная зависимость от формулировки промпта. F1 score: от **44.2% до 74.8%** при переупорядочении информации в промпте. Причина: линейное attention не может эффективно «вернуться» к ранним токенам — информация, попавшая в начало промпта, может быть затёрта более поздними токенами.

### Extended Context
RWKV обучался на 1024 токенов, но может быть дообучен на более длинные контексты через progressive увеличение:
- 1024 → 2048 (10B tokens)
- 2048 → 4096 (100B tokens)
- 4096 → 8192 (100B tokens)

Test loss на The Pile **монотонно уменьшается** с ростом контекста — RWKV может эффективно использовать длинный контекст.

## RWKV vs. другие Linear-Time модели

| Архитектура | Training | Inference | Performance |
|-------------|----------|-----------|-------------|
| Transformer | Параллельно | $O(n)$ KV-cache | Лучшее |
| Linear Transformer | Параллельно | $O(1)$ | Слабое |
| RWKV | Параллельно (по batch/channel) | $O(1)$ | Сопоставимо с Transformer |
| [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] | Параллельно (parallel scan) | $O(1)$ | Превосходит Transformer |
| [[02 Areas/ML & DL/Concepts/Architectures/RetNet|RetNet]] | Параллельно | $O(1)$ | Сопоставимо с Transformer |

RWKV был **первой** моделью, доказавшей на масштабе 14B, что RNN-подход жизнеспособен. Mamba позже превзошла RWKV по quality (благодаря selection mechanism), но RWKV остаётся проще в реализации.

## Хронология RWKV

| Версия | Год | Ключевое изменение |
|--------|-----|-------------------|
| RWKV-1,2,3 | 2021-2022 | Ранние итерации |
| **RWKV-4** | **2023** | **Статья, модели до 14B, The Pile** |
| RWKV-5 (Eagle) | 2024 | Multi-headed attention-like variant |
| RWKV-6 (Finch) | 2024 | Data-dependent linear recurrence |
| RWKV-7 (Goose) | 2025 | Дальнейшие улучшения |

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, которую RWKV стремится заменить
- [[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]] — конкурирующий linear-time подход (SSM-based)
- [[02 Areas/ML & DL/Concepts/Architectures/RetNet|RetNet]] — retention mechanism, третий linear-time кандидат
- [[02 Areas/ML & DL/Concepts/NLP/RNN/RNN|RNN]] — фундамент, на котором строится RWKV
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — WKV оператор как альтернатива attention

## Дополнительные ресурсы

- [Hugging Face — Introducing RWKV](https://huggingface.co/blog/rwkv) — лучшее введение с визуализациями
- [Full Stack Deep Learning — RWKV, Explained](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/) — технический deep dive
- [RWKV Wiki](https://wiki.rwkv.com/) — официальная документация и roadmap
- [GitHub — BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM) — официальная реализация
- [Hugging Face — RWKV models](https://huggingface.co/RWKV) — pretrained модели
