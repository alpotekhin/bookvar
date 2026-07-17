---
title: "RoPE"
aliases: [RoPE, Rotary Position Embedding, Rotary Positional Encoding]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/LLaMA|LLaMA]]"
  - "[[02 Areas/ML & DL/Papers/Mistral 7B|Mistral 7B]]"
  - "[[02 Areas/ML & DL/Papers/Qwen3|Qwen3]]"
courses: []
sources:
  - "[RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)](https://arxiv.org/abs/2104.09864)"
  - "[EleutherAI — Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/)"
---

# RoPE — Rotary Position Embedding

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/positional-encoding/rope-implementation.png]]
*Визуализация Rotary Position Embedding: вращение пар элементов в пространстве embedding'ов (RoFormer paper, 2104.09864)*

## Проблема: зачем нужны позиционные кодировки

[[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] обрабатывает все токены параллельно — в отличие от RNN, у него нет встроенного понятия порядка. Без позиционной информации предложения «кот сидит на коврике» и «коврике на сидит кот» дают одинаковые attention scores.

[[02 Areas/ML & DL/Concepts/Training/Positional Encoding|Позиционные кодировки]] решают эту проблему, но классические подходы имеют ограничения.

## Ограничения предшествующих подходов

### Sinusoidal PE (Vaswani et al., 2017)

Оригинальный Transformer использует фиксированные синусоидальные кодировки:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Проблемы:
- **Абсолютные, не относительные**: кодируют позицию токена $pos$, а не расстояние между токенами $m - n$. Модели приходится «вычислять» относительную позицию из двух абсолютных — лишняя нагрузка на attention
- **Плохая экстраполяция**: на длинах, не виденных при обучении, качество резко падает
- **Аддитивные**: суммируются с token embedding, смешивая семантическую и позиционную информацию

### Learned PE (GPT-2, BERT)

Обучаемые embedding'и для каждой позиции. Фиксированный максимум длины (512 для BERT, 1024 для GPT-2). За пределами обученных позиций — модель не работает.

### Relative PE (Shaw et al., 2018; T5)

Кодируют расстояние между токенами, но добавляют bias к attention logits — требуют модификации attention и не интегрируются элегантно с архитектурой.

## RoPE: ключевая идея

RoPE (Su et al., 2021) кодирует позицию через **вращение** (rotation) в пространстве embedding'ов. Вместо того чтобы складывать позиционный вектор с token embedding, RoPE **поворачивает** query и key на угол, пропорциональный позиции.

### Интуиция в 2D

Рассмотрим пару элементов вектора $(x_1, x_2)$. RoPE поворачивает этот 2D-вектор на угол $m\theta$, где $m$ — позиция токена:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

Это стандартная матрица вращения на угол $m\theta$. Токен на позиции 0 не поворачивается, на позиции 1 — на угол $\theta$, на позиции 2 — на $2\theta$, и т.д.

### Общий случай: d-мерный вектор

Для d-мерного вектора RoPE разбивает его на $d/2$ пар и вращает каждую пару с разной частотой:

$$f(x_m, m) = R_{\Theta,m} \cdot x_m$$

где $R_{\Theta,m}$ — блочно-диагональная матрица из 2D-вращений:

$$R_{\Theta,m} = \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 & & & \\ \sin m\theta_1 & \cos m\theta_1 & & & \\ & & \cos m\theta_2 & -\sin m\theta_2 & \\ & & \sin m\theta_2 & \cos m\theta_2 & \\ & & & & \ddots \end{pmatrix}$$

Частоты $\theta_i$ выбираются аналогично sinusoidal PE:

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2-1$$

Низкочастотные компоненты (большие $i$) кодируют «глобальную» позицию, высокочастотные (малые $i$) — «локальную».

### Применение в attention

RoPE применяется к query и key **перед** вычислением dot product:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(R_{\Theta,m} q_m)^T (R_{\Theta,n} k_n)}{\sqrt{d}}\right) V$$

## Ключевое свойство: относительность через вращение

Главная элегантность RoPE — dot product двух повёрнутых векторов зависит **только от относительной позиции** $m - n$:

$$(R_{\Theta,m} q_m)^T (R_{\Theta,n} k_n) = q_m^T R_{\Theta,m}^T R_{\Theta,n} k_n = q_m^T R_{\Theta, n-m} k_n$$

Это следует из свойства матриц вращения: $R_m^T R_n = R_{n-m}$.

Dot product query на позиции $m$ и key на позиции $n$ эквивалентен dot product query с key, повёрнутым на угол $(n-m)\theta$ — т.е. зависит только от расстояния между токенами. Модель получает relative position encoding «бесплатно», без дополнительных параметров или bias'ов.

### Свойство затухания

Ещё одно полезное свойство: при случайных query и key среднее значение dot product **затухает** с ростом расстояния $|m-n|$. Это создаёт естественный inductive bias к локальности — далёкие токены получают меньше attention, что соответствует языковой интуиции.

## Реализация

Эффективная реализация избегает явного матричного умножения. Используя комплексные числа:

$$f(x_m, m) = \text{Re}\left[(x_{2i} + j \cdot x_{2i+1}) \cdot e^{jm\theta_i}\right]$$

где $e^{jm\theta_i} = \cos(m\theta_i) + j\sin(m\theta_i)$. На практике это сводится к поэлементным операциям:

```python
def apply_rotary_emb(x, freqs_cos, freqs_sin):
    # x: [batch, seq_len, n_heads, head_dim]
    x_r = x[..., ::2]   # чётные элементы
    x_i = x[..., 1::2]  # нечётные элементы
    # Rotation:
    out_r = x_r * freqs_cos - x_i * freqs_sin
    out_i = x_r * freqs_sin + x_i * freqs_cos
    # Interleave back
    return torch.stack([out_r, out_i], dim=-1).flatten(-2)
```

Вычислительная стоимость — $O(d)$ per token, пренебрежимо мала по сравнению с attention ($O(n \cdot d)$).

## Расширение контекста: за пределы обученной длины

RoPE обучена на фиксированной максимальной длине (например, 4096 для LLaMA). На позициях $m > L_{train}$ углы вращения выходят за виденный при обучении диапазон — quality degrades. Это ключевая проблема для long-context моделей.

### Position Interpolation (PI)

Линейно масштабируем позиции, чтобы «вместить» длинный контекст в обученный диапазон:

$$m' = m \cdot \frac{L_{train}}{L_{target}}$$

Для расширения с 4K до 32K: позиция 32000 → $32000 \cdot \frac{4096}{32768} = 4000$. Требуется короткий дообучающий SFT (1000 steps). Проблема: теряется resolution на коротких дистанциях — расстояние между соседними токенами уменьшается в $L_{target}/L_{train}$ раз.

### NTK-aware Scaling

Вместо масштабирования позиций, масштабирует **базу** $\theta$:

$$\theta_i' = \left(\alpha \cdot 10000\right)^{-2i/d}$$

где $\alpha$ — scaling factor. Это растягивает частоты, особенно низкочастотные компоненты, сохраняя высокочастотные (локальные) почти неизменными. Интуиция: для длинного контекста нужно изменить «глобальные» частоты, а «локальные» (различение соседних токенов) должны остаться.

### Dynamic NTK

Адаптивная версия: $\alpha$ увеличивается только когда текущая длина превышает $L_{train}$. На коротких последовательностях — стандартный RoPE, на длинных — масштабированный. Используется в Qwen.

### YaRN (Yet another RoPE extensioN)

Комбинирует NTK-aware scaling с attention scaling и temperature factor. Разделяет dimensions на три группы:
- **Высокочастотные** — не масштабируются (локальная информация)
- **Низкочастотные** — масштабируются по NTK
- **Промежуточные** — интерполяция

YaRN с коротким fine-tuning расширяет контекст LLaMA 2 с 4K до 128K с минимальной потерей качества.

### Сравнение методов расширения

| Метод | Идея | Fine-tuning | Качество на коротких | Качество на длинных |
|-------|------|-------------|---------------------|---------------------|
| Position Interpolation | Сжать позиции | ~1000 steps | Небольшое ухудшение | Хорошее |
| NTK-aware | Растянуть базу | Минимальный | Сохраняется | Хорошее |
| Dynamic NTK | Адаптивная база | Не нужен | Без изменений | Приемлемое |
| YaRN | Гибрид + scaling | ~400 steps | Сохраняется | Отличное |

## Где используется RoPE

RoPE стал **стандартом де-факто** для позиционного кодирования в современных LLM:

- **LLaMA 1/2/3** (Meta) — RoPE с $\theta_{base} = 10000$ (LLaMA 1/2), $500000$ (LLaMA 3 для 128K контекста)
- **Mistral/Mixtral** (Mistral AI) — RoPE + [[02 Areas/ML & DL/Concepts/Architectures/Sliding Window Attention|Sliding Window Attention]]
- **Qwen/Qwen2/Qwen3** (Alibaba) — RoPE с Dynamic NTK для long-context
- **DeepSeek-V3** — RoPE с YaRN
- **Gemma** (Google) — RoPE
- **Falcon** — RoPE

Практически все decoder-only LLM, выпущенные после 2023 года, используют RoPE. Исключение — модели на альтернативных архитектурах ([[02 Areas/ML & DL/Concepts/Architectures/Mamba|Mamba]], [[02 Areas/ML & DL/Concepts/Architectures/RWKV|RWKV]]), которым позиционные кодировки не нужны (рекуррентность даёт позиционность бесплатно).

## Почему RoPE победил: преимущества

1. **Относительность без overhead**: relative position encoding через dot product, не через bias — не нужно менять attention kernel
2. **Нулевые обучаемые параметры**: вся позиционная информация — детерминированная функция позиции. Нечему переобучаться
3. **Совместимость с Flash Attention**: RoPE применяется к Q, K до входа в attention kernel — Flash Attention работает без модификаций
4. **Extensibility**: чётко определённая частотная структура позволяет масштабировать контекст через модификацию частот (NTK, YaRN)
5. **Inductive bias к локальности**: естественное затухание attention с расстоянием помогает модели фокусироваться на релевантном контексте

### Ограничения RoPE

- **Обученная длина — жёсткий потолок** без extension methods. На $L > L_{train}$ высокочастотные компоненты начинают «алиаситься»
- **Не учитывает 2D/3D структуру** — разработан для 1D последовательностей. Для vision transformers нужны модификации (2D RoPE в PaLI)
- **Низкочастотные компоненты медленно меняются** — на коротких дистанциях (< 10 токенов) несколько dimensions несут мало позиционной информации

## RoPE vs ALiBi

ALiBi (Attention with Linear Biases, Press et al., 2022) — альтернативный подход: добавляет линейный bias к attention scores, пропорциональный расстоянию между токенами. Не требует fine-tuning для экстраполяции, но:

- На практике RoPE + context extension (YaRN) даёт лучшее качество на длинных контекстах
- ALiBi имеет слишком агрессивное затухание — «забывает» далёкие токены
- RoPE интегрируется с Flash Attention без overhead, ALiBi требует модификации attention

Именно поэтому community сконвергировал на RoPE.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Positional Encoding|Positional Encoding]] — общий класс методов, RoPE — конкретная реализация
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, в которой применяется RoPE
- [[02 Areas/ML & DL/Concepts/Architectures/Self-Attention|Self-Attention]] — механизм, к query и key которого применяется вращение
