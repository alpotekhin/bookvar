---
title: "Distillation"
aliases: [Distillation, Knowledge Distillation, KD]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Phi-2|Phi-2]]"
  - "[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]]"
  - "[[02 Areas/ML & DL/Papers/Gemma|Gemma]]"
courses: []
sources:
  - "[Hinton et al. — Distilling the Knowledge in a Neural Network (2015)](https://arxiv.org/abs/1503.02531)"
---

# Distillation — Knowledge Distillation

## Идея: передача знаний от большой модели к маленькой

Knowledge Distillation (KD) — метод обучения, при котором **маленькая модель (student)** учится воспроизводить поведение **большой модели (teacher)**. Вместо обучения на ground truth labels student обучается на выходах teacher — soft labels, которые несут больше информации, чем binary/one-hot разметка.

**Ключевой инсайт Хинтона (2015):** выходное распределение обученной модели содержит «тёмные знания» (dark knowledge). Например, для изображения цифры «7» teacher может выдать: P(7) = 0.9, P(1) = 0.05, P(2) = 0.03, P(9) = 0.02. Эти soft probabilities показывают, что «7» визуально ближе к «1» и «9», чем к «0» или «4» — информация, которой нет в hard label «7».

## Soft Labels и Temperature Scaling

### Temperature в Softmax

Стандартный softmax для logits $z_i$:

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

С temperature $T$:

$$p_i^{(T)} = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

| Temperature | Эффект | Распределение |
|------------|--------|---------------|
| $T = 1$ | Стандартный softmax | Peaked — один класс доминирует |
| $T > 1$ | «Размягчение» | Smoother — различия между классами виднее |
| $T \to \infty$ | Uniform | Все классы равновероятны |
| $T \to 0$ | Argmax | Вся масса на одном классе |

При $T = 1$ teacher может выдать [0.98, 0.01, 0.01] — student почти не видит разницы между вторым и третьим классом. При $T = 5$ те же logits дают [0.45, 0.28, 0.27] — структура dark knowledge становится явной.

Типичные значения: $T = 2\text{–}20$. В оригинальной работе Хинтона $T = 20$ для MNIST.

## Loss Function

Distillation loss — комбинация двух компонентов:

$$\mathcal{L}_{KD} = \alpha \cdot T^2 \cdot D_{KL}(p^{(T)}_{teacher} \| p^{(T)}_{student}) + (1 - \alpha) \cdot \mathcal{L}_{CE}(y, p^{(1)}_{student})$$

где:
- $D_{KL}(p^{(T)}_{teacher} \| p^{(T)}_{student})$ — KL-дивергенция между soft labels teacher и student при temperature $T$
- $\mathcal{L}_{CE}(y, p^{(1)}_{student})$ — cross-entropy с ground truth labels при $T = 1$
- $\alpha$ — вес distillation loss (обычно 0.5–0.9)
- $T^2$ — коррекция масштаба градиентов (при высоком $T$ gradient magnitudes уменьшаются пропорционально $1/T^2$)

### Почему KL, а не MSE

KL-дивергенция — естественная мера расстояния между распределениями. MSE на logits тоже работает (и в некоторых случаях даже лучше), но KL позволяет student «игнорировать» классы, к которым teacher безразличен, фокусируясь на информативных различиях.

## Типы Distillation для LLM

### 1. Response-based Distillation (самый распространённый)

Student обучается воспроизводить **output distribution** teacher на token level:

$$\mathcal{L} = D_{KL}(P_{teacher}(\cdot | x, y_{<t}) \| P_{student}(\cdot | x, y_{<t}))$$

Это standard KD, адаптированный для autoregressive generation — matching next-token distributions.

### 2. Synthetic Data Distillation

Teacher генерирует **training data**, а student обучается на нём через обычный [[02 Areas/ML & DL/Concepts/Training/SFT|SFT]]:

```
Teacher → генерирует высококачественные ответы → dataset
Student → SFT на этом dataset
```

Это самый простой и масштабируемый тип distillation. Не требует доступа к logits teacher — достаточно API.

**Примеры:**
- **[[02 Areas/ML & DL/Papers/Phi-2|Phi-2]]** (Microsoft): 1.3B модель, обученная на synthetic data от GPT-4. Конкурирует с моделями 10–25x больше.
- **Orca** (Microsoft): systematic instruction-following data от GPT-4 с chain-of-thought explanations.
- **Alpaca** (Stanford): 52K instructions, сгенерированных GPT-3.5, для fine-tuning LLaMA 7B.

### 3. Reasoning Distillation (DeepSeek-R1)

[[02 Areas/ML & DL/Papers/DeepSeek-R1 Reasoning via RL|DeepSeek-R1]] ввёл **reasoning distillation** — передачу не просто ответов, но **цепочек рассуждений** от teacher к student:

```
Teacher (R1-671B) → генерирует <think>...рассуждение...</think><answer>ответ</answer>
Student (R1-Distill-*) → SFT на этих рассуждениях
```

Результаты R1-Distill моделей на математических бенчмарках (AIME 2024):

| Модель | Params | AIME 2024 |
|--------|--------|-----------|
| DeepSeek-R1 (teacher) | 671B | 79.8% |
| R1-Distill-Qwen-32B | 32B | 72.6% |
| R1-Distill-Qwen-14B | 14B | 69.7% |
| R1-Distill-Qwen-7B | 7B | 55.5% |
| R1-Distill-Qwen-1.5B | 1.5B | 28.9% |

Ключевое наблюдение: reasoning distillation для малых моделей даёт **лучшие результаты, чем прямое RL-обучение** (GRPO). Student на 7B через distillation превзошёл 7B модель, обученную через RL с нуля. Но для больших моделей (32B+) RL даёт больше.

### 4. Feature-based Distillation

Student учится воспроизводить **internal representations** teacher (hidden states, attention patterns):

$$\mathcal{L} = \sum_l \| f(h^{student}_l) - h^{teacher}_l \|^2$$

где $f$ — проекция (student и teacher могут иметь разные hidden dimensions). Этот подход реже используется для LLM из-за различий в архитектурах и больших compute costs.

## Когда Distillation лучше обучения с нуля

| Ситуация | Distillation помогает | Причина |
|----------|----------------------|---------|
| Мало данных | Да | Teacher «расширяет» датасет через soft labels |
| Student значительно меньше teacher | Да | Dark knowledge компенсирует capacity gap |
| Нужен reasoning | Да | Reasoning traces — готовые «рецепты» мышления |
| Задача совпадает с teacher capabilities | Да | Teacher хорошо решает задачу → хороший training signal |
| Student ≈ teacher по размеру | Нет | Мало выгоды, лучше обучать напрямую |
| Задача вне компетенции teacher | Нет | Плохой teacher → плохой training signal |

### Capacity Gap Problem

Если student слишком мал по сравнению с teacher, distillation может не работать: student не имеет достаточной expressiveness, чтобы воспроизвести поведение teacher. Решение — **multi-step distillation**:

```
Teacher (70B) → Assistant Teacher (13B) → Student (1.5B)
```

Каждый шаг — более «мягкая» передача знаний.

## Ограничения

**Потолок качества.** Student ограничен качеством teacher — distillation не создаёт новых знаний, а только сжимает существующие.

**Distributional mismatch.** Teacher обучался на других данных и может иметь biases, которые передаются student. Synthetic data от GPT-4 наследует GPT-4 biases.

**Лицензионные ограничения.** Многие API terms (OpenAI, Anthropic) запрещают использование outputs для обучения конкурирующих моделей. Это ограничивает synthetic data distillation.

**Faithfulness.** Student может обучиться имитировать формат ответов teacher без реального понимания — «shallow mimicry». Особенно заметно при reasoning distillation: student воспроизводит паттерн `<think>...</think>`, но рассуждения внутри неконсистентны.

## Distillation в индустрии 2024–2025

Synthetic data distillation стал **основным методом обучения маленьких моделей**:

- **Google ([[02 Areas/ML & DL/Papers/Gemma|Gemma]])** — synthetic data от Gemini для обучения open-source моделей
- **Microsoft (Phi серия)** — systematic distillation от GPT-4 для ultra-small моделей
- **DeepSeek** — reasoning distillation для серии R1-Distill
- **Meta (LLaMA 3.2 1B/3B)** — distillation от LLaMA 3.1 8B/70B

Тренд: distillation всё чаще рассматривается не как «хак», а как **стандартная стадия pipeline** — наряду с [[02 Areas/ML & DL/Concepts/Training/Pre-training|pre-training]] и [[02 Areas/ML & DL/Concepts/Training/SFT|SFT]].

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/SFT|SFT]] — fine-tuning, через который чаще всего реализуется distillation
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — стадия, после которой применяется distillation
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — distillation позволяет обходить scaling laws для малых моделей
- [[02 Areas/ML & DL/Concepts/Training/Synthetic Data|Synthetic Data]] — ключевой механизм modern distillation
