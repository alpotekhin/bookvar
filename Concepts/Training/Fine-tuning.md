---
title: "Fine-tuning"
aliases: [task-specific fine-tuning, downstream fine-tuning, PEFT, parameter-efficient fine-tuning]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
  - "[[02 Areas/ML & DL/Papers/T5]]"
  - "[[02 Areas/ML & DL/Papers/LoRA|LoRA]]"
  - "[[02 Areas/ML & DL/Papers/QLoRA|QLoRA]]"
courses: []
sources:
  - "[Sebastian Raschka — Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)"
  - "[Encora — Comparing Fine-Tuning Optimization Techniques](https://www.encora.com/interface/comparing-fine-tuning-optimization-techniques-lora-qlora-dora-and-qdora)"
---

# Fine-tuning

## Зачем это нужно: от general representations к конкретной задаче

Pre-trained LLM — это «универсальная модель языка». Она знает грамматику, факты, умеет рассуждать. Но она не знает, что от неё хотят *конкретно*: классифицировать тексты? Извлекать сущности? Следовать инструкциям? Отвечать на вопросы в стиле компании?

Fine-tuning — адаптация pre-trained модели к downstream задаче через продолжение gradient updates на task-specific данных. Инициализация = pre-trained веса; обновляются все параметры (full fine-tuning) или подмножество (PEFT).

Это **вторая фаза** парадигмы «[[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-train]] → Fine-tune», которая полностью изменила NLP с 2018 года.

## Эволюция парадигм fine-tuning

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/fine-tuning/finetuning-approaches.png]]
*Три подхода к fine-tuning: feature-based (замороженная модель + классификатор), output-layer tuning (только верхний слой), full fine-tuning (все параметры). С 2021+ добавились PEFT-методы (LoRA, adapters), а с 2022+ — prompting без fine-tuning вообще (источник: Sebastian Raschka)*

## Full Fine-Tuning: классический подход

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/fine-tuning/regular-finetuning.png]]
*Full Fine-Tuning: все веса модели обновляются через backpropagation. Для больших моделей (>10B) требует значительных ресурсов (источник: Sebastian Raschka)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig1.png]]
*LoRA: вместо обновления полной матрицы $W$ обучаются только низкоранговые матрицы $A$ и $B$, которые при инференсе сливаются с оригинальными весами (источник: Sebastian Raschka)*

### Как это работает

После pre-training добавляется task-специфичный head (classification, token labeling, generation), и **все параметры** модели обновляются через backpropagation:

| Задача | Head | Входной вектор |
|--------|------|---------------|
| Text classification | Linear + softmax | `[CLS]` вектор (BERT) |
| Token classification (NER) | Linear per-token | Каждый $T_i$ |
| Span QA (SQuAD) | Start + End vectors $S, E$ | $\text{argmax}(S \cdot T_i)$ и $\text{argmax}(E \cdot T_i)$ |
| Seq2Seq (summarization) | Decoder | Encoder output |
| Instruction following | Нет (LM head) | Авторегрессивно |

### BERT fine-tuning рецепт (Devlin et al., 2019)

Из [[02 Areas/ML & DL/Papers/BERT]] Section 4 — рецепт, ставший стандартом:

- Batch size: 32
- Epochs: 3
- Learning rate: grid search из {5e-5, 4e-5, 3e-5, 2e-5}
- Warm-up: первые 10% steps, затем linear decay
- Dropout: 0.1

**Важно:** этот рецепт разработан для encoder-only моделей (BERT, RoBERTa) на задачах NLU с ~100K-1M примеров. Для LLM рецепт другой.

### Проблема нестабильности при малых датасетах

Из [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]] — три источника нестабильности fine-tuning BERT на маленьких датасетах:

**Источник 1: BERTAdam без bias correction**

Оригинальная реализация BERT (и HuggingFace до июля 2019) пропускала bias correction в Adam. Это приводит к overestimation learning rate в начале обучения → **48% degenerate runs** на RTE (accuracy < 55%).

Исправление: использовать `torch.optim.AdamW` с bias correction.

**Источник 2: плохая инициализация верхних слоёв**

Верхние слои BERT специализированы под MLM pre-training — для downstream задачи их лучше **re-initialize** из $\mathcal{N}(0, 0.02^2)$. Результат: +3.1% на RTE.

**Источник 3: слишком мало шагов**

На 1K примерах 3 эпохи = 96 шагов — недостаточно. Увеличение до 3200 steps:
- MRPC: 80.5% → 86.0%
- MNLI: 52.2% → 68.8%

## Parameter-Efficient Fine-Tuning (PEFT): обзор методов

С ростом моделей до десятков и сотен миллиардов параметров full fine-tuning стал непрактичным. PEFT методы обновляют **<1% параметров**, сохраняя 90-100% качества.

### Сравнительная таблица методов

| Метод | Обучаемые параметры | VRAM | Inference overhead | Качество vs Full FT |
|-------|--------------------|----- |--------------------|---------------------|
| **Full Fine-Tuning** | 100% | 12 байт/param | Нет | 100% (baseline) |
| **[[02 Areas/ML & DL/Concepts/Training/LoRA\|LoRA]]** | 0.01-1% | ~3x меньше | **Нет** (merge) | 95-100% |
| **QLoRA** | 0.01-1% | ~6x меньше | **Нет** (merge) | 90-99% |
| **Adapter layers** | ~1-3% | Чуть меньше | 5-30% latency | 95-99% |
| **Prefix Tuning** | ~0.1% | Минимум | Minimal | 85-95% |
| **Prompt Tuning** | ~0.01% | Минимум | Minimal | 80-95% |
| **(IA)^3** | ~0.01% | Минимум | Minimal | 85-95% |

### LoRA — Low-Rank Adaptation

Де-факто стандарт PEFT. Замораживает все pre-trained веса, вводит обучаемые матрицы низкого ранга $B \cdot A$ параллельно attention матрицам:

$$h = W_0 x + BA \cdot x$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig1.png]]
*LoRA: замороженные веса $W$ + обучаемые $B$ и $A$. При inference сливаются — нулевой overhead (источник: Hu et al., 2021)*

- $r = 4-8$ — стандартный rank
- 10,000x меньше обучаемых параметров
- Checkpoint: 350 GB → 35 MB
- Подробнее: [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]]

### QLoRA — Quantized LoRA

Квантизация замороженных весов до **4 бит** (NF4) + LoRA в bf16. Позволяет fine-tune LLaMA 65B на одной 48 GB GPU.

- NF4: информационно-оптимальный тип данных для нормально распределённых весов
- Double Quantization: квантизация scaling constants
- Paged Optimizers: автоматическая выгрузка optimizer states в CPU
- Подробнее: [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA (раздел QLoRA)]]

### Adapter Layers

Bottleneck layers между Transformer блоками (Houlsby et al., 2019):

```
x → LayerNorm → Down-project (d → r) → GELU → Up-project (r → d) → + x
```

Обучаются только adapter параметры (~1-3% от модели). Минус: добавляют **sequential computation** → 5-30% latency overhead при inference (особенно заметно при batch\_size=1).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig5.png]]
*Adapter layers (AdapterL/AdapterH) добавляют 5-30% latency при batch\_size=1. LoRA = нулевой overhead (источник: Hu et al., 2021)*

### Prefix Tuning

Learnable prefix-векторы добавляются к Key и Value в каждом attention слое. LM полностью заморожен. Хорошо работает для generation задач, хуже для classification.

### Prompt Tuning (Lester et al., 2021)

Простейший PEFT: обучаемые soft-token embeddings добавляются перед входом. При достаточном масштабе модели (>10B) приближается к full fine-tuning.

## Требования к compute: практическая таблица

| Модель | Full FT | LoRA (r=8) | QLoRA (4-bit) |
|--------|---------|------------|---------------|
| **7B** (LLaMA) | 2x A100 80GB | 1x A100 40GB | 1x RTX 4090 24GB |
| **13B** | 4x A100 80GB | 1x A100 80GB | 1x A100 40GB |
| **33B** | 8x A100 80GB | 2x A100 80GB | 1x A100 80GB |
| **65-70B** | 16x A100 80GB | 4x A100 80GB | 1x A100 48GB |
| **175B** | Multi-node | 8x A100 80GB | 4x A100 80GB |

*Примерные значения, зависят от batch size, sequence length, optimizer.*

## Catastrophic Forgetting: когда fine-tuning вредит

Fine-tuning может «забыть» знания из pre-training — это **catastrophic forgetting**:

1. **Full FT на узкой задаче** — модель теряет general capabilities
2. **RLHF без pretraining mix** — PPO деградирует на SQuAD, HellaSwag (InstructGPT)
3. **Длительный fine-tuning** — overfitting к training data, потеря diversity

**Решения:**
- **PPO-ptx:** подмешивание pretraining loss (InstructGPT)
- **LoRA:** замороженные веса сохраняют pre-trained knowledge → меньше forgetting
- **Replay:** подмешивание данных из pre-training
- **EWC / L2 regularization:** штраф за отклонение от pre-trained весов

## Когда какой метод выбрать: decision tree

```
Есть ли аннотированные данные?
├── Нет → Prompting / ICL (zero/few-shot)
│         GPT-4, Claude → достаточно хороши без fine-tuning
│
├── Мало (<1K) → PEFT (LoRA, r=4)
│                Осторожно с overfitting
│                Рассмотреть few-shot fine-tuning
│
├── Средне (1K-100K) → LoRA (r=8-16) или QLoRA
│                      Стандартный рецепт для LLM
│
└── Много (>100K) → Full FT (если ресурсы есть)
                    или LoRA (если ресурсов нет)
                    Full FT лучше при domain shift

Нужен alignment (instruction following)?
└── SFT → DPO/RLHF
    Обычно через LoRA
```

| Сценарий | Рекомендация |
|----------|-------------|
| NLU classification (BERT-scale) | Full FT на encoder-only модели |
| Domain-specific LLM | LoRA/QLoRA на decoder-only модели |
| Instruction following | SFT + DPO через LoRA |
| Multi-task / multi-client | LoRA адаптеры (один base model, много адаптеров) |
| Production с жёстким latency | LoRA (merge) или Full FT |
| Consumer GPU (RTX 3090/4090) | QLoRA |

## Fine-tuning vs Prompting: когда что лучше

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]:

| Задача | Fine-tuned BERT/T5 | LLM (prompting) | Лучше |
|--------|-------------------|-----------------|-------|
| NER (CoNLL-03) | ~93% F1 | ~47% F1 | **Fine-tuned** (~2x) |
| Classification | SOTA | Competitive | **Fine-tuned** (при labels) |
| Summarization (ROUGE) | Выше | Ниже | Fine-tuned |
| Summarization (human pref) | — | — | **LLM** |
| Few/zero-shot | Невозможно | Работает | **LLM** |
| Open-ended generation | Ограничено | Сильно | **LLM** |

**Вывод:** при достаточном количестве labels fine-tuned специализированная модель побеждает. При few/zero-shot — prompting крупного LLM.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — первая фаза обучения
- [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] — стандартный PEFT метод
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — fine-tuning через RL
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — fine-tuning через preference optimization
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — SFT на инструкциях
- [[02 Areas/ML & DL/Concepts/Training/Few-shot Fine-tuning|Few-shot Fine-tuning]] — fine-tuning на малых данных
- [[02 Areas/ML & DL/Concepts/Training/PEFT|PEFT]] — обзор PEFT методов
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — пионер fine-tuning парадигмы

## Дополнительные ресурсы

- [Sebastian Raschka — Practical Tips for LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — лучшие практические советы по LoRA fine-tuning
- [Encora — Comparing Fine-Tuning Techniques](https://www.encora.com/interface/comparing-fine-tuning-optimization-techniques-lora-qlora-dora-and-qdora) — сравнение LoRA, QLoRA, DoRA
- [Modal — LoRA vs QLoRA](https://modal.com/blog/lora-qlora) — практическое сравнение с бенчмарками
