---
title: "SFT"
aliases: [SFT, Supervised Fine-Tuning, Supervised Fine Tuning]
type: concept
status: legacy
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/InstructGPT|InstructGPT]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA 2|LLaMA 2]]"
  - "[[02 Areas/ML & DL/Papers/ASTRA|ASTRA]]"
courses: []
sources:
  - "[Sebastian Raschka — Finetuning LLMs (2024)](https://magazine.sebastianraschka.com/p/finetuning-large-language-models)"
  - "[Hugging Face — SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)"
---

# SFT — Supervised Fine-Tuning

## Что это и зачем

Supervised Fine-Tuning — этап обучения, на котором [[02 Areas/ML & DL/Concepts/Training/Pre-training|предобученная]] языковая модель адаптируется к конкретному формату взаимодействия (инструкция → ответ) на размеченных парах данных. В отличие от pre-training, где модель учится предсказывать следующий токен на терабайтах неразмеченного текста, SFT использует относительно небольшой (тысячи — десятки тысяч примеров) датасет высокого качества.

SFT — первый этап post-training pipeline, принятого после InstructGPT (2022):

```
Pre-training → SFT → RLHF/DPO/GRPO
              ↑                ↑
         формат ответов    alignment
```

SFT учит модель **формату**: следовать инструкциям, отвечать в нужном стиле, использовать нужную структуру. [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]/[[02 Areas/ML & DL/Concepts/Training/DPO|DPO]]/[[02 Areas/ML & DL/Concepts/Training/GRPO|GRPO]] учат модель **предпочтениям**: какие ответы лучше, безопаснее, полезнее.

## Отличие от pre-training

| Свойство | Pre-training | SFT |
|----------|-------------|-----|
| Данные | Терабайты неразмеченного текста | Тысячи–десятки тысяч размеченных пар |
| Цель | Общее языковое моделирование | Следование инструкциям |
| Compute | Месяцы на кластере | Часы–дни на нескольких GPU |
| Learning rate | ~3e-4 | ~1e-5 – 2e-5 |
| Loss | CE на всех токенах | CE **только на completion** токенах |
| Результат | Base model (продолжает текст) | Instruct model (отвечает на вопросы) |

## Формат данных: chat templates

SFT-данные — это диалоги со специальными токенами, разделяющими роли. Два наиболее распространённых формата:

### ChatML (OpenAI, Qwen)

```
<|im_start|>system
Ты полезный ассистент.<|im_end|>
<|im_start|>user
Что такое GRPO?<|im_end|>
<|im_start|>assistant
GRPO — это алгоритм...<|im_end|>
```

Каждое сообщение обёрнуто в `<|im_start|>role` ... `<|im_end|>`. Все эти токены — **специальные**: они добавляются в словарь модели при SFT и получают свои embedding'и.

### Llama 3 format

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Ты полезный ассистент.<|eot_id|><|start_header_id|>user<|end_header_id|>

Что такое GRPO?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

GRPO — это алгоритм...<|eot_id|>
```

Llama 3 использует отдельные токены для начала/конца заголовков (`start_header_id`, `end_header_id`) и конца реплики (`eot_id`). Между заголовком и текстом — два newline.

**Критически важно:** несоответствие chat template при inference тому, на чём модель обучена, приводит к деградации качества. Tokenizer хранит шаблон в `tokenizer_config.json` → `chat_template` (Jinja2).

## Loss masking: учим только на ответах

Ключевая деталь SFT — **loss считается только по токенам ответа ассистента**, а не по промпту.

Для каждого токена $t_i$ в последовательности:

$$\mathcal{L} = -\sum_{i \in \text{completion}} \log p_\theta(t_i | t_{<i})$$

На практике это реализуется через `labels`:

```python
# Промпт: label = -100 (игнорируется CrossEntropyLoss)
# Completion: label = token_id
labels = [-100, -100, ..., -100, tok_1, tok_2, ..., tok_n, eos]
#         ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#              prompt tokens              completion tokens
```

PyTorch `CrossEntropyLoss(ignore_index=-100)` автоматически исключает эти позиции из градиента. Без loss masking модель тратит capacity на запоминание промптов — это расточительно и может вредить обобщению.

## Проблема эффективности: пacking

Наивный подход к SFT батчированию — каждый пример padding'ится до `max_seq_len`. Если средняя длина примера 300 токенов, а `max_seq_len=2048`, то ~85% compute тратится на padding-токены.

**Packing** решает эту проблему: несколько примеров конкатенируются в одну последовательность:

```
[example_1][EOS][example_2][EOS][example_3][EOS][PAD][PAD]
|<----------- max_seq_len -------------------------------->|
```

При packing важно использовать **block-diagonal attention mask**, чтобы примеры не «видели» друг друга — иначе модель может использовать информацию из чужого примера для предсказания токенов текущего.

В TRL `SFTTrainer` packing включается через `packing=True`. Flash Attention 2 поддерживает variable-length sequences через `cu_seqlens`, что делает packing эффективным без явного attention mask.

## LoRA и QLoRA для SFT

Full fine-tuning 7B модели требует ~56 GB в FP16 (параметры + градиенты + optimizer states). [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] (Low-Rank Adaptation) замораживает основные веса и обучает только low-rank матрицы:

$$W' = W + \Delta W = W + BA$$

где $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$ (обычно $r = 8...64$).

**QLoRA** идёт дальше: базовая модель квантизуется в 4-bit (NF4), а LoRA-адаптеры обучаются в BF16. Это позволяет дообучать 70B модель на одной GPU с 80 GB.

Типичная конфигурация LoRA для SFT:

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,            # scaling = alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

На практике LoRA SFT даёт 95-99% качества full fine-tuning при 10x меньшем потреблении памяти.

## Качество данных важнее количества

Ключевой инсайт из research: **для SFT качество данных критически важнее количества.**

- **LIMA** (2023): 1,000 тщательно отобранных примеров дали модель, сопоставимую с GPT-4 по human preference на некоторых бенчмарках
- **Alpaca** (2023): 52K примеров, сгенерированных GPT-3.5, дали заметно худшую модель
- **LLaMA 2**: Meta использовала 27,540 SFT примеров — отобранных вручную аннотаторами

Основные принципы курирования SFT-данных:

1. **Diversity** — покрывать максимум задач и форматов
2. **Quality** — каждый пример должен быть эталоном ответа
3. **Consistency** — единый стиль и формат ответов
4. **Decontamination** — убрать overlap с eval бенчмарками

Распространённая ошибка: генерировать огромные SFT-датасеты через более сильную LLM. Модель быстро переобучается на стилистические паттерны и теряет разнообразие. Лучше 5K отличных примеров, чем 100K посредственных.

## Когда SFT достаточно, а когда нужен RL

### SFT достаточно, когда:

- Задача детерминирована — есть один правильный ответ (классификация, NER, extraction)
- Формат ответа важнее «мягких» качеств (JSON output, структурированные ответы)
- Данные высокого качества и покрывают целевое распределение
- Бюджет ограничен — SFT в 10-100x дешевле RL pipeline

### RL нужен, когда:

- Задача требует exploration — модель должна «изобретать» стратегии (reasoning, tool-use)
- Нет чётко правильного ответа — нужно учить preferences (safety, tone, helpfulness)
- SFT вызывает memorization — модель копирует примеры вместо обобщения (см. [[02 Areas/ML & DL/Papers/ToolRL|ToolRL]]: GRPO cold-start лучше SFT→GRPO на single-turn tool-use)
- Нужна самокоррекция — [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] через GRPO стимулирует emergence self-reflection

### Типичные pipeline

| Сценарий | Pipeline |
|----------|----------|
| Chatbot общего назначения | SFT → RLHF/DPO |
| Reasoning модель | SFT → GRPO с verifiable rewards |
| Доменный ассистент | SFT (достаточно) |
| Tool-use агент (multi-turn) | SFT → GRPO ([[02 Areas/ML & DL/Papers/ASTRA|ASTRA]]) |
| Tool-use агент (single-turn) | GRPO cold-start ([[02 Areas/ML & DL/Papers/ToolRL|ToolRL]]) |

## SFT как knowledge injection vs format learning

Важно различать два режима SFT:

### Format learning (основной)

Модель **уже знает** ответы — pre-training дал ей знания. SFT учит отвечать в нужном формате. Именно поэтому хватает тысяч примеров: нужно показать формат, а не передать знания.

Пример: base модель знает, что Париж — столица Франции. SFT учит её отвечать «Столица Франции — Париж» на вопрос «Какая столица Франции?» вместо продолжения «Какая столица Франции? А вот ещё вопрос...»

### Knowledge injection

SFT на доменных данных (медицина, юриспруденция, внутренние документы компании) пытается **добавить** знания. Это работает хуже: модель может «забыть» pre-training знания (catastrophic forgetting), а новые знания плохо обобщаются. Для injection лучше подходят:
- **RAG** — retrieval-augmented generation, знания в контексте
- **Continued pre-training** — дообучение в формате CLM на доменном корпусе, затем SFT
- **LoRA merge** — отдельные адаптеры для домена и формата

## Распространённые ошибки при SFT

1. **Слишком много эпох**: 3+ эпохи на маленьком датасете → модель заучивает ответы наизусть, теряет способность генерализовать
2. **Нет loss masking**: обучение на промптах тратит compute и может вредить — модель «запоминает» промпты вместо обучения генерации
3. **Смешение форматов**: разные chat templates в одном датасете → модель путается с special tokens
4. **Переоценка бенчмарков**: SFT легко переобучается на формат конкретного бенчмарка (MMLU, GSM8K), давая inflate scores
5. **Игнорирование system prompt**: если модель будет использоваться с system prompt, он должен быть в SFT-данных

## Практические советы

- **Learning rate**: 1e-5 — 2e-5 для full fine-tuning, 1e-4 — 3e-4 для LoRA
- **Epochs**: 1-3 эпохи. Больше — переобучение. На маленьких датасетах (< 1K) — 1 эпоха
- **Batch size**: gradient accumulation до effective batch size 32-128
- **Warmup**: 3-10% от total steps
- **Max seq length**: подбирать под распределение длин в данных, обрезая top-1% outliers
- **Eval**: проверять на held-out примерах + qualitative evaluation (читать ответы)
- **Reproducibility**: фиксировать seed, логировать гиперпараметры, сохранять чекпоинты каждую эпоху

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — общий термин, SFT — специфичный для instruction-following
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — близкий термин, иногда используется как синоним SFT
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — предыдущий этап pipeline
- [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] — parameter-efficient метод для SFT
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — следующий этап pipeline (alignment через human feedback)
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — альтернатива RLHF для alignment
- [[02 Areas/ML & DL/Concepts/Training/RLVR|RLVR]] — RL с verifiable rewards, альтернатива/дополнение к SFT
