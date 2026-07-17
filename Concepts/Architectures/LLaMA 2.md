---
title: "LLaMA 2"
aliases: [Llama 2, Llama2, LLaMA-2, Llama 2-Chat]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/LLaMA 2]]"
  - "[[02 Areas/ML & DL/Papers/LLaMA]]"
courses: []
sources:
  - "[Touvron et al. — Llama 2: Open Foundation and Fine-Tuned Chat Models (2023)](https://arxiv.org/abs/2307.09288)"
  - "[Meta AI — Llama 2 Release Blog](https://ai.meta.com/llama/)"
---

# LLaMA 2

## Зачем эта модель появилась

До LLaMA 2 (июль 2023) в мире LLM существовал чёткий разрыв: **закрытые «продуктовые» модели** (ChatGPT, Claude, Bard) были значительно лучше **открытых** (LLaMA 1, BLOOM, Falcon) для реальных задач. Причина — закрытые модели проходили многоитерационный RLHF alignment, детали которого никто не публиковал.

LLaMA 2 закрыла этот разрыв. Meta выпустила модели от 7B до 70B параметров **с коммерческой лицензией**, а главное — опубликовала **беспрецедентно подробное описание RLHF pipeline**. Впервые open-source модель стала реальной альтернативой ChatGPT для продуктовых задач.

## Архитектура: эволюция LLaMA 1

Архитектурная основа та же — decoder-only Transformer с **pre-normalization (RMSNorm)**, **SwiGLU activation**, и **Rotary Positional Embeddings (RoPE)**. Ключевые изменения:

### 1. Увеличенный контекст: 2048 → 4096

Удвоение context length позволило модели работать с более длинными документами и многоходовыми диалогами. RoPE позволяет это сделать относительно безболезненно, потому что relative positional encoding лучше обобщается на unseen длины.

### 2. Grouped-Query Attention (GQA) для 34B и 70B

Стандартный Multi-Head Attention (MHA): каждая голова имеет свои Q, K, V проекции.
Multi-Query Attention (MQA): все головы делят один K, V → быстрее, но может терять качество.
**GQA — промежуточное решение**: головы объединены в **группы**, и каждая группа делит одну KV проекцию.

$$\text{GQA:} \quad \text{num\_kv\_heads} < \text{num\_query\_heads}$$

**Почему это критично:** при авторегрессивном инференсе нужно хранить KV-cache для всех предыдущих токенов. Для 70B модели с 64 головами и длинным контекстом — это гигабайты GPU памяти. GQA сокращает KV-cache в несколько раз при минимальной потере качества.

### 3. Больше данных: 1T → 2T токенов

| | LLaMA 1 | LLaMA 2 |
|--|---------|---------|
| Context | 2048 | **4096** |
| Training tokens | 1.0-1.4T | **2.0T** |
| GQA (34B/70B) | Нет | **Да** |
| Data mix | CommonCrawl + Wiki + Books | **Новый mix** (без Meta данных) |
| Personal data filtering | Нет | **Да** |

Важно: training loss на 2T токенов **не показал сатурации** (Figure 5 в paper). Модели продолжали учиться — Meta просто остановила обучение из compute-budget ограничений.

### Семейство моделей

| Модель | Params | Context | GQA | LR |
|--------|--------|---------|-----|-----|
| Llama 2 7B | 7B | 4096 | Нет | 3.0e-4 |
| Llama 2 13B | 13B | 4096 | Нет | 3.0e-4 |
| Llama 2 34B | 34B | 4096 | Да | 1.5e-4 |
| Llama 2 70B | 70B | 4096 | Да | 1.5e-4 |

## Fine-tuning pipeline: от Base к Chat

Pipeline обучения Llama 2-Chat — самая детально описанная RLHF процедура в открытой литературе.

### Stage 1: Supervised Fine-Tuning (SFT)

Ключевой инсайт: **Quality Is All You Need** (название подсекции в paper).

- Начали с публичных инструкционных данных (Flan, etc.)
- Но быстро обнаружили: **тысячи высококачественных примеров > миллионы низкокачественных**
- Итоговый датасет: всего **27,540 аннотаций** (удивительно мало!)
- Third-party SFT данные отвергнуты из-за низкого качества
- Качество проверялось ручным сравнением: model outputs часто были **не хуже human-written** SFT данных

### Stage 2: Итеративный RLHF

Самое детальное описание RLHF в открытой литературе:

**a) Reward Modeling:** две отдельные reward models — **Helpfulness RM** и **Safety RM**.

Почему две: helpfulness и safety **конфликтуют** — самый helpful ответ на «как сделать бомбу» — подробная инструкция, но это unsafe. Одна модель не может хорошо оптимизировать оба критерия.

Loss с margin:

$$\mathcal{L} = -\log(\sigma(r_\theta(x, y_c) - r_\theta(x, y_r) - m(r)))$$

где $m(r)$ — margin, зависящий от степени предпочтения (significantly better → большой margin, slightly better → маленький). Это учит модель давать **более разные скоры** для более разных ответов.

**b) Rejection Sampling** (ранние итерации): генерируем K ответов на каждый промпт, выбираем лучший по reward model, используем как новый training example.

**c) PPO** (поздние итерации): стандартный Proximal Policy Optimization поверх rejection sampling чекпоинта.

**Итеративность**: RLHF проводился в 5 итераций (V1-V5). После каждой итерации собирались новые preference данные на **улучшенной модели**, чтобы reward model оставалась on-distribution. Всего собрано **1.4M binary comparisons** (vs ~170K у InstructGPT).

### Stage 3: Ghost Attention (GAtt) — multi-turn consistency

**Проблема:** после нескольких ходов диалога модель «забывает» системный prompt. Например, инструкция «говори как Оскар Уайльд» работает 2-3 хода, потом модель переключается на обычный стиль.

**Ghost Attention — элегантный хак:** при создании SFT данных для multi-turn dialogues системный промпт **синтетически вставляется в начало каждого хода**, а потом на inference маскируется (loss считается только для ответов). Это учит attention механизм **постоянно обращать внимание на системный промпт**, даже если физически он присутствует только в начале.

Результаты (Table 30 из paper):

| Ход диалога | Без GAtt | С GAtt |
|-------------|----------|--------|
| 2 | 100% | 100% |
| 4 | 10% | **100%** |
| 6 | 0% | **100%** |
| 20 | 0% | **100%** |

GAtt обобщается на ранее невиданные инструкции (zero-shot): «отвечай хайку», «всегда используй один абзац» — модель следует им стабильно до конца контекстного окна.

## Safety: многоуровневая защита

Meta подошла к safety с беспрецедентной для open-source серьёзностью:

1. **Safety-specific SFT**: аннотаторы писали adversarial промпты и safe ответы на них
2. **Отдельная Safety RM**: reward model, оптимизированная на безопасность
3. **Red teaming**: несколько раундов с внутренними и внешними командами (~350 экспертов)
4. **Context distillation**: техника переноса знаний из safety-prompted модели в стандартную

## Ключевые результаты

### Pretrained модель vs конкуренты

| Модель | MMLU | GSM8K | HumanEval | BBH |
|--------|------|-------|-----------|-----|
| Llama 2 70B | 68.9 | 56.8 | 29.9 | 51.2 |
| GPT-3.5 | 70.0 | 57.1 | 48.1 | — |
| PaLM 540B | 69.3 | 56.5 | 26.2 | 52.3 |
| GPT-4 | **86.4** | **92.0** | **67.0** | — |

### Llama 2-Chat (human evaluation, ~4K промптов)

- **Превосходит все открытые модели** на helpfulness
- **Сопоставима с ChatGPT** на helpfulness (human eval)
- **Значительно безопаснее** LLaMA 1, Vicuna, WizardLM

## Почему LLaMA 2 изменила экосистему

| Аспект | Влияние |
|--------|---------|
| **Коммерческая лицензия** | Первая открытая модель такого уровня с коммерческим использованием (LLaMA 1 — research only) |
| **Детальный RLHF** | Де-факто учебник по RLHF для индустрии |
| **GQA стандарт** | GQA стал нормой для всех последующих моделей (Mistral, Llama 3, Qwen2) |
| **Ghost Attention** | Новая техника для multi-turn consistency |
| **Open chat model** | Первая реальная альтернатива ChatGPT в open-source |

## Carbon footprint

Общий compute: **3.3M GPU hours** на NVIDIA A100-80GB. Carbon: 539 tCO2eq (100% offset через Meta sustainability program). Использовались два кластера — один с InfiniBand, другой с commodity RoCE ethernet — демонстрируя что дешёвый interconnect работает для обучения до 2000 GPU.

## Key papers

- [[02 Areas/ML & DL/Papers/LLaMA 2]] — оригинал (Touvron et al., Jul 2023)
- [[02 Areas/ML & DL/Papers/LLaMA]] — predecessor

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — архитектурная основа
- [[02 Areas/ML & DL/Concepts/Architectures/Decoder-only|Decoder-only]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — alignment метод
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — SFT stage
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — GQA как оптимизация
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — GQA экономит KV-cache
- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — наследник, также использует GQA + SWA
