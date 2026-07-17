---
title: "LLaMA 3"
aliases: [LLaMA 3, Llama 3, LLaMA-3, Llama 3.1, Llama 3.1 405B, Meta Llama 3]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/LLaMA 3 Herd of Models|LLaMA 3]]"
courses: []
sources:
  - "[Meta AI Blog — Introducing Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/)"
  - "[The Llama 3 Herd of Models Technical Report (2024)](https://arxiv.org/abs/2407.21783)"
  - "[HuggingFace — meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)"
  - "[DataCamp — What Is Llama 3.1 405B?](https://www.datacamp.com/blog/llama-3-1-405b-meta-ai)"
  - "[Meta AI Blog — Llama 3.2 Multimodal](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)"
---

# LLaMA 3

## Зачем эта модель появилась

К 2024 году open-source LLM значительно отставали от закрытых моделей (GPT-4, Claude 3): разрыв был особенно заметен на complex reasoning, long context и multimodal задачах. LLaMA 2 70B проигрывала GPT-4 на большинстве бенчмарков с большим отрывом.

**LLaMA 3** (Meta, апрель 2024) и **LLaMA 3.1** (июль 2024) — попытка Meta **закрыть разрыв** между open и closed моделями. Flagship: **405B параметров**, обученная на **15T+ токенов** с контекстом **128K**. Впервые open-weight модель **сопоставима с GPT-4** на основных бенчмарках.

## Философия: scaling > architecture

Принципиальное решение Meta:

> **Стандартная архитектура + максимальный масштаб данных и compute.**

В отличие от DeepSeek (MoE, MLA) или Qwen (MoE, thinking modes), Meta намеренно выбрала **dense Transformer** без экзотических компонентов. Аргумент: архитектурные инновации могут быть нестабильны на большом масштабе, а dense model проще масштабировать и обучать.

## Архитектура

### Dense Decoder-only Transformer

| Параметр | LLaMA 3 8B | LLaMA 3 70B | LLaMA 3.1 405B |
|----------|------------|-------------|-----------------|
| Params | 8B | 70B | 405B |
| Layers | 32 | 80 | 126 |
| Hidden Size | 4,096 | 8,192 | 16,384 |
| Attention | GQA (8 KV-heads) | GQA (8 KV-heads) | GQA (8 KV-heads) |
| Context | 8K → 128K | 8K → 128K | 128K |
| Vocab Size | 128,256 | 128,256 | 128,256 |
| Training Tokens | 15T+ | 15T+ | 15T+ |

### Ключевые архитектурные решения

**Grouped Query Attention (GQA)** с 8 KV-головами — все размеры моделей, включая 8B. В LLaMA 2 GQA использовалось только в 70B. Это обеспечивает эффективный inference без потери качества.

**Расширенный словарь (128K токенов):** в 4 раза больше, чем у LLaMA 2 (32K). Новый tokenizer (tiktoken-based) лучше покрывает non-English языки и код. Это уменьшает число токенов на текст → меньше compute при inference.

**RoPE с увеличенной частотой:** Rotary Position Embeddings с $\theta = 500{,}000$ (вместо 10,000 в LLaMA 2), что позволяет расширить контекст до 128K через длинный fine-tuning.

**SwiGLU активация:** $\text{SwiGLU}(x) = x \cdot \sigma(\beta x) \cdot Wx$ — более выразительная, чем ReLU.

## Данные: 15T+ токенов

### Масштаб

| Модель | Training Tokens | Соотношение |
|--------|-----------------|-------------|
| LLaMA 1 | 1.4T | 1x |
| LLaMA 2 | 2T | 1.4x |
| **LLaMA 3** | **15T+** | **10.7x** |

LLaMA 3 обучена на **в 7.5 раз больше данных**, чем LLaMA 2. Это одно из самых масштабных pre-training на тот момент.

### Data pipeline (детальный разбор)

1. **Web crawl** — основной источник (>80%), многоэтапная фильтрация и деduplication
2. **Heuristic quality filters** — URL quality, text quality, n-gram analysis
3. **Model-based quality filters** — classifier предсказывает, был бы текст ответом LLM на вопрос (similar to Phi approach)
4. **Code data** — отдельный pipeline для GitHub repos, Stack Overflow, documentation
5. **Math data** — scientific papers, textbooks, проблемные сборники
6. **Multilingual data** — 8 основных языков + дополнительные (balanced by quality, not volume)

### Data Mix Table

| Домен | Приблизительная доля | Фильтрация |
|-------|---------------------|------------|
| Web (English) | ~50% | URL + heuristic + model-based |
| Web (Multilingual) | ~10% | Language ID + quality filter |
| Code | ~15% | Syntax validation + dedup |
| Math/Science | ~10% | Domain classifiers |
| Books | ~5% | Dedup, quality |
| Other (Wikipedia, forums) | ~10% | Standard pipeline |

**Data annealing:** в финальной фазе обучения доля high-quality данных (curated, math, code) **увеличивается**, а доля web crawl уменьшается. Это улучшает quality без увеличения total compute.

### Scaling Law для Data Mix

Meta провела масштабное исследование **data mix optimization**: обучение тысяч маленьких моделей с разными пропорциями данных для нахождения оптимального mix для 405B модели. Результат: оптимальный mix для маленькой модели **не совпадает** с оптимальным для большой — Meta разработала методологию extrapolation через scaling laws.

## Пост-обучение: iterative alignment

### Шесть раундов

Meta использует **итеративный** подход к alignment:

1. **SFT** на human-written данных
2. **Rejection Sampling** — генерация N ответов, выбор лучшего по reward model
3. **DPO (Direct Preference Optimization)** — обучение на парах preferred/rejected ответов
4. Повторение шагов 1-3 с синтетическими данными из предыдущего раунда

Каждый раунд улучшает и модель, и данные (модель генерирует лучшие synthetic данные → лучшее обучение → ещё лучшие данные).

### Safety training

**Многоуровневая система безопасности:**

1. **Safety SFT:** обучение на adversarial prompts с безопасными ответами
2. **Red teaming:** внутренняя и внешняя команды пытаются вызвать нежелательное поведение
3. **Context distillation:** модель учится генерировать безопасные ответы через prepending safety system prompt при SFT, но без system prompt при inference
4. **Safety RLHF:** отдельный reward model для safety, оптимизирующий одновременно helpfulness и safety
5. **Llama Guard:** отдельная модель-классификатор для фильтрации unsafe inputs/outputs при deployment

**Trade-off helpfulness vs safety:** Meta использует **Pareto-optimal** подход — находится точка, где увеличение safety минимально снижает helpfulness. Избыточная safety (over-refusals) считается не менее проблемной, чем under-safety.

## Обучение: инженерный подвиг

### Масштаб инфраструктуры

- **16,384 GPU H100** (два кластера по 24K, использовались 16K)
- **~30.84M GPU-часов** обучения 405B модели
- **Более 54 дней** непрерывного обучения
- **Uptime >95%** — Meta разработала специальные failure recovery механизмы

### Reliability Engineering

При 16K GPU отказ оборудования — **не если, а когда**. Meta зафиксировала **466 job interruptions** за время обучения 405B, из которых:
- 47 — запланированные
- 419 — неожиданные (GPU failures, network issues, software bugs)

Автоматический checkpoint recovery позволял возобновлять обучение за минуты, теряя менее 3% compute.

## Мультимодальные расширения: Llama 3.2

В сентябре 2024 Meta выпустила **Llama 3.2** — первые мультимодальные модели в серии:

### Vision модели

| Модель | Params | Возможности |
|--------|--------|-------------|
| Llama 3.2 11B Vision | 11B | Image understanding, OCR, VQA |
| Llama 3.2 90B Vision | 90B | Advanced image reasoning, charts, diagrams |

**Архитектура vision:**
- **Vision encoder:** отдельно обученный ViT (Vision Transformer)
- **Cross-attention adapter:** интеграция visual tokens с LLM через cross-attention (не simple concatenation)
- **Frozen LLM weights:** text capabilities не деградируют при добавлении vision

**Обучение vision:**
1. **Image-text pretraining:** десятки миллионов image-text pairs (captioning, VQA, OCR, spatial reasoning)
2. **Synthetic data augmentation:** Llama 3.1 генерирует вопросы/ответы по изображениям, reward model отбирает лучшие
3. **Multi-round alignment:** SFT + rejection sampling + DPO, аналогично text-only pipeline

### Lightweight модели

| Модель | Params | Назначение |
|--------|--------|-------------|
| Llama 3.2 1B | 1B | Mobile/edge deployment |
| Llama 3.2 3B | 3B | On-device AI |

Обучены через **pruning + distillation** из больших моделей. Поддерживают 128K контекст.

### Llama Guard 3 Vision

Отдельная safety модель для мультимодального контента — **Llama-Guard-3-11B-Vision**. Классифицирует и текстовые, и визуальные inputs/outputs на безопасность.

## Результаты

| Бенчмарк | LLaMA 3.1 405B | GPT-4 (0125) | Claude 3.5 Sonnet | DeepSeek-V3 |
|-----------|----------------|--------------|-------------------|-------------|
| MMLU | 88.6 | 86.5 | 88.3 | 88.5 |
| HumanEval | 89.0 | 86.6 | 92.0 | 82.6 |
| GSM8K | 96.8 | 95.3 | 96.4 | — |
| MATH | 73.8 | 64.5 | 71.1 | 90.2 |
| GPQA | 50.7 | 41.4 | 59.4 | 59.1 |

LLaMA 3.1 405B — первая open-weight модель, **сравнимая с GPT-4** по breadth of capabilities. Но уступает DeepSeek-V3 на math при значительно большем compute cost.

## Влияние на экосистему

1. **Open-weight frontier:** доказательство, что open models могут конкурировать с closed
2. **Стандарт для fine-tuning:** LLaMA 3 стала базой для тысяч specialized моделей
3. **Distillation source:** DeepSeek-R1 и другие использовали LLaMA 3 как target для дистилляции
4. **Infrastructure playbook:** детальный отчёт о training reliability стал reference для индустрии
5. **Multimodal open standard:** Llama 3.2 Vision — первые competitive open multimodal models

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA 2|LLaMA 2]] — предыдущее поколение
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — оригинальная модель
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — используется в alignment
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — мотивация scaling-подхода
- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-V3|DeepSeek-V3]] — альтернативный подход (MoE vs dense)
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — GQA оптимизация

## Дополнительные ресурсы

- [Llama 3 Herd of Models Technical Report](https://arxiv.org/abs/2407.21783) — 92-страничный отчёт
- [Meta AI Blog — Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/) — официальный анонс
- [Meta AI Blog — Llama 3.2](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) — мультимодальные модели
