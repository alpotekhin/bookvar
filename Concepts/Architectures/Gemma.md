---
title: "Gemma"
aliases: [Gemma, Gemma 2B, Gemma 7B]
type: concept
status: legacy
category: Architectures
papers: ["[[02 Areas/ML & DL/Papers/Gemma|Gemma]]"]
courses: []
sources:
  - "[Gemma Team — Gemma: Open Models Based on Gemini Research and Technology (2024)](https://arxiv.org/abs/2403.08295)"
  - "[Google DeepMind — Gemma Release](https://blog.google/technology/developers/gemma-open-models/)"
  - "[HuggingFace Gemma](https://huggingface.co/google/gemma-7b)"
---

# Gemma

## Зачем эта модель появилась

К началу 2024 года Google оказалась в парадоксальной ситуации: компания, создавшая Transformer, BERT, и T5, **не имела конкурентоспособной open-weight модели**. LLaMA (Meta) и Mistral доминировали в open-source, а Gemini (Google) был закрыт. 

Gemma (февраль 2024) — **первая открытая модель от Google**, построенная на технологиях Gemini. Два размера — 2B (on-device/CPU) и 7B (GPU) — покрывающие основные deployment сценарии. Gemma 7B превосходит LLaMA 2 7B и Mistral 7B на **11 из 18 бенчмарков**.

## Архитектура: Gemini DNA в компактной форме

Gemma — **decoder-only Transformer** с архитектурными решениями, унаследованными от Gemini:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma/fig1.png]]
*Сравнение Gemma 7B с LLaMA 2 7B, LLaMA 2 13B и Mistral 7B по четырём категориям (источник: Gemma paper)*

### Параметры моделей

| Параметр | Gemma 2B | Gemma 7B |
|----------|----------|----------|
| d_model | 2048 | 3072 |
| Layers | 18 | 28 |
| FFN hidden dims | 32768 | 49152 |
| Num heads | 8 | 16 |
| **Num KV heads** | **1 (MQA)** | **16 (MHA)** |
| Head size | 256 | 256 |
| Context length | 8192 | 8192 |
| Vocabulary | 256,128 | 256,128 |
| Embedding params | 525M | 787M |
| Non-embedding params | 1.98B | 7.75B |

### Ключевые архитектурные решения

**Multi-Query Attention (MQA) для 2B:** все query-головы делят **одну** KV-голову ($\text{num\_kv\_heads} = 1$). Агрессивная оптимизация для on-device inference — минимальный KV-cache. Для 7B используется стандартный MHA ($\text{num\_kv\_heads} = 16$), потому что ablation показал: MQA работает хорошо **на малом масштабе**, но теряет качество на большом.

**RoPE (Rotary Positional Embeddings):** те же RoPE, что в LLaMA, — вращение в комплексном пространстве для кодирования позиций. Обеспечивает extrapolation на длинные последовательности.

**GeGLU Activations:** замена стандартного ReLU на approximate GeGLU:

$$\text{GeGLU}(x, W, V) = \text{GELU}(xW) \odot (xV)$$

GeGLU даёт лучшее quality/compute trade-off, чем ReLU или даже SwiGLU. Выбор activation function — один из тех «мелких» решений, которые в сумме дают значительный прирост.

**RMSNorm:** нормализация входа каждого sublayer (attention + FFN) через RMSNorm вместо LayerNorm. Быстрее (нет mean computation) и стабильнее при обучении.

**Shared Input/Output Embeddings:** одна матрица используется и для input embedding, и для output projection. Это **экономит параметры** — для vocabulary 256K одна embedding матрица занимает ~525M-787M параметров. Без sharing модель 2B была бы ~2.5B.

**Большой Vocabulary (256K):** унаследован от Gemini, оптимизирован для многоязычности. Для сравнения: LLaMA = 32K, Mistral = 32K. Больший vocabulary = лучшая compression ratio = меньше токенов для того же текста → **больше информации** в контекстном окне.

## Training Data

| Модель | Tokens | Данные |
|--------|--------|--------|
| Gemma 2B | **3T** | Web, math, code (primarily English) |
| Gemma 7B | **6T** | Web, math, code (primarily English) |

Для сравнения: LLaMA 2 7B обучена на 2T токенов, Mistral 7B — предположительно на ~2T. Gemma обучена на **значительно большем количестве данных** — в духе Chinchilla scaling (больше данных, не больше параметров).

Gemma НЕ является multimodal (в отличие от Gemini) и НЕ оптимизирована для state-of-the-art multilingual (хотя vocabulary это поддерживает).

## Data Filtering: safety-first подход

Google применила многоуровневую фильтрацию pre-training данных:

1. **Heuristic filters**: длина, повторения, разметка, качество текста
2. **Model-based classifiers**: фильтрация harmful и low-quality контента через отдельные модели
3. **Google Cloud Sensitive Data Protection**: удаление personal information (телефоны, email, SSN)
4. **Evaluation set removal**: все evaluation sets **явно удалены** из pre-training data для предотвращения contamination
5. **Memorization mitigation**: минимизация proliferation sensitive outputs

Результат: exact memorization rates **сопоставимы с PaLM**, нет случаев memorization sensitive data.

## Instruction Tuning: двухэтапный процесс

**Stage 1: SFT** на синтетических и human-generated prompt-response парах:
- Данные отбираются через **LM-based side-by-side evaluations** (модель сравнивает пары ответов)
- Синтетические данные генерируются большими Gemini моделями

**Stage 2: RLHF** с reward model:
- Bradley-Terry reward model обучена на human preferences
- Обучение через **novel RL algorithm** (детали не раскрыты)

## Ключевые результаты

### Gemma 7B vs конкуренты

| Benchmark | LLaMA 2 7B | LLaMA 2 13B | Mistral 7B | **Gemma 7B** |
|-----------|------------|-------------|------------|-------------|
| MMLU (5-shot) | 45.3 | 54.8 | 62.5 | **64.3** |
| HellaSwag (0-shot) | 77.2 | 80.7 | 81.0 | **81.2** |
| GSM8K (5-shot) | 14.6 | 28.7 | 35.4 | **46.4** |
| HumanEval (pass@1) | 12.8 | 18.3 | 26.2 | **32.3** |
| PIQA (0-shot) | 78.8 | 80.5 | 82.1 | **81.2** |
| Average | — | 52.4 | — | **56.9** |

Gemma 7B превосходит **LLaMA 2 13B** (почти вдвое большую модель) по average score (56.9 vs 52.4).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gemma/fig2.png]]
*Подробные результаты Gemma по категориям бенчмарков (источник: Gemma paper)*

### Gemma 2B: on-device champion

Gemma 2B предназначена для **CPU и мобильных устройств**. При 1.98B non-embedding параметрах она:
- Конкурентна с моделями 3-4B параметров
- Работает на CPU с приемлемой скоростью
- Подходит для edge deployment (смартфоны, IoT)

## Почему Gemma важна

### 1. Первый open-weight шаг Google

Google десятилетиями делала open-source для ML (TensorFlow, BERT, T5), но в era LLM замкнулась. Gemma — сигнал возврата к открытости и начало серии Gemma 2, Gemma 3.

### 2. Демонстрация Gemini технологий

Gemma показала, что **технологии Gemini** (vocabulary, training recipes, safety practices) работают на маленьком масштабе. Это информирует community о direction исследований Google.

### 3. Safety-first open release

Уровень проработки safety (фильтрация, memorization analysis, responsible deployment guide) стал **ориентиром** для других open-weight releases.

### 4. On-device AI

Gemma 2B — одна из первых серьёзных моделей, **целенаправленно оптимизированных для CPU/mobile**. Это расширило применимость open LLM за пределы GPU кластеров.

## Gemma 2 и далее

В июне 2024 Google выпустила **Gemma 2** (9B, 27B) с значительными улучшениями:
- Knowledge distillation от больших моделей
- Grouped-Query Attention для обоих размеров
- Sliding Window Attention (как Mistral)
- Значительно лучшие результаты при том же размере

Gemma стала **семейством**, а не одноразовым release.

## Key papers

- [[02 Areas/ML & DL/Papers/Gemma|Gemma]] — семейство 2B/7B open-weight моделей

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектурная основа
- [[02 Areas/ML & DL/Concepts/Architectures/Mistral 7B|Mistral 7B]] — конкурент в 7B классе
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA 2|LLaMA 2]] — конкурент, которого Gemma превосходит
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — alignment метод
- [[02 Areas/ML & DL/Concepts/Training/Instruction Tuning|Instruction Tuning]] — SFT stage
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — RoPE
- [[02 Areas/ML & DL/Concepts/Architectures/Qwen2|Qwen2]] — конкурент из Alibaba

## Дополнительные ресурсы

- [Gemma paper (arXiv)](https://arxiv.org/abs/2403.08295) — полный технический отчёт
- [Google DeepMind Blog — Gemma](https://blog.google/technology/developers/gemma-open-models/) — анонс
- [HuggingFace Gemma-7B](https://huggingface.co/google/gemma-7b) — pretrained чекпоинт
- [Kaggle Gemma](https://www.kaggle.com/models/google/gemma) — чекпоинты и tutorials
