---
title: "Qwen2"
aliases: [Qwen2, Qwen2-72B, Qwen2-7B, Qwen2-MoE]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Qwen2|Qwen2]]"
  - "[[02 Areas/ML & DL/Papers/Qwen 2.5 Technical Report|Qwen 2.5]]"
courses: []
sources:
  - "[Yang et al. — Qwen2 Technical Report (2024)](https://arxiv.org/abs/2407.10671)"
  - "[Qwen GitHub](https://github.com/QwenLM/Qwen2)"
  - "[HuggingFace Qwen2](https://huggingface.co/Qwen)"
---

# Qwen2

## Зачем эта модель появилась

К середине 2024 года гонка open-weight LLM вышла на новый уровень. LLaMA 3 от Meta и GPT-4o от OpenAI установили новые стандарты, но обе были либо закрыты, либо ограничены в размерах. **Qwen2** (Alibaba, июль 2024) — серия LLM, которая **бросила вызов всем**: 5 dense моделей от 0.5B до 72B + MoE модель 57B-A14B, обученные на 7T+ токенов с поддержкой ~30 языков.

Qwen2-72B набрала **84.2 MMLU** — превосходя Llama-3-70B (79.5) на +4.7%. Qwen2-7B показала **70.3 MMLU** — превосходя Llama-3-8B (66.6) на +3.7%. Модели стали одними из сильнейших open-weight LLM 2024 года, особенно в math и coding.

## Архитектура: decoder-only Transformer с современными оптимизациями

### Конфигурации моделей

| Параметр | 0.5B | 1.5B | 7B | 72B | 57B-A14B (MoE) |
|----------|------|------|-----|------|----------------|
| Hidden Size | 896 | 1,536 | 3,584 | 8,192 | 3,584 |
| Layers | 24 | 28 | 28 | 80 | 28 |
| Query Heads | 14 | 12 | 28 | 64 | 28 |
| **KV Heads** | **2** | **2** | **4** | **8** | **4** |
| Head Size | 64 | 128 | 128 | 128 | 128 |
| Intermediate Size | 4,864 | 8,960 | 18,944 | 29,568 | 2,560 |
| Vocabulary | 151,646 | 151,646 | 151,646 | 151,646 | 151,646 |
| Training Tokens | **12T** | **7T** | **7T** | **7T** | **4.5T** |

### Grouped Query Attention (GQA)

Все модели Qwen2 используют **GQA** (в отличие от Qwen 1, который использовал MHA). KV heads значительно меньше query heads: например, Qwen2-7B имеет 28 query heads и только 4 KV heads (ratio 7:1).

**Почему это важно:** GQA сокращает KV-cache при inference в $\frac{\text{num\_query\_heads}}{\text{num\_kv\_heads}}$ раз. Для Qwen2-72B с 8 KV heads (вместо 64) — это **8x экономия памяти** при long-context inference.

### Dual Chunk Attention (DCA) + YARN

Уникальная техника для **длинных контекстов** (до 131,072 токенов):

1. **DCA** разбивает длинные последовательности на chunks фиксированной длины
2. Внутри chunk — стандартный attention
3. Между chunks — специальный механизм сохранения relative positional information
4. **YARN** (Yet Another RoPE extensioN) rescales attention weights для лучшей length extrapolation

Если входная последовательность помещается в один chunk — DCA эквивалентна обычному attention (нет overhead'а на коротких последовательностях).

### Long-Context Training

В финальной фазе pre-training:
- Context length увеличен: **4096 → 32768**
- RoPE base frequency изменена: **10,000 → 1,000,000**
- Добавлен большой объём high-quality длинных текстов

С DCA + YARN модель обрабатывает до **131,072 токенов** с минимальной деградацией perplexity.

### Другие архитектурные решения

- **SwiGLU activation**: $\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$
- **RoPE positional embeddings**: rotary encoding позиций
- **QKV bias**: bias в attention projections (нетипично — большинство моделей убирают bias)
- **RMSNorm + pre-normalization**: стандарт для стабильного обучения
- **BPE tokenizer**: byte-level, 151,646 tokens, высокая compression efficiency для multilingual

## MoE модель: Qwen2-57B-A14B

### Fine-Grained Experts

Ключевое отличие от Mixtral: **fine-grained experts** — каждый эксперт **меньше** стандартного FFN, но одновременно активируется **больше** экспертов:

| Параметр | Mixtral 8x7B | Qwen2-57B-A14B |
|----------|-------------|----------------|
| Routed experts | 8 | **64** |
| Activated per token | 2 | **8** |
| Shared experts | 0 | **8** |
| Expert intermediate size | Full FFN | **2,560** (мало!) |

**Больше мелких экспертов** = **richer combinations**. При 64 экспертах и 8 активных — это $\binom{64}{8}$ = 4.4 миллиарда возможных комбинаций vs $\binom{8}{2}$ = 28 у Mixtral.

### Shared + Routed Experts

8 shared experts **всегда активны** (для общих знаний), 8 из 64 routed experts выбираются **gated network** для каждого токена. Это обеспечивает баланс между general capability и специализацией.

### Expert Initialization: upcycling

Эксперты инициализируются из dense модели (Qwen2-7B):
1. FFN реплицируется $\lceil \frac{n \times h_E}{h_{FFN}} \rceil$ раз
2. Параметры **shuffled** по intermediate dimension для diversification
3. **50% параметров каждого эксперта** случайно переинициализируются

Это даёт экспертам **разнообразие** при старте, ускоряя специализацию при обучении.

## Pre-training Data: 7T+ токенов

### Улучшения над Qwen 1.5

| Аспект | Qwen 1.5 | Qwen2 |
|--------|----------|-------|
| Tokens | 3T | **7T** (72B), **12T** (0.5B) |
| Languages | ~15 | **~30** |
| Code quality | Standard | **Усиленная фильтрация + синтез** |
| Math quality | Standard | **Rejection sampling + execution feedback** |
| Filtering | Basic | **Qwen-based model filtering** |

Интересный finding: увеличение до 12T токенов (с пониженным quality threshold) **не дало** значительного улучшения над 7T. Авторы заключают: объём данных не всегда компенсирует снижение качества.

## Post-Training: SFT + двухэтапный RLHF

### Supervised Fine-Tuning

- **500K+ примеров**: instruction following, coding, math, reasoning, role-playing, multilingual, safety
- 2 эпохи, sequence length 32,768, LR от 7e-6 до 7e-7
- Сложный data pipeline: InsTag (ontology extraction), instruction evolution, rejection sampling

### RLHF: двухэтапный DPO

Уникальный подход — **не PPO, а DPO** (Direct Preference Optimization):

**Stage 1: Offline DPO** на pre-compiled preference dataset:

$$\mathcal{L}_\text{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y^+ | x)}{\pi_\text{ref}(y^+ | x)} - \beta \log \frac{\pi_\theta(y^- | x)}{\pi_\text{ref}(y^- | x)}\right)$$

**Stage 2: Online DPO** — модель **сама генерирует** responses, reward model выбирает лучший/худший, и DPO applied в реальном времени.

**Online Merging Optimizer** — техника для уменьшения alignment tax (деградации base capabilities при alignment).

## Ключевые результаты

### Qwen2-72B vs конкуренты (base model)

| Benchmark | Mixtral-8x22B | Llama-3-70B | **Qwen2-72B** |
|-----------|---------------|-------------|---------------|
| **MMLU** | 77.8 | 79.5 | **84.2** |
| MMLU-Pro | 49.5 | 52.8 | **55.6** |
| BBH | 78.9 | 81.0 | **82.4** |
| **HumanEval** | 46.3 | 48.2 | **64.6** |
| **GSM8K** | 83.7 | 83.0 | **89.5** |
| MATH | 41.7 | 42.5 | **51.1** |
| C-Eval (Chinese) | 54.6 | 65.2 | **91.0** |

### Qwen2-7B vs конкуренты

| Benchmark | Mistral-7B | Gemma-7B | Llama-3-8B | **Qwen2-7B** |
|-----------|-----------|----------|-----------|-------------|
| MMLU | 64.2 | 64.6 | 66.6 | **70.3** |
| HumanEval | 29.3 | 37.2 | 33.5 | **51.2** |
| GSM8K | 52.2 | 46.4 | 56.0 | **79.9** |
| MATH | 13.1 | 24.3 | 20.5 | **44.2** |
| C-Eval | 47.4 | 43.6 | 49.5 | **83.2** |

**Qwen2-7B массивно превосходит** конкурентов в math (GSM8K: +23.9 vs Llama-3-8B) и coding (HumanEval: +17.7 vs Llama-3-8B).

### Qwen2-72B-Instruct

- **MT-Bench**: 9.12 (на уровне лучших closed-source)
- **Arena-Hard**: 48.1
- **LiveCodeBench**: 35.7

## Почему Qwen2 важна

### 1. Сильнейший open-weight в math/coding

Qwen2 показала, что **целенаправленная курация** code и math данных даёт взрывной рост performance в этих областях. GSM8K 89.5% для 72B — это уровень GPT-4.

### 2. Полный спектр размеров

От 0.5B (smartphones) до 72B (GPU кластеры) + MoE — покрывает **все** deployment сценарии. Ни одно другое семейство не предлагало такого разнообразия.

### 3. Fine-grained MoE

Подход с множеством мелких экспертов (64 routed + 8 shared) — альтернатива Mixtral'овскому подходу с малым числом крупных экспертов. Qwen2-57B-A14B конкурирует с 34B dense моделями при активации лишь 14B.

### 4. Multilingual (~30 языков)

91.0% C-Eval (Chinese) — уровень, недоступный Western-centric моделям. Сильная поддержка русского, корейского, японского, арабского, и других.

### 5. DPO вместо PPO

Qwen2 показала, что **DPO (offline + online)** — жизнеспособная альтернатива PPO для alignment. DPO проще в реализации, не требует reward model inference при обучении, и даёт сопоставимые результаты.

## Key papers

- [[02 Areas/ML & DL/Papers/Qwen2|Qwen2]] — серия LLM 0.5B-72B + MoE

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектурная основа
- [[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral of Experts]] — альтернативный MoE подход
- [[02 Areas/ML & DL/Concepts/Architectures/Gemma|Gemma]] — конкурент от Google
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — alignment метод
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]] — umbrella alignment technique
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — RoPE + YARN
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — GQA оптимизирует KV-cache

## Дополнительные ресурсы

- [Qwen2 paper (arXiv)](https://arxiv.org/abs/2407.10671) — полный технический отчёт
- [Qwen GitHub](https://github.com/QwenLM/Qwen2) — код, примеры, quantization
- [HuggingFace Qwen2-72B](https://huggingface.co/Qwen/Qwen2-72B) — крупнейший чекпоинт
- [ModelScope Qwen](https://modelscope.cn/organization/qwen) — чекпоинты для Chinese community
