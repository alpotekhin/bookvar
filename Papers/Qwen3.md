---
title: "Qwen3 Technical Report"
url: "https://arxiv.org/abs/2505.09388"
authors: [Qwen Team, Alibaba]
year: 2025
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/qwen3/paper.pdf|PDF]]"
concepts:
  - Mixture-of-Experts
  - Reinforcement Learning
  - Chain-of-Thought
  - Multilingual Models
---

# Qwen3 Technical Report

## TL;DR

Qwen3 — серия LLM от Alibaba, включающая dense (0.6B-32B) и MoE (30B-A3B, 235B-A22B) модели. Ключевая инновация: **unified thinking/non-thinking modes** в одной модели с thinking budget контролем. Pre-training на 36T токенах, 119 языков. Flagship 235B-A22B достигает 85.7 AIME'24, 70.7 LiveCodeBench v5, 2056 CodeForces — state-of-the-art среди open-source.

## Проблема

До Qwen3 пользователям приходилось выбирать между chat-моделями (GPT-4o, Qwen2.5) для быстрых ответов и reasoning-моделями (QwQ, o1) для сложных задач. Переключение между моделями неудобно и дорого. Кроме того, Qwen2.5 поддерживал только 29 языков.

## Метод

### Архитектура

Dense модели повторяют Qwen2.5: GQA, SwiGLU, RoPE, RMSNorm с pre-normalization. Изменения от Qwen2:
- **Убран QKV-bias**, добавлен **QK-Norm** для стабильности
- MoE модели: 128 total experts, 8 activated per token
- **Без shared experts** (в отличие от Qwen2.5-MoE)
- **Global-batch load balancing loss** вместо per-sample

Flagship: Qwen3-235B-A22B — 94 layers, 64 Q-heads / 4 KV-heads, 128 experts (8 active).

### Pre-training: 3 стадии, 36T токенов

**Stage 1 — General (30T+ tokens, seq 4096):** Основа языковых и world knowledge. 119 языков и диалектов.

**Stage 2 — Reasoning (5T tokens, seq 4096):** Увеличена доля STEM, code, reasoning, synthetic data. Ускоренный learning rate decay.

**Stage 3 — Long Context (seq 32768):** Сотни миллиардов токенов. RoPE base frequency: 10K -> 1M (ABF). YARN + Dual Chunk Attention для 4x context extension при инференсе.

### Расширение данных

Innovative подходы к scaling data:
- **Qwen2.5-VL** fine-tuned для OCR из PDF документов — извлечение text из документов
- **Qwen2.5-Math / Qwen2.5-Coder** для генерации synthetic data (textbooks, QA, instructions, code snippets)
- Multilingual annotation system: 30T+ токенов аннотированы по educational value, domain, safety

### Post-training: 4 стадии

**Stage 1 — Long-CoT cold start:** SFT на reasoning data с длинными цепочками рассуждений.

**Stage 2 — Reasoning RL:** Reinforcement learning фокусированный на math и code (verifiable tasks).

**Stage 3 — Thinking mode fusion:** Объединение data с/без reasoning paths для unified модели. Модель учится переключаться между thinking и non-thinking mode.

**Stage 4 — General RL:** Reinforcement learning across широкий спектр downstream задач.

### Strong-to-weak distillation

Для малых моделей — distillation из flagship вместо RL. Off-policy + on-policy knowledge transfer. Distillation **значительно превосходит RL** по performance и training efficiency для малых моделей.

### Thinking Budget

Пользователь может контролировать вычислительный бюджет reasoning: сколько thinking-токенов модель может использовать. Баланс latency vs performance для задач разной сложности.

## Ключевые результаты

### Pre-trained base models

| Benchmark | Qwen3-235B-A22B | DeepSeek-V3 | Llama-4-Maverick | Qwen2.5-72B |
|-----------|-----------------|-------------|------------------|-------------|
| MMLU | **87.81** | 87.19 | 85.16 | 86.06 |
| MMLU-Pro | **68.18** | 59.84 | 63.91 | 58.07 |
| BBH | **88.87** | 86.22 | 83.62 | 86.30 |
| MATH | **71.84** | 62.62 | 63.32 | 62.12 |
| EvalPlus | **77.60** | 63.75 | 68.38 | 65.93 |
| GSM8K | **94.39** | 87.57 | 87.72 | 91.50 |

При этом 235B total / 22B active — значительно меньше DeepSeek-V3 (671B / 37B) и Llama-4-Maverick (402B / 17B).

### Post-trained (thinking mode)

- **AIME'24:** 85.7 (vs o1: 74.4, DeepSeek-R1: ~79)
- **AIME'25:** 81.5
- **LiveCodeBench v5:** 70.7
- **CodeForces:** 2056 rating
- **BFCL v3:** 70.8

### Scaling efficiency

Qwen3 dense модели ~= Qwen2.5 на одну шкалу выше:
- Qwen3-1.7B ~= Qwen2.5-3B
- Qwen3-4B ~= Qwen2.5-7B
- Qwen3-8B ~= Qwen2.5-14B
- Qwen3-14B ~= Qwen2.5-32B
- Qwen3-32B ~= Qwen2.5-72B

MoE: 1/5 activated parameters = comparable performance с dense.

## Мои заметки

**Unified thinking/non-thinking** — практически полезная инновация. Вместо двух deployment (chat + reasoning), одна модель покрывает оба сценария. Thinking budget даёт fine-grained control, что важно для production, где latency критична.

**36T токенов** — это много. Qwen2.5 использовал 18T. Двукратное увеличение + 119 языков (vs 29) говорит о серьёзном инвестировании в data pipeline. Использование VLM для OCR из PDF — creative подход к data scaling.

**Убрали shared experts** — интересное расхождение с DeepSeek (у которых shared experts — key design choice). Qwen3 вместо этого полагается на global-batch load balancing loss для expert specialization. Empirically работает — результаты сильные.

**Distillation > RL для малых моделей** — важный практический вывод. При наличии сильного teacher, distillation эффективнее по compute и даёт лучший результат. Это объясняет, почему маленькие Qwen3 так сильны.

**Qwen3-32B-Base превосходит Qwen2.5-72B-Base** на coding и reasoning задачах при вдвое меньшем размере — data quality + training strategy > raw scale.

**Apache 2.0 лицензия** для всей серии — значимо для коммерческого использования.

**4-stage post-training pipeline** — хорошо структурированный подход. Разделение на reasoning-focused RL (stage 2) и general RL (stage 4) позволяет сначала развить сильные reasoning skills, а потом сбалансировать их с general capabilities. Если делать всё в одном RL run, reasoning и general objectives могут конфликтовать.

**QK-Norm вместо QKV-Bias** — architectural change от Qwen2. QKV-bias может приводить к training instability при scale. QK-Norm (нормализация Q и K перед dot product) стабилизирует training для больших моделей. Аналогичный подход используется в ViT и других архитектурах.

**Global-batch load balancing loss** — отличие от per-sample load balancing. Global-batch позволяет экспертам быть загружены неравномерно на уровне отдельных samples, но balanced в aggregate. Это даёт лучшую expert specialization: каждый sample может routing к оптимальным экспертам, а balance поддерживается статистически.

**Instance-level data mixture optimization** — Qwen3 оптимизирует data mixture не на уровне data source (web, books, code), а на уровне отдельных instances через fine-grained data labels. Это требует колоссальной аннотации (30T+ tokens annotated!), но даёт более granular control.

**Dual Chunk Attention + YARN** — комбинация для 4x context extension при инференсе. DCA разбивает sequence на chunks и обрабатывает intra-chunk и inter-chunk attention отдельно, что позволяет extrapolate далеко за пределы training context length.

**Qwen3-30B-A3B** — интересная MoE точка: всего 3B activated из 30B total. При 1/10 activated params от Qwen2.5-32B dense достигает comparable performance. Extreme sparsity ratio, демонстрирующий эффективность MoE architecture.

**3-stage pre-training** — structured approach. Stage 1 (General, 30T) -> Stage 2 (Reasoning, 5T) -> Stage 3 (Long Context). Ключевая деталь: Stage 2 ускоряет learning rate decay, что позволяет модели стабилизироваться на reasoning-heavy данных без overshooting.

**119 языков и диалектов** — 4x расширение vs Qwen2.5 (29 языков). При 36T токенах хватает data budget для meaningful coverage даже minor languages. Multilingual annotation system позволяет filtering на уровне instance — не все Chinese/English данные одинаково полезны.

**Qwen2.5-VL для OCR из PDF** — meta-использование предыдущего поколения моделей для создания training data нового поколения. VLM модель fine-tuned для text extraction, затем Qwen2.5 refinement для quality. Это self-improving data pipeline, который масштабируется с каждым поколением.

**Competitive with o1 and o3-mini** — в thinking mode Qwen3-235B-A22B shows 85.7 AIME'24, что превосходит o1 (74.4 по данным Kimi paper). Однако comparison не полностью fair — разные eval setups. Тем не менее, open-source reasoning models достигли parity с frontier closed-source.

**8 activated experts из 128** — sparsity 6.25%. DeepSeek-V2 использует 6 из 160 (3.75%). Qwen3 чуть менее sparse, но с более крупными экспертами. Отсутствие shared experts (vs DeepSeek) компенсируется global-batch load balancing, что позволяет некоторым экспертам специализироваться на "shared" knowledge органически.

**Thinking mode architecture** — не требует отдельных model weights. Единая модель переключается между modes через chat template или user instruction. Thinking tokens генерируются в специальных тегах, non-thinking mode просто пропускает thinking phase. Это elegant design: один checkpoint, два режима.

**BBPE tokenizer, vocab 151,669** — один из крупнейших в open-source. Покрывает 119 языков. Byte-level fallback обеспечивает robustness для unseen scripts.

**AIME'25: 81.5** — benchmark появившийся уже после training. Показывает genuine reasoning ability, а не memorization. Delta между AIME'24 (85.7) и AIME'25 (81.5) — reasonable drop для new competition problems.

**CodeForces rating 2056** — expert-level competitive programming. Для сравнения: median human competitive programmer ~1200. Модель outperforms vast majority of human participants.

**Post-training stages по порядку важности:** reasoning cold-start (foundation) -> reasoning RL (strength) -> mode fusion (flexibility) -> general RL (balance). Each stage builds on the previous, и порядок matters: reasoning first, generalization second.

**Model family completeness** — 8 моделей от 0.6B до 235B покрывают все deployment scenarios: edge devices (0.6B), mobile (1.7B), laptop (4B-8B), server (14B-32B), cloud (235B MoE). Единая architecture family упрощает tooling и optimization.

**Scaling laws for hyper-parameters** — Qwen3 использует scaling laws не только для prediction loss, но и для optimal learning rate и batch size. Systematic study relationship model architecture + data + stage -> optimal hyper-params. Это reduces expensive grid search на full-scale models.
