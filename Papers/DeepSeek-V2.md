---
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
url: "https://arxiv.org/abs/2405.04434"
authors: [DeepSeek-AI]
year: 2024
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/deepseek-v2/paper.pdf|PDF]]"
concepts:
  - "[[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]]"
  - Mixture-of-Experts
  - KV Cache Compression
  - Sparse Models
---

# DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model

## TL;DR

DeepSeek-V2 — MoE-модель на 236B параметров (21B активных на токен) с двумя ключевыми архитектурными инновациями: **Multi-head Latent Attention (MLA)**, которая сжимает KV cache на 93.3% через low-rank проекцию, и **DeepSeekMoE** с fine-grained экспертами и shared experts. Результат: экономия 42.5% стоимости тренировки по сравнению с DeepSeek 67B, 5.76x throughput при генерации, топ-уровень среди open-source моделей.

## Проблема

Стандартный Multi-Head Attention (MHA) создаёт огромный KV cache при инференсе — для каждого токена нужно хранить $2 n_h d_h l$ элементов (heads x dim x layers). Это bottleneck для batch size и длины контекста. Существующие решения (GQA, MQA) уменьшают cache, но жертвуют качеством.

Вторая проблема: обучение dense-моделей масштаба 100B+ требует колоссальных ресурсов. MoE позволяет активировать малую часть параметров, но стандартные архитектуры (GShard) не достигают полного потенциала из-за грубой сегментации экспертов.

## Метод

### Multi-head Latent Attention (MLA)

Ключевая идея — **low-rank joint compression** для keys и values. Вместо хранения полных K, V векторов для каждой головы, MLA сжимает их в единый латентный вектор:

$$c_t^{KV} = W^{DKV} h_t$$

$$k_t^C = W^{UK} c_t^{KV}, \quad v_t^C = W^{UV} c_t^{KV}$$

где $c_t^{KV} \in \mathbb{R}^{d_c}$ — сжатый латентный вектор, $d_c \ll d_h n_h$. При инференсе нужно кэшировать только $c_t^{KV}$, а матрицы $W^{UK}$ и $W^{UV}$ можно "поглотить" в $W^Q$ и $W^O$ соответственно — даже не нужно восстанавливать полные K, V!

**Проблема с RoPE:** позиционное кодирование несовместимо с low-rank сжатием, потому что RoPE-матрица встаёт между $W^Q$ и $W^{UK}$, и их нельзя перемножить заранее. Решение — **Decoupled RoPE**: отдельные query $q_t^R$ и shared key $k_t^R$ несут позиционную информацию, а content-часть attention работает через сжатые латенты.

Итоговый KV cache для MLA: $(d_c + d_h^R) l \approx \frac{9}{2} d_h l$ — эквивалент GQA с 2.25 группами, но **performance выше чем у полного MHA**.

### DeepSeekMoE

Две ключевые идеи:
1. **Fine-grained expert segmentation** — 160 routed экспертов (вместо стандартных 8-16), из которых активируются top-6. Мелкие эксперты лучше специализируются.
2. **Shared expert isolation** — 2 эксперта обрабатывают все токены, снижая redundancy в routed экспертах.

Формула FFN:

$$h_t' = u_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(u_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)}(u_t)$$

Дополнительные механизмы для эффективного параллелизма:
- **Device-limited routing** — target эксперты каждого токена ограничены M устройствами (M >= 3 достаточно)
- **3 типа balance loss**: expert-level, device-level, communication balance
- **Token-dropping** с capacity factor 1.0 для выравнивания нагрузки

### Параметры модели

| Параметр | Значение |
|----------|----------|
| Layers | 60 |
| Hidden dim | 5120 |
| Attention heads | 128 (dim 128 per head) |
| KV compression dim | 512 |
| Query compression dim | 1536 |
| Shared experts | 2 |
| Routed experts | 160 (top-6 activated) |
| Total params | 236B |
| Active params | 21B |
| Pre-training tokens | 8.1T |
| Context length | 128K |

### Pre-training

Корпус: 8.1T токенов, multi-source (web, books, code, research papers), Chinese токены ~12% больше чем English. Tokenizer: BBPE, vocab 100K (тот же что в DeepSeek 67B). Корпус расширен по сравнению с предыдущей версией, особенно за счёт Chinese data, с улучшенным quality-based filtering и debiasing.

Long context extension через YaRN (Yet another RoPE extension) до 128K. Двухэтапный continue training: сначала увеличиваем длину, потом alignment на длинных контекстах.

### Alignment

**SFT:** 1.5M conversational sessions по доменам (math, code, writing, reasoning, safety). Высокое качество данных приоритетнее количества.

**RL (GRPO):** Group Relative Policy Optimization — variant of RL без critic model. Для каждого prompt генерируется группа ответов, rewards нормализуются внутри группы. Преимущества: (1) нет need для value network, (2) relative rewards стабильнее absolute, (3) дешевле PPO по compute. Этот подход стал signature technique DeepSeek и был развит далее в V3 и R1.

**DeepSeek-V2-Lite (16B / 2.4B active)** — уменьшенная версия с теми же архитектурными инновациями (MLA + DeepSeekMoE) для community research.

## Ключевые результаты

### Benchmarks (base model)

| Benchmark | DeepSeek-V2 | DeepSeek 67B | LLaMA 3 70B | Mixtral 8x22B |
|-----------|-------------|--------------|-------------|---------------|
| MMLU | 78.5 | 71.3 | 78.9 | 77.8 |
| BBH | 78.9 | 68.7 | 81.0 | 78.9 |
| HumanEval | 48.8 | 45.1 | 48.2 | 46.3 |
| GSM8K | 79.2 | 63.4 | 83.0 | 78.6 |
| MATH | 43.6 | 18.7 | 42.5 | 41.7 |

### Эффективность

- **Стоимость тренировки:** 42.5% экономии vs DeepSeek 67B
- **KV cache:** сокращение на 93.3% (с ~320 KB/token до ~21 KB/token)
- **Throughput генерации:** 5.76x vs DeepSeek 67B
- **Aligned версия (RL):** 38.9 AlpacaEval 2.0, 8.97 MT-Bench

### Chat модель

DeepSeek-V2 Chat (RL) — топ среди open-source: 7.91 на AlignBench (лучше всех open-source и большинства closed-source на китайском).

## Мои заметки

**Почему это важно:** MLA — это по сути архитектурный прорыв, который показывает, что можно одновременно уменьшить KV cache и улучшить качество. Стандартная дилемма "cache vs quality" (MHA хорош, но тяжёлый; MQA лёгкий, но слабый) решена через elegant подход с латентным сжатием. Этот подход потом переехал в DeepSeek-V3 и стал де-факто стандартом для всей линейки DeepSeek.

**Decoupled RoPE** — красивое инженерное решение. Проблема несовместимости RoPE с low-rank compression неочевидна: если применить RoPE к сжатым keys, матрицу $W^{UK}$ нельзя поглотить в $W^Q$ из-за некоммутативности матричного умножения. Выход — вынести позиционную часть в отдельные query/key.

**DeepSeekMoE** — fine-grained эксперты (160 штук) vs грубые (8-16 в GShard/Mixtral) дают лучшую специализацию. Shared experts решают проблему redundancy — общие знания хранятся отдельно, routed эксперты могут фокусироваться на специфике.

**Три balance loss** — практичное решение для production: expert-level предотвращает routing collapse, device-level выравнивает compute, communication balance — выравнивает сетевой трафик. Без этого MoE на кластере теряет эффективность.

**Alignment:** используют GRPO (Group Relative Policy Optimization) вместо PPO — без critic model, только relative rewards внутри группы samples. Проще и дешевле.

**Сравнение attention mechanisms (из ablation):**
- MHA: сильный performance, но 2x KV cache overhead
- GQA: moderate performance, reduced cache
- MQA: слабый performance, minimal cache
- **MLA: strongest performance + minimal cache** — Pareto-optimal решение

Это ключевое преимущество: авторы не просто уменьшили cache (это делают GQA/MQA), а одновременно **улучшили** quality. Low-rank compression работает как implicit regularization для attention.

**Влияние на индустрию:** MLA стала blueprint для следующих моделей DeepSeek (V3, R1). Device-limited routing и communication balance loss показывают, как масштабировать MoE в production — это не только про architecture, но и про systems engineering.

**8.1T токенов** — значительное увеличение vs DeepSeek 67B, особенно Chinese data. Авторы описывают процесс восстановления "mistakenly deleted data" при улучшении cleaning pipeline — инсайт: aggressive filtering может быть counterproductive.

**DeepSeek-V2-Lite** — 16B/2.4B active. Полезный research artifact: позволяет воспроизвести MLA и DeepSeekMoE на меньшем compute budget.

**Tokenizer и data composition:** BBPE vocab 100K, Chinese tokens ~12% больше English. Оптимизация для bilingual use case. В улучшенном pipeline "recovered mistakenly deleted data" — aggressive filtering может удалять полезные данные, и итеративное уточнение pipeline критично.

**Context extension до 128K** — используется YaRN (Yet another RoPE extension): scale=40, alpha=1, beta=32. Двухэтапный continue training (32K -> 128K) с reduced batch size на втором этапе. Long context alignment (LongAlign from GLM team) применяется для обеспечения quality на длинных контекстах.

**Chat evaluations:** AlpacaEval 2.0: 38.9% length-controlled win rate, MT-Bench: 8.97, AlignBench: 7.91 — на уровне GPT-4 All Tools. На AlignBench (Chinese) outperforms все open-source **и** большинство closed-source моделей. Chinese alignment — clear strength.

**MATH: 43.6 vs DeepSeek 67B 18.7** — более чем 2x improvement при сокращении training cost. Это показывает, что MoE + better data + alignment recipe могут давать superlinear improvements.

**Expert specialization visualization:** В оригинальной DeepSeekMoE paper авторы показали, что fine-grained experts действительно специализируются на разных topics/capabilities. При 160 экспертах каждый может cover narrow domain, что impossible при 8-16 coarse experts.

**Roadmap к DeepSeek-V3:** MLA и DeepSeekMoE, введённые в V2, стали foundation для V3 (671B / 37B active). V3 добавил Multi-Token Prediction и FP8 training, но core architecture — V2's MLA + MoE. Это подтверждает, что architectural bets в V2 были correct и scalable.

**Query compression** ($d_c' = 1536$) — даже queries сжимаются через low-rank projection, хотя это не уменьшает KV cache. Цель — reduce activation memory при training, что позволяет larger batch size и faster training.
