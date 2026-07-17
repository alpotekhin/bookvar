---
title: "Baichuan 2: Open Large-scale Language Models"
url: "https://arxiv.org/abs/2309.10305"
authors: [Aiyuan Yang, Bin Xiao, Bingning Wang, Baichuan Inc.]
year: 2023
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/baichuan-2/paper.pdf|PDF]]"
concepts:
  - LLM Pre-training
  - Scaling Laws
  - RLHF
  - Chinese LLM
---

# Baichuan 2: Open Large-scale Language Models

## TL;DR

Baichuan 2 — серия open-source bilingual (Chinese-English) моделей на 7B и 13B параметров, обученных на 2.6T токенов (на тот момент — максимум для open-source). Достигает или превосходит другие open-source модели аналогичного размера на MMLU, CMMLU, GSM8K, HumanEval. Особенно силён в vertical domains (медицина, право). Выпущены все промежуточные checkpoints для исследования training dynamics.

## Проблема

В 2023 году большинство мощных LLM (GPT-4, PaLM-2, Claude) были closed-source. Open-source модели (LLaMA, Falcon, MPT) фокусировались на английском и имели ограниченные capabilities для других языков, особенно китайского. Это тормозило развитие и применение LLM в Chinese-dominant контекстах.

## Метод

### Pre-training Data

Диверсифицированный корпус: web pages, books, research papers, code. Процессинг:
1. **LSH + dense embedding** кластеризация и дедупликация на trillion-scale (за часы)
2. Quality scoring на уровне документов, параграфов и предложений
3. Score-based sampling для pre-training

### Архитектура

Базовый Transformer с модификациями:

**Tokenizer:** BPE через SentencePiece, vocab 125,696 (vs 64K у Baichuan 1). Числа разбиваются на отдельные digits. No dummy prefix (не помогает для Chinese).

**Positional Embeddings:**
- **Baichuan 2-7B:** RoPE (стандарт, хорошая поддержка FlashAttention)
- **Baichuan 2-13B:** ALiBi (лучшая extrapolation, но хуже оптимизирован)

Авторы отмечают: в preliminary experiments выбор между RoPE и ALiBi не влиял значимо на performance.

**Activations:** SwiGLU с reduced hidden size до $\frac{8}{3}h$ (вместо стандартных $4h$) для компенсации тройной матрицы.

**Normalization:** Pre-LayerNorm (RMSNorm implementation) — более стабильный к warm-up schedule.

**NormHead:** Нормализация output embeddings. Два преимущества:
1. Стабилизация training dynamics (norm редких токенов уменьшается при тренировке, что дестабилизирует)
2. Semantic info в cosine similarity, а не в L2 distance — NormHead убирает L2 компонент из logits

**Max-z loss:** $L_{\text{max-z}} = 2 \times 10^{-4} \cdot z^2$, где $z$ — maximum logit. Стабилизирует тренировку, делает инференс robust к repetition penalty hyperparameters (большие logits -> softmax sensitivity).

### Hyper-parameters

| Model | Hidden | FFN | Heads | Layers | Seq Len | Max LR |
|-------|--------|-----|-------|--------|---------|--------|
| 7B | 4096 | 11008 | 32 | 32 | 4096 | 2e-4 |
| 13B | 5120 | 13696 | 40 | 40 | 4096 | 1.5e-4 |

Training: AdamW (beta1=0.9, beta2=0.95), weight decay 0.1, grad clip 0.5, 2000 linear warmup steps, cosine decay. BFloat16 mixed precision (full precision для position embeddings из-за коллизий в arange при integers > 256).

### Scaling Laws

Обучили модели от 10M до 3B на до 1T токенов. Fitted power law:

$$L_C = a \times C^b + L_\infty$$

Предсказания финального loss для 7B и 13B на 2.6T токенов **совпали с высокой точностью** — scaling laws работают.

### Alignment (RLHF)

**SFT:** 100K+ supervised samples с cross-validation качества.

**Reward Model:**
- 3-уровневая taxonomy: 6 primary -> 30 secondary -> 200+ tertiary categories
- Responses только от Baichuan 2 model family (responses от других моделей не улучшают RM!)
- Test accuracy: от 54.5% (gap 1, "unsure") до 81.5% (gap 5, "significantly better")

**PPO:** 4 модели (actor, reference, reward, critic). Critic warmup 20 steps. KL penalty beta=0.2 -> decay to 0.005. PPO clip epsilon=0.1. 350 итераций.

### Infrastructure

- Tensor parallelism + ZeRO data parallelism
- Machine-level elasticity (dynamic GPU scaling)
- Topology-aware distributed training (minimize cross-switch communication)
- Hybrid hierarchical ZeRO partitioning
- 1024 NVIDIA A800 GPUs, >180 TFLOPS efficiency

## Ключевые результаты

### General Benchmarks

| Benchmark | Baichuan 2-7B | Baichuan 2-13B | LLaMA 2-13B | GPT-3.5 |
|-----------|--------------|----------------|-------------|---------|
| MMLU | 54.16 | 59.17 | 55.09 | 68.54 |
| CMMLU | 57.07 | 61.97 | 37.99 | 54.06 |
| C-Eval | 54.00 | 58.10 | 35.80 | 51.10 |
| GSM8K | 24.49 | **52.77** | 28.89 | 57.77 |
| HumanEval | 18.29 | 17.07 | 15.24 | 52.44 |

### Прогресс vs Baichuan 1

- MMLU: +30% для 7B
- GSM8K: почти 2x
- HumanEval: почти 2x
- Объём training data: 2x (2.6T vs 1.2T)

### Domain-specific

Baichuan 2 outperforms другие open-source на медицинских (MedQA) и юридических (JEC-QA) задачах.

## Мои заметки

**NormHead — простая, но мощная идея.** Проблема: norm редких токенов в embedding head уменьшается при тренировке, а logits = dot product = L2 + cosine. NormHead убирает L2 компонент, оставляя чистый cosine similarity. Плюс стабилизация training dynamics. Это было потом picked up другими моделями.

**Max-z loss** — тоже elegant: penalize large logits через $z^2$ term. Motivation: repetition penalty в inference (multiply logits by scalar) ломается при больших logits из-за softmax sensitivity. Маленькая добавка к loss ($2 \times 10^{-4}$) решает проблему. Сходная идея z-loss есть в PaLM.

**RM только на своих моделях** — интересное наблюдение. Responses от других LLM или open-source datasets не помогают. Это говорит о внутренней consistency model family и importance of distribution matching для reward modeling.

**2.6T токенов для 7B/13B** — значительный overtrain по Chinchilla optimal. Но как показали LLaMA, Qwen и другие, overtraining для моделей inference-constrained размеров — правильная стратегия. Даже после 2.6T performance продолжал расти.

**ALiBi vs RoPE на разных размерах** — unusual design choice использовать разные PE для 7B и 13B. Авторы говорят "не влияет на performance" — тогда зачем? Вероятно, legacy от Baichuan 1 + желание дать community обе версии для research.

**Все промежуточные checkpoints** — ценный вклад в research, позволяет изучать training dynamics и emergent abilities.

**Scaling laws точно предсказали loss** — fitted power law $L_C = a \times C^b + L_\infty$ предсказала финальный loss для 7B и 13B с high accuracy. Модели от 10M до 3B на до 1T токенов. Это говорит о том, что scaling laws работают и для bilingual data distributions, не только для English-dominant corpora.

**Infrastructure: A800 GPUs** — Baichuan тренировалась на A800 (ограниченная версия A100 для China). 1024 GPU, >180 TFLOPS efficiency. Topology-aware distributed training + hybrid hierarchical ZeRO partitioning — серьёзная systems работа для компенсации hardware limitations.

**3-уровневая taxonomy для RLHF:** 6 primary -> 30 secondary -> 200+ tertiary categories. Это самая detailed classification для reward model data среди опубликованных papers того периода. Granular taxonomy помогает reward model generalize better и обеспечивает coverage всех типов user needs.

**SFT: 100K+ samples** — значительно больше, чем Yi (< 10K) и LIMA (1K), но всё ещё managed scale с cross-validation quality checks. Баланс между coverage и quality.

**BFloat16 precision challenges** — детальное описание проблем: torch.arange коллизии для integers > 256 из-за low precision BFloat16. Solution: full precision для position embeddings. Practical insight для training practitioners.

**Domain-specific strength** (MedQA, JEC-QA) — advantage для Chinese market, где медицинские и юридические assistants — high-value use case. Open-source foundation model, оптимизированный для vertical domains, привлекает enterprise adoption.

**Compression rate tokenizer:** 0.498 vs LLaMA 2 1.037 — Baichuan 2 tokenizer вдвое эффективнее для Chinese+English mixed text. Vocabulary 125,696 vs 32,000 у LLaMA 2. Это прямой factor в training и inference speed.

**PPO training details:** 4-model setup (actor, reference, reward, critic). KL penalty coefficient beta starts at 0.2, decays to 0.005. 350 iterations. Critic warmup 20 steps. Это стандартный но well-tuned RLHF setup. Decay KL penalty — позволяет модели постепенно diverge от reference policy, что can unlock higher performance.

**Safety через все стадии** — не только alignment, но и pre-training (data filtering) и inference. Multi-stage safety approach: pre-training data cleansing -> SFT safety data -> RM с Helpful+Harmless objectives -> PPO safety reinforcement. Holistic подход.

**Baichuan 2-13B GSM8K: 52.77** — почти 2x vs Baichuan 1-13B (26.76) и значительно лучше LLaMA 2-13B (28.89). Math improvement — один из strongest gains от Baichuan 1 к Baichuan 2, driven by more training data и improved data quality.

**Open-source checkpoints от 200B до 2.6T** — unique contribution. Позволяет исследователям изучить, как capabilities emerge с training: на каком количестве токенов появляется math ability? Code? Это data, которая обычно недоступна community.
