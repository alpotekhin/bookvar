---
title: "Phi-4 Technical Report"
url: https://arxiv.org/abs/2412.08905
authors: "Marah Abdin et al., Microsoft Research"
year: 2024
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/phi-4/source]]"
concepts: [Synthetic Data, Small Language Models, Data Quality, Direct Preference Optimization, Reasoning, Pivotal Token Search]
---
# Phi-4 Technical Report

## TL;DR

Phi-4 -- 14B параметров language model от Microsoft, обученная на **~10T токенов** с центральным фокусом на **data quality, а не scale**. 40% pretraining data -- synthetic, сгенерированные через multi-agent prompting, self-revision и instruction reversal. Phi-4 **превосходит свою teacher-модель GPT-4o** на STEM/reasoning бенчмарках (GPQA 56.1 vs 50.6, MATH 80.4 vs 74.6) и конкурирует с моделями в 5-25x больше себя. Ключевая инновация -- Pivotal Token Search для DPO и доказательство, что synthetic data **не просто distillation**, а может превзойти teacher.

## Problem

Предыдущие модели Phi-family (Phi-1/2/3) демонстрировали, что small models могут быть сильными, но преимущественно через distillation из GPT-4. Открытые вопросы: (1) Может ли synthetic data дать модели capabilities **beyond teacher**? (2) Как оптимально балансировать synthetic vs organic data? (3) Как бороться с benchmark contamination при heavy synthetic data use?

## Method

### Synthetic Data Pipeline (40% pretraining tokens)
- **50 типов synthetic datasets**, ~400B уникальных токенов (используются с ~14 эпохами).
- **Seed Curation**: web/code seeds фильтруются двухступенчато -- educational potential -> factual/reasoning scoring. Question datasets фильтруются через plurality voting (отсев слишком легких и слишком сложных).
- **Rewrite & Augment**: passages переписываются в exercises, discussions, structured reasoning tasks через multi-step prompting.
- **Self-revision**: iterative feedback loop, где модель критикует и улучшает собственные outputs.
- **Instruction Reversal**: для code -- берётся snippet, генерируется instruction, проверяется fidelity regenerated code vs original. Retain only high-fidelity pairs.
- **Validation**: code через execution tests, scientific data -- через extracted questions с groundedness check.

### Ключевое наблюдение: Synthetic > Web для reasoning
- Эксперименты на 7B ablation models: **12 эпох synthetic data > 4 эпохи synthetic + свежие web tokens** при одинаковом token budget. Synthetic не переобучается даже при 13.8 эпохах!
- Модель, обученная **только на synthetic**, уступает лишь на knowledge benchmarks (TriviaQA -14.8 от phi-3), но сильнее на reasoning (HumanEval +12.1, MATH +4.9).
- Web rewrites (synthetic перезаписи web-контента) частично закрывают knowledge gap.

### Data Mixture (финальная)

| Source | Fraction | Unique Tokens | Epochs |
|---|---|---|---|
| Synthetic | 40% | 290B | 13.8 |
| Web rewrites | 15% | 290B | 5.2 |
| Filtered web | 15% | 1.3T | 1.2 |
| Code | 20% | 820B | 2.4 |
| Acquired sources | 10% | 580B | 1.7 |

### Архитектура
- **Decoder-only Transformer, 14B params** -- минимальные изменения от phi-3-medium.
- Tiktoken tokenizer, vocab 100,352 (vs BPE в phi-3) -- better multilingual support.
- Full attention на 4K context (vs 2K sliding window в phi-3).
- Midtraining: extension до 16K context, RoPE base frequency 250K, 250B tokens, LR = peak/10.
- ~10T tokens total, peak LR 3e-4, constant weight decay 0.1, global batch size 5760.
- Hyperparameter tuning: interpolation from shorter horizon runs + stress testing LR warmup for stability.

### Post-Training: Three Stages

#### Stage 1: Supervised Fine-Tuning (SFT)
- **8B tokens** across diverse domains: math, coding, reasoning, conversation, model identity, safety.
- 40 languages для multilingual support.
- Standard chatml format: `<|im_start|>system<|im_sep|>...<|im_end|>`.
- Learning rate 1e-6.

#### Stage 2: Pivotal Token Search DPO
- **Новый метод генерации DPO pairs**. Идея: для данного prompt и correct response, найти "pivotal tokens" -- те, чья замена радикально меняет correctness output.
- Пример: в math solution, token "+" vs "-" -- pivotal, т.к. меняет всю downstream correctness.
- Создаются pairs (correct continuation, wrong continuation) в точке pivotal token.
- Это даёт **fine-grained training signal** -- модель учится specifically в критических decision points, а не на full response level.
- Data mix: 133K multiple-choice Q&A, 77K math, 16K Python, 22K other code, 3K safety (Table 7).

#### Stage 3: Judge-Guided DPO
- ~850K pairs от GPT-4o/GPT-4t/phi-4 responses.
- GPT-4o как judge: scores по accuracy, style, detail. Higher score = positive response.
- Data mix: 266K any-vs-any overall, 532K any-vs-any accuracy, 44K safety (Table 8).
- Both stages include small amount safety + anti-hallucination data.

### Decontamination
- Улучшенный процесс decontamination vs Phi-3.
- Удаление overlap с test sets из всех collected questions и solutions -- critical, т.к. benchmark variants существуют на Hugging Face и web platforms.
- **AMC 10/12 November 2024** -- свежий бенчмарк, выпущенный **после** всех training data cutoffs. Phi-4 набирает 91.8 avg score (из 150), обходя GPT-4o-mini (78.7) и Llama-3.3-70B (77.9).
- Reliance на "contamination-proof" benchmarks: GPQA (original questions, не на web), internal PhiBench (original prompts от team).

### Pretraining vs Phi-3 Comparison (pretrained only, before post-training)

| Metric | phi-4 vs phi-3-medium |
|---|---|
| MMLU | +3.0 |
| MMLU-Pro | +10.3 |
| MATH | +8.9 |
| HumanEval | +7.8 |
| GSM8K | +2.2 |
| MBPP | +6.8 |
| TriviaQA | -0.7 |

Significant gains на reasoning, minimal regression на knowledge -- confirms data quality > architecture changes.

## Key Results

### Post-trained Model (simple-evals framework, temp=0.5)

| Benchmark | phi-4 (14B) | phi-3 (14B) | GPT-4o-mini | Llama-3.3-70B | Qwen-2.5-72B | GPT-4o |
|---|---|---|---|---|---|---|
| MMLU | 84.8 | 77.9 | 81.8 | 86.3 | 85.3 | 88.1 |
| GPQA | **56.1** | 31.2 | 40.9 | 49.1 | 49.0 | 50.6 |
| MATH | **80.4** | 44.6 | 73.0 | 66.3 | 80.0 | 74.6 |
| HumanEval | 82.6 | 67.8 | 86.2 | 78.9 | 80.4 | 90.6 |
| HumanEval+ | 82.8 | 69.2 | 82.0 | 77.9 | 78.4 | 88.0 |
| MMLU-Pro | 70.4 | 51.3 | 63.4 | 64.4 | 69.6 | 73.0 |
| DROP | 75.5 | 68.3 | 79.3 | 90.2 | 76.7 | 80.9 |
| ArenaHard | 75.4 | 45.8 | 76.2 | 65.5 | 78.4 | 75.6 |
| LiveBench | 47.6 | 28.1 | 48.1 | 57.6 | 55.3 | 57.6 |
| IFEval | 63.0 | 57.9 | 80.0 | 89.3 | 85.0 | 84.8 |

- **GPQA 56.1** -- превосходит GPT-4o (50.6) при 14B параметрах. Graduate-level STEM.
- **MATH 80.4** -- превосходит teacher GPT-4o (74.6). Доказательство, что synthetic data != чистый distillation.
- **AMC 10/12**: 91.8 -- fresh benchmark, нет contamination risk.
- **SimpleQA 3.0** -- слабое место: factual knowledge (GPT-4o = 39.4). Expected для synthetic-heavy training.
- **IFEval 63.0** -- instruction following ниже, чем у larger models (Llama-3.3: 89.3).

### Long Context (HELMET)

| Task | phi-4 (16K) | Qwen-2.5-14B | Llama-3.3-70B | GPT-4o |
|---|---|---|---|---|
| Recall | 99.0 | 100.0 | 92.0 | 100.0 |
| ICL | 77.0 | 67.6 | 70.0 | 85.6 |
| QA | 36.0 | 29.7 | 36.7 | 43.7 |

## My Notes

- **"Surpassing the teacher"** -- phi-4 превосходит GPT-4o на GPQA и MATH. Это сильный сигнал: synthetic data при правильной генерации -- не просто knowledge transfer, а **curriculum**, структурирующий reasoning paths для learner. Teacher генерирует step-by-step, learner учится следовать цепочке лучше, чем teacher сам.
- **Pivotal Token Search** -- элегантная идея для DPO. Вместо целых response pairs, фокус на конкретных "pivotal" точках. Это как микро-хирургия vs замена органа. Paper не даёт полных деталей алгоритма, но потенциал значительный.
- **Synthetic data scaling curve** -- 13.8 эпох на 290B tokens без degradation это контринтуитивно. Обычно >4 эпох ведёт к overfitting. Hypothesis: diversity synthetic генерации + seed curation дают effectively разные примеры каждую эпоху.
- **SimpleQA 3.0** -- критическая слабость. Модель плохо запоминает факты, потому что web data = 15% mixture. Для practical applications (chatbot, Q&A) это проблема. Phi-4 -- это reasoning engine, не knowledge base.
- **IFEval 63.0** тоже тревожный: 14B модель значительно хуже в instruction following чем 70B+ models. Видимо, post-training оптимизирован для reasoning, не для compliance.
- Approach Microsoft с Phi-family -- прямая альтернатива scaling paradigm. Вместо "bigger model + more tokens" -- "better data + smart curriculum". При 14B params конкурирует с 70B models на reasoning. Но trade-off: слабее на knowledge и instruction following.
- **Decontamination effort** похвален -- AMC 10/12 как fresh benchmark и GPQA (original questions) дают confidence в результатах.
- **Web rewrites** как bridge между synthetic и organic data -- rewriting web content в стиле LLM interactions делает факты accessible во время inference. Это прямо адресует distribution mismatch между web training data и chat inference context.
- **Data mixture ablation** (Table 4) -- strong insight: uniform allocation suboptimal (-2.2 avg), synthetic-heavy лучше, но TQA требует web data (+6.9). Final mixture -- compromise для balanced capabilities. End-to-end optimization pretraining+post-training -- acknowledged future work.
- **Midtraining** для context extension (4K -> 16K): naturally long documents > artificially padded sequences. Upweight >16K samples + new synthetic >4K datasets. RoPE base 250K. 250B tokens при 10x lower LR.
- Phi-4 как demonstration of **"data-centric AI"** paradigm: architecture changes minimal (same as phi-3-medium), но data engineering (50 types of synthetic datasets, seed curation, plurality voting, instruction reversal) -- основной driver improvement.
- Comparison с long-chain-of-thought models (O1, QwQ) -- phi-4 uses 4x fewer tokens и 2x fewer params vs QwQ при comparable AMC scores. Different compute-accuracy tradeoff: phi-4 optimizes для latency, not just accuracy.
- **Organic vs synthetic questions**: organic questions substantially more effective чем synthetic для seeding. Synthetic augmentation (rewriting) helps but gains less pronounced. Это important nuance -- seed quality matters more than seed quantity.
- **Phi family trajectory**: Phi-1 (code) -> Phi-2 (textbooks) -> Phi-3 (distillation) -> Phi-4 (beyond distillation). Each generation pushes frontier of what small models can achieve, с progressive shift от pure distillation к independent capability generation.
- **13.8 epochs на synthetic data** без overfitting -- possible explanation: each epoch effectively sees different examples due to diversity of 50 synthetic dataset types и multiple prompting strategies. This is fundamentally different from repeating web data.
- **Post-training closes gap** между data mixtures: synthetic-heavy и balanced mixtures converge после SFT+DPO. This suggests post-training is a strong equalizer -- good pretraining data gives head start, но alignment can compensate.
- **Code validation** через execution loops -- synthetic code данные проверяются через actual execution. Для scientific datasets -- extracted questions проверяются на groundedness и difficulty balance. Quality assurance pipeline, не just generation.
- **SFT dataset**: 8B tokens -- значительно больше чем типичные SFT datasets (Llama 3: quality>quantity, Nemotron: 200K general). Covers 40 languages, что unusual для 14B model.
- **DPO data**: ~1.1M total pairs across two rounds. First round (PTS): ~250K pairs focused на accuracy. Second round (judge-guided): ~850K pairs broader scope. Progressive refinement approach.
- Paper released Dec 2024, coinciding с inference-time scaling trend (O1, R1). Phi-4 shows complementary approach: better pretraining data vs longer inference chains. Both valid, different cost profiles.
- **MGSM 80.6** -- strong multilingual math reasoning для 14B model (GPT-4o-mini: 86.5, Llama-3.3-70B: 89.1). Multilingual data составляет small fraction, но synthetic data implicitly teaches reasoning patterns transferable across languages.
- **DROP 75.5** -- discrete reasoning over paragraphs. Competitive с GPT-4o-mini (79.3) но значительно ниже Llama-3.3-70B (90.2). Suggests reading comprehension benefits from scale more than synthetic data.
- **PhiBench (internal)**: 56.2 vs GPT-4o 72.4. Internal benchmark composed primarily of original prompts -- provides contamination-free signal, но not reproducible by community.
- **Architecture unchanged from phi-3-medium** -- strongest evidence что data engineering, not architecture, drives Phi-4's improvements. Same 14B params, same structure, dramatically better results. This is the "data-centric AI" thesis in action.
