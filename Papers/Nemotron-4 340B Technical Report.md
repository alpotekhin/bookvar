---
title: "Nemotron-4 340B Technical Report"
url: https://arxiv.org/abs/2406.11704
authors: "NVIDIA"
year: 2024
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/nemotron-4/source]]"
concepts: [Synthetic Data Generation, Reward Model, RLHF, DPO, Iterative Alignment, Supervised Fine-Tuning, Open Source LLM]
---
# Nemotron-4 340B Technical Report

## TL;DR

Nemotron-4 340B -- семейство из трёх моделей от NVIDIA: Base (340B), Instruct и Reward. Обучена на **9T токенов** (8T pretraining + 1T continued training). Ключевая инновация -- **98% данных alignment процесса синтетические**, сгенерированные через iterative weak-to-strong pipeline с использованием Nemotron-4-340B-Reward для quality filtering. Reward модель достигает #1 на RewardBench (92.0). Открытый synthetic data generation pipeline -- главный вклад для community.

## Problem

(1) Качественные данные для alignment дороги -- human annotation стоит значительных ресурсов. (2) Существующие open datasets недостаточны для тренировки frontier instruct моделей. (3) Нет открытых pipelines для генерации высококачественных synthetic alignment данных. NVIDIA решает все три проблемы, показывая, что **20K human-annotated examples + synthetic pipeline = competitive instruct model**.

## Method

### Pretraining
- **9T tokens**: 70% English NL, 15% multilingual (53 языка), 15% code (43 языка).
- 8T formal pretraining + 1T continued training с upsampled high-quality sources.
- **Architecture**: standard decoder-only Transformer:
  - 96 layers, hidden dim 18432, 96 attention heads, 8 KV heads (GQA)
  - Sequence length 4096, vocabulary 256,000
  - **9.4B embedding params + 331.6B non-embedding params**
  - RoPE, SentencePiece tokenizer, **squared ReLU** (не SwiGLU), no bias, no dropout, **untied embeddings**

### Training Infrastructure
- **768 DGX H100 nodes** (6144 GPUs), H100 80GB SXM5
- Each H100: 989 TFLOP/s peak (BF16 without sparsity)
- Intra-node: NVLink/NVSwitch (900 GB/s bidirectional)
- Inter-node: 8x 400 Gbps HDR InfiniBand HCAs
- 8-way TP, 12-way PP с interleaving, data parallelism 16->64 (batch size ramp)

### Training Efficiency

| DP Size | GPUs | Iter Time (s) | MFU (%) | Batch Size | Tokens (B) |
|---|---|---|---|---|---|
| 16 | 1536 | 10.3 | 42.4 | 768 | 200 |
| 32 | 3072 | 10.3 | 42.3 | 1536 | 200 |
| 64 | 6144 | 8.0 | 41.0 | 2304 | 7600 |

### Continued Training
- Последний 1T tokens: shifted data distribution к high-quality sources.
- **Two distributions**: (1) majority -- reweighted pretraining data favoring higher quality, (2) minority -- QA-style alignment examples + upweighted low-accuracy domains.
- Steeper LR decay slope (priority over magnitude). Gentle transition от pretraining distribution.
- Значительный uplift quality на downstream benchmarks.

### Reward Model -- Nemotron-4-340B-Reward
- **Multi-attribute regression** (не pairwise ranking): предсказывает 5 атрибутов HelpSteer -- Helpfulness, Correctness, Coherence, Complexity, Verbosity.
- Обучена на **10K human-annotated HelpSteer2** preference data.
- Weighted sum атрибутов = overall reward at inference.
- **#1 на RewardBench** (92.0 overall) на момент публикации, выше GPT-4o (84.7) и Gemini 1.5 Pro (88.1).
- Особенно сильна на **Chat-Hard** (87.1 vs GPT-4o: 70.4) -- сложные случаи, где chosen/rejected трудно различить.

### Synthetic Data Generation Pipeline

#### Prompt Generation
- Synthetic prompts через Mixtral-8x7B-Instruct -- 3K topics, diverse tasks (open Q&A, writing, closed Q&A, math/coding).
- Instruction-following prompts с verifiable constraints.
- Two-turn prompts для multi-turn conversation skills.
- LMSYS-Chat-1M prompts для real-world coverage.

#### Response Generation & Quality Filtering
- Responses от multiple intermediate models для diversity.
- **Three judging strategies**:
  1. Ground-Truth-as-Judge -- для задач с объективными ответами (GSM8K, MATH).
  2. LLM-as-Judge -- GPT-4 class model сравнивает pairs, double-evaluation для positional bias.
  3. **Reward-Model-as-Judge** -- Nemotron-4-340B-Reward. Accuracy 87% на Chat-Hard vs 54% для LLM-as-Judge. Заменил LLM-as-Judge в поздних итерациях.

#### Iterative Weak-to-Strong Alignment
- **Iteration 1**: Mixtral-8x7B-Instruct генерирует data -> trains intermediate 340B -> surpasses Mixtral.
- **Iteration 2**: Improved 340B-Instruct генерирует better data -> trains better 340B.
- **Self-reinforcing flywheel**: stronger base model + better data -> even stronger instruct model.
- Teacher **не ставит ceiling** на student -- student consistently surpasses teacher.

### Alignment Algorithms

#### Two-Stage SFT
1. **Code SFT**: 800K samples, generated through Genetic Instruct (evolutionary approach: self-instruction + wizard coder mutations + LLM fitness function). 1 epoch, constant LR 3e-7, batch 128.
2. **General SFT**: 200K samples, diverse tasks (see Prompt Preparation). 3 epochs, batch 128, LR search [1e-7, 5e-7]. Includes 2% code replay для prevention forgetting.
- Loss only on assistant turns (user turns masked).

#### DPO + RPO
- **DPO** с weighted SFT loss на chosen responses + Reward-Model quality filtering. 160K examples, 1 epoch, batch 256. Tuned: LR [3e-8, 3e-7], KL coef [3e-4, 3e-3], SFT weight [1e-5, 1e-3].
- **RPO (Reward-aware Preference Optimization)** -- новый алгоритм:
  - DPO treats all preference pairs equally (binary order). RPO аппроксимирует **magnitude** reward gap.
  - Loss: D(implicit_reward_gap || actual_reward_gap). Prevents overfitting на easy cases и unnecessary "unlearning" high-quality rejected responses.
  - Particularly useful для synthetic data где reward scores available (Nemotron-4-340B-Reward provides continuous scores).

#### Additional Training Data
- **Topic following**: CantTalkAboutThis dataset -- dialogues с distractor turns.
- **Incapable tasks**: few-shot generated rejection responses для tasks requiring internet/real-time knowledge.
- **STEM**: Open-Platypus subsets (PRM800K, SciBench, ARB, OpenBookQA).
- **Document QA**: FinQA, WikiTableQuestions.
- **Function calling**: subset от Glaive AI.

## Key Results

### Base Model

| Benchmark | Nemotron-4-340B | Llama-3-70B | Mixtral-8x22B | Qwen-2-72B |
|---|---|---|---|---|
| ARC-c (25-shot) | **94.28** | 93.00 | 91.30 | 68.90 |
| Winogrande (5-shot) | **89.50** | 85.30 | 84.70 | 85.10 |
| HellaSwag (10-shot) | **90.53** | 88.00 | 88.50 | 87.60 |
| MMLU (5-shot) | 81.10 | 79.50 | 77.75 | **84.20** |
| BBH (3-shot) | **85.44** | 81.30 | 78.90 | 82.40 |
| HumanEval (0-shot) | 57.32 | 48.20 | 45.10 | **64.60** |

### Instruct Model
- **Arena Hard**: превосходит Llama-3-70B-Instruct и Mixtral-8x22B-Instruct.
- **IFEval**: strong instruction following.
- **AlpacaEval 2.0 LC**: competitive с top instruct models.

### Reward Model

| Model | Overall | Chat | Chat-Hard | Safety | Reasoning |
|---|---|---|---|---|---|
| **Nemotron-4-340B-Reward** | **92.0** | 95.8 | **87.1** | 91.5 | 93.7 |
| Cohere May 2024 | 89.5 | 96.4 | 71.3 | 92.7 | 97.7 |
| GPT-4o-0513 | 84.7 | 96.6 | 70.4 | 86.7 | 84.9 |

## My Notes

- **98% synthetic alignment data** -- это proof-of-concept, что human annotation можно радикально сократить. 10K HelpSteer2 + 10K SFT human = 20K total human samples для alignment модели frontier-level. Это меняет economics of alignment.
- **Iterative Weak-to-Strong** -- простая но мощная идея. Mixtral-8x7B (значительно слабее) генерирует data, которая при alignment на stronger base model даёт model, превосходящую teacher. Это подтверждает findings Phi-4 и Burns et al. о weak-to-strong generalization.
- **Reward-Model-as-Judge** значительно лучше LLM-as-Judge (87% vs 54% на Chat-Hard). Это контринтуитивно -- reward model обучена на 10K samples, а LLM видела триллионы. Видимо, specialization на preference task > general capabilities для judging.
- **RPO** -- интересная extension DPO. DPO treats all preference pairs equally (binary), RPO учитывает magnitude разницы. Предотвращает overfitting на easy cases и unnecessary unlearning хороших rejected responses.
- **Squared ReLU** вместо SwiGLU -- unusual choice. Большинство modern LLMs используют SwiGLU. NVIDIA видимо следует собственной Nemotron-4 15B recipe. Performance competitive, но прямых ablations нет.
- **340B dense model** на single DGX H100 в FP8 -- practical deployment target. MoE модели (DeepSeek, Mixtral) эффективнее по active params, но dense проще в inference.
- **Continued training** (1T tokens с shifted distribution) -- стандартная практика, но полезно что NVIDIA подтверждает: upsampling quality sources + QA data в конце тренинга значительно улучшает benchmarks.
- Открытый SDG pipeline -- реально ценно для community. Code для pretraining, alignment, reward model training все опубликованы.
- **Two-stage SFT** (Code -> General) -- practical решение для multi-task conflicts. Code SFT требует 800K samples -- значительно больше, чем General SFT (200K). Sequential training с 2% replay предотвращает catastrophic forgetting.
- **Genetic Instruct** для code data synthesis -- evolutionary approach: self-instruction + wizard coder mutations + fitness function (LLM-based). Efficient parallel execution с multiple colonies. Creative application of evolutionary algorithms к data generation.
- **HelpSteer2 multi-attribute regression** -- 5 dimensions (helpfulness, correctness, coherence, complexity, verbosity) вместо single score. Disentangles "helpful" от "long" -- решает known bias DPO/RLHF к verbose responses.
- **LMSYS prompts harder than synthetic** -- helpfulness distribution shows synthetic prompts avg 3.24 vs LMSYS 3.04. Synthetic prompts tend к simpler questions. Important для balancing training data -- need real-world difficulty.
- **DPO + SFT loss on chosen** -- addresses known DPO failure mode где likelihoods обоих chosen и rejected падают. Weighted SFT loss на chosen prevents policy от shifting too far. Simple fix, significant impact.
- Limitation: 340B dense model -- expensive для inference. При FP8 fits на single DGX H100, но это всё ещё 8x H100 GPUs. MoE architectures (Mixtral, DeepSeek) значительно эффективнее per-token. Trade-off: simplicity + stability vs cost-efficiency.
- **Positional bias mitigation** в LLM-as-Judge: evaluate каждую pair дважды с swapped order, retain only consistent judgments. Simple но effective debiasing.
- **Vocabulary 256K** -- largest среди reviewed models (Llama 3: 128K, Falcon: 65K, InternLM2: 92K). Large vocabulary помогает multilinguality (53 languages) и code (43 languages), но может быть excessive для English-focused use.
- Paper позиционирует Nemotron-4 как **synthetic data generation platform**, не просто как LLM. Base model -> generates data -> trains smaller models. Это paradigm shift: value модели не только в inference, но в data generation capability.
- **Permissive license** (NVIDIA Open Model License) allows commercial use -- important для actual adoption vs research-only licenses. Combined с open SDG pipeline, это enables full reproducibility of alignment process.
- **Prior Sets score** на RewardBench: 67.4 -- значительно ниже Cohere (78.2). Авторы attribute это к not using training data from those datasets. Honest limitation acknowledgment.
- **Synthetic prompt diversity**: multi-dimensional -- task diversity (writing, Q&A, coding), topic diversity (STEM, humanities, daily life), instruction diversity (json output, paragraph count, yes/no). 3K topics total. Systematic approach к coverage.
- Paper implicitly argues for **vertical integration**: base model + reward model + SDG pipeline as unified system. Each component reinforces others через iterative improvement loop.
