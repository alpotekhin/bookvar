---
title: "InternLM2 Technical Report"
url: https://arxiv.org/abs/2403.17297
authors: "Zheng Cai et al., Shanghai AI Laboratory"
year: 2024
date_reviewed: 2026-04-08
type: paper-review
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/internlm-2/source]]"
concepts: [RLHF, Conditional Reward Model, Long Context, Grouped Query Attention, Data Curation, PPO, Tool Use]
---
# InternLM2 Technical Report

## TL;DR

InternLM2 -- семейство open-source LLM (1.8B, 7B, 20B) от Shanghai AI Lab, обученных на **2.0-2.6T токенов** с поддержкой **200K context window**. Ключевые инновации: (1) **COOL RLHF** -- Conditional OnLine RLHF с единой reward моделью, использующей system prompts для reconciliation разных preferences, (2) three-phase pretraining (4K -> 32K -> capability enhancement), (3) подробная документация data pipeline для text, code и long context. InternLM2-20B достигает MMLU 67.7 и strong results на Chinese benchmarks, outperforming Llama2-13B и конкурируя с Mixtral-8x7B.

## Problem

(1) Open-source модели значительно уступали closed-source (GPT-4, Claude) по широкому спектру задач. (2) Расширение context window остаётся сложной задачей -- требует данных и инфраструктуры. (3) RLHF страдает от **preference conflicts** (helpful vs harmless) и **reward hacking** при масштабировании. (4) Technical reports LLM редко детализируют data preprocessing, что мешает воспроизводимости.

## Method

### Инфраструктура: InternEvo
- Custom pretraining framework с комбинацией data, tensor, sequence и pipeline parallelism.
- **MFU 53%** на 1024 GPUs для 7B модели (vs 36% DeepSpeed).
- **88% MFU** при training с 256K sequence length на 128 GPUs.
- Adaptive sharding: Full-Replica / Full-Sharding / Partial-Sharding для каждого компонента model state.
- Communication-computation overlap: AllGather для параметров overlap с forward computation.
- Fault tolerance: async checkpointing, automatic failure diagnosis, decoupled eval scheduling.

### Архитектура
- **LLaMA-style Transformer**: RMSNorm, SwiGLU activation, GQA.
- Interleaved Wk/Wq/Wv layout -- упрощает изменение TP degree через split/concat по last dim (5% training acceleration от merged QKV).
- GQA: 8 KV heads для всех размеров (1.8B: 2 queries/KV, 7B: 4, 20B: 6).
- GPT-4 tiktoken tokenizer, адаптированный: 60K cl100k + 32K Chinese tokens = ~92K vocabulary.

### Data Pipeline (исключительно подробный)

#### Text Data
- Sources: web pages (86.46%), books, patents, technical literature.
- Pipeline: Formatting -> Rule-based filtering -> LSH dedup (MinHash, 128 hash functions, 5-gram, threshold 0.7) -> Safety filtering (domain blocking 13M unsafe domains, word blocking 36K terms, BERT toxicity + pornography classifiers) -> Quality filtering (BERT classifiers для advertisements и fluency).
- **Iterative refinement**: ручная annotation + model predictions + guidelines update, 3 итерации.

#### Code Data
- GitHub crawl + public datasets + Q&A forums + tutorials + API docs.
- Three-tier quality: High (16.8%, multi-epoch), Moderate (69.9%, 1 epoch), Low (13.3%, dropped).
- **Dependency sorting**: regexp для import relations -> topological sort -> files concatenated в single long markdown. Позволяет модели учить cross-file dependencies.
- Unified markdown format для interleaving code + natural language.

#### Long Context Data
- Subset of pretraining corpus >32K bytes.
- **Three filters**: (1) Length selection, (2) Statistical filters (conjunction words, linguistic features), (3) Perplexity filters -- conditional probability P(S2|S1) vs P(S2) для detection distracting context.
- Thresholds tailored per domain (code vs textbooks vs patents).

### Pre-training Hyper-parameters

| Params | Layers | Dim | KV Heads | Queries/KV | Learning Rate | Batch Size |
|---|---|---|---|---|---|---|
| 1.8B | 24 | 2048 | 8 | 2 | 3e-4 | 4M |
| 7B | 32 | 4096 | 8 | 4 | 3e-4 | 4M |
| 20B | 48 | 6144 | 8 | 6 | 3e-4 | 5M |

- AdamW: beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.1
- Cosine LR decay до 10% от max
- Total tokens: 2.0T-2.6T в зависимости от model size

### Three-Phase Pretraining
1. **4K Context Training** (~90% steps): standard pretraining, 4096 token truncation. Mixed English + Chinese + code.
2. **Long Context Training** (~9% steps): 50% 32K data + 50% <4K data. RoPE base 50K -> 1M. Training speed -40% от 4K (благодаря InternEvo + FlashAttention scalability).
3. **Capability Enhancement** (~1% steps): 24B tokens of curated data:
   - 65% Retrieved STEM Data (15.97B tokens)
   - 26% Selected High Quality Data (6.26B tokens)
   - 8% Retrieved Special Domain Data (2.17B tokens)
   - Smaller LR и batch size для gentle transition. Значительный uplift на coding, reasoning, exams.

### COOL RLHF

#### Conditional Reward Model
- **Single reward model** с разными **system prompts** для разных preferences (helpful, harmless, code, math). Вместо отдельных reward models как в Llama 2.
- Инициализация из SFT weights, output -> 1D linear projection.
- **2.4M preference pairs** across multiple domains.
- **Focal ranking loss** -- difficulty decay coefficient, снижает вклад easy samples: L = -(1 - 2*max(0, P-0.5))^gamma * log(P). Gamma = 2.
- **Log-barrier penalty** -- constricts score distribution в [-5, 5], стабилизирует PPO hyper-parameters.

#### Online RLHF (3 rounds)
- **Fast Path**: detection и patching reward hacking. 20-100 preference pairs per hacking pattern, формируемые из early vs late-stage PPO responses. Быстрый fix, не требует human annotation.
- **Slow Path**: general improvement через human annotation новых response pairs от latest models. More nuanced, но slow из-за annotation cost.
- Implementation: 3 rounds, thousands of patches per round.

#### PPO Details
- 4 models (actor, critic, reference, reward), all same size.
- Critic initialized from reward model (не SFT) -- higher initial loss, но consistently lower loss и higher rewards после ~20 iterations.
- 200K queries, ~400 iterations, best checkpoint on validation.

### Tool Use
- Modified ChatML: "environment" role + keywords <|interpreter|> и <|plugin|>.
- Code interpreter как special tool -- RICO (Reasoning Interleaved with Coding) strategy.
- Agent-FLAN aligned training для agent capabilities.

## Key Results

### Comprehensive Examination (Base Models)

| Model | MMLU (5-shot) | CMMLU | C-Eval | AGIEval | GAOKAO |
|---|---|---|---|---|---|
| InternLM2-7B | **65.8** | **66.3** | **65.8** | **49.9** | **58.6** |
| Mistral-7B-v0.1 | 64.0 | 44.6 | 47.5 | 32.9 | 28.7 |
| Qwen-7B | 59.7 | 62.5 | 63.1 | 45.6 | 52.8 |
| ChatGLM3-6B | 62.7 | 66.5 | 67.2 | 47.5 | 59.4 |

| Model | MMLU | CMMLU | C-Eval | AGIEval | GAOKAO |
|---|---|---|---|---|---|
| InternLM2-20B | 67.7 | **68.7** | **68.5** | **53.0** | 57.1 |
| Mixtral-8x7B | **71.8** | 53.3 | 55.4 | 40.9 | 32.3 |
| Qwen-14B | 67.9 | 70.1 | 71.8 | 52.0 | **62.5** |

### Chat Models

| Model | MMLU | CMMLU | C-Eval | AGIEval | GAOKAO |
|---|---|---|---|---|---|
| InternLM2-Chat-7B | **63.7** | **63.0** | **60.8** | 47.2 | **58.0** |
| GPT-3.5 | 69.1 | 53.9 | 52.5 | 39.9 | 51.1 |
| Qwen-7B-Chat | 57.1 | 57.9 | 59.8 | 39.7 | 62.1 |

- InternLM2-Chat-7B **превосходит GPT-3.5** на Chinese benchmarks (CMMLU 63.0 vs 53.9, C-Eval 60.8 vs 52.5).
- 200K Needle-in-a-Haystack: near-perfect retrieval.
- COOL RLHF имеет minimal impact на exam performance, но значительный на subjective evaluations.

## My Notes

- **COOL RLHF** -- элегантное решение для preference conflicts. Вместо N reward models (как Llama 2), одна модель с conditional system prompts. Это: (1) меньше compute при PPO, (2) reward model видит все типы preferences при training -> лучшая generalization, (3) проще pipeline.
- **Focal ranking loss** для reward model -- хорошая адаптация Focal Loss из object detection. Easy preference pairs доминируют в training data; без decay model переобучается на тривиальные случаи.
- **Three-phase pretraining** -- practical и reproducible approach. Phase 3 (capability enhancement на 24B tokens) даёт massive uplift: AGIEval 38.7 -> 49.9 для 7B model. Это "annealing on quality data" approach, аналогичный Llama 3.
- **Data documentation** -- одна из наиболее подробных в LLM литературе. Описание code dependency sorting, perplexity-based filtering для long context, iterative annotation -- всё reproducible. Это ценный reference для data engineering.
- **Chinese-English bilingual focus** -- InternLM2 особенно силен на Chinese benchmarks, что логично при 19% Chinese web data. На English MMLU (65.8/67.7) уступает Mixtral (71.8), что ожидаемо.
- **Critic from reward model** -- интересный practical finding. Counter-intuitive: different tasks (reward estimation vs value estimation), но shared knowledge о preference landscape ускоряет convergence.
- **Fast Path reward hacking fix** -- 20-100 examples достаточно для patching конкретного hacking pattern. Это makes online RLHF practical: не нужно ждать massive human annotation, достаточно targeted patches.
- InternLM2 как framework paper: больше про process и methodology, чем про raw benchmarks. 20B model не frontier по absolute numbers, но documentation и tooling делают его ценным для research community.
- Limitation: no MoE, no multi-token prediction, no FP8 -- architecture conservative. Frontier performance gap остаётся.
- **Perplexity-based filtering** для long context data -- elegant: если P(S2|S1) > P(S2), то S1 -- distracting context. Bias-resistant, т.к. использует разницу perplexity, не абсолютное значение. Catches failed HTML parsing, random social media snippets, recognition errors.
- **Code dependency sorting** через topological sort import relations -- позволяет модели учить cross-file dependencies в рамках 32K context. Regexp-based import detection с heuristics для batched imports. Practical engineering contribution.
- **Interleaved QKV layout** -- minor но impactful optimization. Вместо stacked [Wq, Wk, Wv], interleaving per head позволяет менять TP size через simple split/concat по last dim. 5% training acceleration от merged QKV computation.
- **Release strategy** -- models released at multiple stages (Base, enhanced Base, SFT, Chat) позволяет community анализировать impact каждого этапа. Это transparency best practice, которому мало кто следует.
- **Safety filtering** -- comprehensive multi-layer approach: 13M domain blocklist, 36K word blocklist, BERT toxicity classifier (Kaggle dataset), BERT pornography classifier (Perspective API annotations). Multiple layers catch what single filters miss.
- **SFT dataset**: 10M instruction data instances -- значительно больше, чем у большинства models (Llama 3: quality > quantity, Nemotron: 200K general SFT). InternLM2 делает ставку на coverage и diversity.
- **MFU comparison**: InternEvo 53% на 1024 GPUs vs DeepSpeed 36% -- 47% improvement. Для long sequences (256K tokens): InternEvo 88% vs DeepSpeed-Ulysses/Megatron-LM 65%. Серьёзное infrastructure contribution.
- **GQA vs MHA trade-off**: 8 KV heads для всех sizes. Ratio query/KV: 2:1 (1.8B), 4:1 (7B), 6:1 (20B). Conservative по сравнению с Llama 3 (16:1 для 405B), что даёт более высокое quality per KV head но больший memory footprint.
- **LSH dedup parameters**: 128 hash functions, 5-gram, threshold 0.7 -- более aggressive чем стандартные settings. Priority на keeping most recent data (larger CC dump numbers) -- temporal freshness over historical coverage.
- **Code quality annotation**: iterative 3-round process где annotators verify model predictions и refine guidelines. Automatic validation ensures previously annotated samples correctly classified. Practical methodology для любого ML quality annotation task.
- **200K Needle-in-a-Haystack** -- near-perfect retrieval через positional encoding extrapolation (RoPE base 50K -> 1M). Remarkable given that pretraining max context = 32K. Suggests RoPE extrapolation works well для simple retrieval, may degrade на complex reasoning at extreme lengths.
- **FLORES translation**: InternLM2-Chat-7B achieves 15.5 BLEU (8-shot) vs GPT-3.5's... note that GPT-3.5 scores shown are on exam benchmarks, not translation. InternLM2's bilingual Chinese-English focus gives natural advantage на translation tasks involving these languages.
