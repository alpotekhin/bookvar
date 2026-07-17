---
title: "The Falcon Series of Open Language Models"
url: https://arxiv.org/abs/2311.16867
authors: "Falcon LLM Team, Technology Innovation Institute"
year: 2023
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/falcon/source]]"
concepts: [Transformer, RefinedWeb, Data Curation, Multiquery Attention, Grouped Query Attention, Scaling Laws, Open Source LLM]
---
# The Falcon Series of Open Language Models

## TL;DR

Falcon -- семейство causal decoder-only моделей (7B, 40B, 180B) от TII (Abu Dhabi), обученных **преимущественно на web-данных** через RefinedWeb -- отфильтрованный и дедуплицированный корпус из CommonCrawl на 5T токенов. Falcon-180B обучен на **3.5T токенов** на 4096 A100 40GB в AWS cloud, достигает 99.5% производительности PaLM-2 Large и превосходит GPT-3.5 на большинстве бенчмарков. Ключевой вклад -- демонстрация того, что **web-данные с качественной фильтрацией могут превосходить curated corpora**.

## Problem

На момент работы (2022-2023) open-source LLM значительно уступали closed-source моделям (GPT-4, PaLM-2). Главные вызовы: (1) масштабирование pretraining data без деградации качества, (2) эффективный distributed training на cloud-инфраструктуре с ограниченным interconnect, (3) документирование процесса для воспроизводимости -- большинство frontier моделей публиковали минимум деталей.

## Method

### Данные: RefinedWeb
- **Философия**: web-данные с качественной фильтрацией могут заменить curated corpora. 84% датасета -- RefinedWeb (English + European), 16% -- curated (books, conversations, code, technical).
- **Macrodata Refinement pipeline**: URL filtering -> HTML extraction (trafilatura) -> language ID -> heuristic filtering -> fuzzy dedup (MinHash) -> exact substring dedup (suffix array). В итоге сохраняется ~12% от CommonCrawl.
- **Эксперимент**: 1B/3B модели, обученные на RefinedWeb, **превосходят** модели на The Pile (curated) по zero-shot NLP задачам. Добавление curated данных к сильному web baseline **не улучшает** и даже **вредит** при >50%.
- **Без upsampling**: все 3.5T токенов уникальны, ни один документ не повторяется.

### Архитектура
- **Decoder-only Transformer**, архитектурно близкий к PaLM:
  - Grouped Query Attention (multigroup): nkv = TP (1/8/8 для 7/40/180B). Радикальное сокращение KV-cache: 40MB vs 10GB для vanilla attention на 180B.
  - Rotary Positional Embeddings (RoPE) -- минимальное преимущество над ALiBi при 3B scale, но выбрано для совместимости.
  - **Без GLU/SwiGLU** -- не оправдано на A100 40GB из-за увеличения memory footprint на 33% при отсутствии uplift в zero-shot.
  - Parallel attention + MLP blocks (как GPT-J) -- сокращает all_reduce с 2 до 1 на слой.
  - Без bias в linear layers, z-loss для стабильности.
  - Tied embeddings, GeLU activation, vocabulary 65K.

### Distributed Training: Gigatron
- **3D parallelism**: Tensor (TP) + Pipeline (PP, PipeDream-Flush/1F1B) + Data (DP) + ZeRO-1 optimizer sharding.
- **4096 A100 40GB** на AWS cloud с ограниченным interconnect (50 Gbps/GPU).
- Custom **Triton kernels** для FlashAttention и RoPE -- основной драйвер throughput. FlashAttention eliminates need for activation checkpointing, все FLOPs идут на training.
- **Selective recomputation** через "monolayer" -- single custom autograd function для entire decoder block. Recompute activations + layernorms (save only LN statistics). Снижение activation memory в **2x** без потери throughput.
- **Только bfloat16**, без mixed precision complexity. Stochastic rounding не помогает когда optimizer state в fp32.
- **Scatter-gather оптимизация** для pipeline parallelism -- scatter activations по TP degree перед inter-node send, gather после receive. Сокращение comm volume в 8x.
- **ZeRO-1**: optimizer state sharding. Memory per param: 4 + 16/DP bytes. Falcon-180B: 765GB (vs 3600GB без sharding). **Zero throughput overhead** vs traditional DP.
- **Topology discovery** + Gromov-Wasserstein optimal transport для rank placement в cloud. O(n) measurement steps вместо O(n^2).
- **Topology-agnostic checkpoints** -- checkpoint readable/writable между any topology configurations. Critical для scaling cluster during training.
- **Low-discrepancy data loading** -- deterministic sampling pattern (10K sequences) guaranteeing exact dataset weights, не just в expectation.

### Model Shapes

| | Falcon-7B | Falcon-40B | Falcon-180B |
|---|---|---|---|
| Layers | 32 | 60 | 80 |
| Dim | 4,544 | 8,192 | 14,848 |
| Query heads | 71 | 128 | 232 |
| KV heads | 1 | 8 | 8 |
| Tokens | 1,500B | 1,000B | 3,500B |
| GPUs | 384 | 384 | 4,096 |
| PF-days | 730 | 2,800 | 43,500 |

## Key Results

### Falcon-180B vs PaLM / GPT

| Benchmark | Falcon-180B | PaLM-2 Large | GPT-3.5 | GPT-4 |
|---|---|---|---|---|
| HellaSwag (10-shot) | 89.0 | 86.8 | 85.5 | 95.3 |
| Winogrande (5-shot) | 87.1 | 83.0 | 81.6 | 87.5 |
| ARC Challenge (2-shot) | 87.8 | -- | 85.2 | 96.3 |
| MMLU (5-shot) | 70.6 | -- | 70.0 | 86.5 |
| PIQA | 86.1 | 85.0 | -- | -- |
| OpenBookQA | 64.2 | 58.5 | -- | -- |

- **99.5% performance от PaLM-2 Large** на агрегате из 20 NLP задач
- Систематически выше GPT-3.5, ниже GPT-4
- На commonsense задачах (HellaSwag, Winogrande) -- **лучше PaLM-2 Large**
- Слабое место: MMLU (70.6) -- вероятно из-за 84% web data и мало technical/academic sources

### Comparison with PaLM on NLP aggregate (1-shot)

| Task | PaLM | PaLM-2 S | PaLM-2 M | PaLM-2 L | Falcon-180B |
|---|---|---|---|---|---|
| WebQuestions (EM) | 22.6 | 21.8 | 26.9 | 28.2 | **31.9** |
| HellaSwag | 83.6 | 82.0 | 84.0 | 86.8 | **87.5** |
| LAMBADA | 81.8 | 80.7 | 83.7 | **86.9** | 84.4 |
| Winogrande | 83.7 | 77.9 | 79.2 | 83.0 | **85.1** |
| PIQA | 83.9 | 82.2 | 83.2 | 85.0 | **86.1** |
| OpenBookQA | 53.6 | 57.4 | 56.2 | 58.5 | **64.2** |
| ANLI R1 | 52.6 | 53.1 | 58.1 | **73.1** | 60.5 |
| Task average | 73.1 | 71.6 | 73.4 | **77.5** | 77.1 |
| Fraction of PaLM-2 L | 94.4% | 92.4% | 94.8% | 100% | **99.5%** |

### Commonsense State-of-the-Art

| | PIQA | HellaSwag | Winogrande | BoolQ | LAMBADA |
|---|---|---|---|---|---|
| LLaMA-2 70B | 82.8 | 87.3 | 80.2 | 85.0 | -- |
| Inflection-1 | 84.2 | 85.8 | 83.3 | 89.7 | 78.5 |
| **Falcon-180B** | **84.9** | **89.0** | **87.1** | 87.8 | 79.8 |

### Base Model Comparison (EAI Harness)

| | Falcon-7B | Falcon-40B | Falcon-180B |
|---|---|---|---|
| Aggregate zero-shot | 60.8 | 67.1 | 70.3 |
| Closest model | <GPT-3 | Chinchilla | PaLM-2 Large |

### Code (HumanEval)
- Falcon-180B: best среди NL-focused models, сопоставим с Inflection-1.
- Уступает dedicated code models (Codex, StarCoder, Code LLaMA), что ожидаемо при 3% code в data mix.

### Спайки и стабильность
- 9 спайков за весь тренинг 40B и 180B -- recovery через rollback + skip 1B tokens.
- Z-loss и gradient clipping (0.4 для 180B) -- основные механизмы стабильности.

### Data Ablation Highlights
- RefinedWeb **превосходит** The Pile и другие web datasets (OSCAR, C4) на zero-shot NLP tasks при 1B/3B ablations.
- Curated data (books, technical, conversations) при добавлении к RefinedWeb **не улучшает** zero-shot performance. При >50% -- **вредит** из-за mode collapse.
- 5% code и 10% multilingual data -- **минимальная деградация** на English tasks.
- Weight decay (0.1) особенно важен для low-quality/undeduplicated datasets (The Pile).

## My Notes

- **"Web data is all you need"** -- главный тезис Falcon. Фильтрация + дедупликация CommonCrawl дает корпус, превосходящий curated datasets вроде The Pile. Это радикально упрощает data pipeline и масштабирование.
- Выбор **без SwiGLU** -- контринтуитивный, но обоснованный: на A100 40GB memory overhead от GLU перевешивает минимальный quality gain. Последующие модели (Llama, DeepSeek) всё же используют SwiGLU, но на 80GB картах.
- **Cloud training на 4096 A100 40GB** с 50 Gbps interconnect -- значимый engineering contribution. Большинство аналогичных моделей тренировались на dedicated кластерах с NVLink/InfiniBand everywhere.
- Решение использовать **multigroup attention = TP** -- elegant. Falcon фактически одновременно с Ainslie et al. пришёл к идее GQA, что позже стало стандартом (Llama 2/3, Qwen, etc.).
- Ablations в paper исключительно подробные -- 50+ страниц экспериментов. Это одна из наиболее document-transparent работ в LLM space.
- Слабость: отсутствие post-training (SFT/RLHF) анализа. Paper фокусируется исключительно на pretraining, что ограничивает practical applicability.
- MMLU 70.6 на 180B -- ниже ожиданий. Для сравнения, Llama 2 70B достигает схожих результатов при 2.6x меньшем compute. Data mix, тяжёлый на web, вероятно причина.
- Открытая модель + 600B extract RefinedWeb -- значимо для community. На момент публикации Falcon-180B был крупнейшей полностью открытой моделью.
- **Evaluation methodology** -- одна из самых честных в LLM space. Авторы документируют impact prompt formatting (outlining choices: +10-20% на 180B), unconditional normalization, и стараются match evaluation setup каждого сравниваемого model. Секция 6.1 -- must-read для любого, кто занимается LLM evaluation.
- **Conversation tree masking** -- creative approach к training на Reddit/forum data. Вместо sampling trajectories или repeating data, используется attention mask для encoding всех branch-ов tree одновременно. Depth-first serialization с position = tree depth.
- **Topology discovery + optimal transport** для GPU placement в cloud -- серьёзный systems contribution. AWS не expose topology, поэтому TII делает O(n) bandwidth measurements между нодами и используют Gromov-Wasserstein OT для размещения ranks. Это practical knowledge, редко публикуемый.
- **Numerical precision**: pure bfloat16 training без fp16 mixed precision. Stochastic rounding не нужен когда optimizer state в fp32. FlashAttention в bf16 **более numerically accurate** чем traditional attention в bf16 (vs fp32 ground truth) -- counter-intuitive finding.
- **Learning rate search heuristic**: 4-6 candidate LRs, 500M tokens warmup, pick lowest loss. Simple, effective, reproducible. Результат -- LR значительно выше чем у предшественников (6e-4 для 7B vs 3e-4 для GPT-3 7B). Higher LR = faster convergence, acceptable trade-off vs occasional spikes.
- **No dropout, no sequence parallelism** -- simplicity choices. Models trained for single epoch on unique data, dropout unnecessary. Sequence parallelism не нужен after memory optimizations (FlashAttention + monolayer selective recomputation + ZeRO-1).
- **Cost efficiency focus** -- A100 40GB вместо 80GB, cloud вместо on-prem. Показывает что frontier training возможен без Google/Meta level infrastructure investment. Important democratization signal.
- **Scaling law adherence**: Falcon-180B -- первая publicly documented GPT-3-scale модель, следующая updated scaling laws от Hoffmann et al. (2022). 3.5T tokens на 180B params, без upsampling. Chinchilla-optimal lower bound для training length, upper bound для model size.
- **RefinedWeb 600B release** -- largest open web dataset на момент публикации. Enabling reproducibility и community research на data quality.
- **Context length 2048** -- архаичное по современным стандартам (Llama 3: 128K, InternLM2: 200K). Авторы acknowledge possible a posteriori extension (Chen et al., 2023), но не реализовали.
- Temporal context: paper вышел в Nov 2023, training начат в Dec 2022. За это время landscape радикально сменился -- появились Llama 2, Mistral, GPT-4. Paper remains valuable как engineering reference, но модель устарела.
- **Falcon-7B**: обучен отдельно и позднее 40B/180B. Используется single KV head (true multiquery, не multigroup), увеличенный batch size (2304 vs 1152 для 40B), single layer norm вместо двух отдельных. Некоторые findings из 40B/180B incorporated.
- **No GLU decision** -- interesting counterpoint к industry consensus. Большинство моделей после PaLM (Llama, Mistral, Qwen, DeepSeek) используют SwiGLU. Falcon shows it's not strictly necessary -- GeLU competitive при правильном hyperparameter tuning. Memory savings на A100 40GB -- practical reason.
- **Pipeline parallelism**: PipeDream-Flush (1F1B), не Interleaved-1F1B. Interleaving beneficial only when microbatches < 64; после batch size rampup, standard 1F1B достаточен. Practical guidance для practitioners.
- **Batch size warmup**: 100B tokens для 40B/180B (30B для 7B). Longer warmups never hurt downstream performance. Meta's Llama 3 uses similar long warmup strategy. This allows gradual cluster scaling during warmup.
- **Hardware failures** on 4096 A100: ~11 GPU-years per day. Majority linked to corrupted memory rows -- detected through large matrix multiplication tests on startup + NaN tracking during training. Essential systems engineering for large-scale training.
- **Web data findings validated at scale**: RefinedWeb advantage confirmed не только на 1B/3B ablations, но и на 7B model (350B tokens). Architecture improvements -- minor impact on task performance, primarily improve inference/training scalability.
- Open questions for future: (1) post-training (SFT/RLHF) quality -- not addressed, (2) context extension beyond 2K, (3) code specialization through FIM, (4) multilinguality beyond European languages.
