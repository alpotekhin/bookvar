---
title: "The Llama 3 Herd of Models"
url: https://arxiv.org/abs/2407.21783
authors: "Llama Team, AI @ Meta"
year: 2024
date_reviewed: 2026-04-08
type: source-note
status: legacy
category: paper
raw: "[[02 Areas/ML & DL/raw/papers/llama-3/source]]"
concepts: [Transformer, Scaling Laws, Grouped Query Attention, DPO, Supervised Fine-Tuning, Multimodal, Safety, Data Curation]
---
# The Llama 3 Herd of Models

## TL;DR

Llama 3 -- семейство dense Transformer моделей (8B, 70B, 405B) от Meta, обученных на **15.6T токенов** с контекстом 128K. Flagship 405B модель сопоставима с GPT-4 по широкому спектру задач. Три ключевых принципа: **data quality, scale, managing complexity**. Осознанный выбор в пользу **стандартного dense Transformer** вместо MoE и **DPO вместо PPO** для максимальной стабильности при масштабировании. ~3.8x10^25 FLOPs на pretraining -- ~50x больше Llama 2. Также представлены экспериментальные multimodal extensions (image, video, speech), но не релизнуты.

## Problem

Предыдущие версии Llama уступали closed-source моделям (GPT-4, Claude). Необходимо было: (1) масштабировать данные с 1.8T до 15T+ токенов без деградации качества, (2) разработать scaling laws для предсказания downstream performance, (3) улучшить post-training pipeline для instruction following и safety, (4) добавить tool use, multilinguality, long context -- при сохранении training stability на ~16K GPUs.

## Method

### Архитектура
- **Dense Transformer** -- осознанный выбор в пользу простоты вместо MoE для максимизации training stability. Минимальные изменения от Llama 2:
  - **Grouped Query Attention (GQA)** с 8 KV heads для всех размеров -- ускорение inference + меньше KV cache
  - **128K vocabulary** (100K tiktoken + 28K multilingual) -- compression rate 3.17 -> 3.94 chars/token
  - **RoPE** с base frequency 500,000 для long context support
  - **Document attention mask** -- no cross-document attention in same sequence. Limited impact в standard pretraining, но критично для long-context continued pretraining
  - **SwiGLU** activation function

### Model Shapes

| | 8B | 70B | 405B |
|---|---|---|---|
| Layers | 32 | 80 | 126 |
| Model Dim | 4,096 | 8,192 | 16,384 |
| FFN Dim | 14,336 | 28,672 | 53,248 |
| Attention Heads | 32 | 64 | 128 |
| KV Heads | 8 | 8 | 8 |
| Peak LR | 3e-4 | 1.5e-4 | 8e-5 |

### Pre-training Data (15.6T tokens)
- **Data mix**: ~50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual.
- **Web data curation pipeline**:
  - Custom HTML parser (не trafilatura, не third-party) -- optimized для precision in boilerplate removal + content recall. Markdown **вредит** performance, удаляется.
  - PII + safety filtering: domain blocklists, unsafe content removal.
  - **Three-level dedup**: URL (keep most recent) -> document-level MinHash -> aggressive line-level (>6 occurrences per 30M docs).
  - Heuristic filtering: duplicated n-gram coverage, dirty word counting, KL divergence token distribution.
  - **Model-based quality filtering**: DistilRoberta trained on Llama 2 judgments (Llama 2 как annotator quality -> distill в fast classifier).
- **Code + Math pipelines**: separate DistilRoberta classifiers, prompt-tuned для web pages с math deduction и code interleaved с NL. Domain-specific HTML extraction.
- **Multilingual**: fasttext language ID (176 languages), per-language dedup и filtering, multilingual Llama 2-based quality ranker.
- **Data mix selection**: knowledge classifier для downsampling over-represented categories (arts/entertainment) + scaling law experiments на small models для предсказания large model performance.

### Annealing
- Upsampling high-quality code + math data в конце тренинга.
- **На 8B**: GSM8K +24%, MATH +6.4%.
- **На 405B**: negligible improvement -- flagship model уже имеет strong in-context learning.
- Annealing как tool для оценки quality новых datasets: 50% trained 8B, linear LR decay на 40B tokens, 30% new data + 70% default mix.

### Scaling Laws
- **Two-stage methodology** для предсказания downstream performance:
  1. Linear correlation: NLL на benchmark vs training FLOPs (scaling law models до 10^22 FLOPs).
  2. Sigmoidal relation: NLL vs task accuracy, калиброванная через Llama 2 family.
- Extrapolation на 4 порядка величин для предсказания 405B performance -- slightly underestimates actual performance.
- IsoFLOPs curves становятся **flatter** при больших compute budgets -> 405B robust к small changes в model size / token tradeoff.
- Scaling law prediction: 402B params на 16.55T tokens для 3.8x10^25 FLOPs.

### Infrastructure
- **16,384 H100 80GB GPUs** на Meta's production clusters.
- 4D parallelism: TP (tensor) + PP (pipeline) + CP (context) + DP (data, FSDP).
- **405B**: TP=8, PP=16, CP=1 (pre-train) / CP=16 (context extension), DP=~128.
- Batch size: ~16M tokens -> rampup.
- **GPU utilization**: 38-43% MFU при BF16.
- Context parallelism: Ring Attention-like для sequence splitting, overlap comm с compute.
- Reliability: automated failure detection, checkpointing каждые ~1000 steps, ~93% effective training time.

### Post-training
- **Multiple rounds SFT + DPO** (не PPO/GRPO -- explicit choice для simplicity и stability).
- **SFT data**: human annotations + synthetic data, **quality > quantity** (реально помогает, уменьшение dataset при повышении quality даёт лучший model).
- **Rejection Sampling (RS)**: для каждого prompt генерируется K responses, выбирается лучший через reward model. Используется для SFT data augmentation.
- **DPO**: preference pairs из RS outputs + human annotations. Simpler than PPO, easier to scale.
- **Tool use**: zero-shot tool calling through system prompt + training data. Search engine, code interpreter, mathematical computation tools. Single-turn и multi-turn tool use.
- **Safety**: Llama Guard 3 (input/output classifier), system prompt based safety. Borderline prompts handled through nuanced refusal policies.
- Smaller models benefit from **distillation** из 405B: synthetic data generated by 405B used в SFT для 8B/70B.

### Multimodal (experimental, not released)
- **Image encoder**: ViT-H/14, pre-trained на image-text pairs (600M images), cross-entropy + CLIP loss.
- **Vision adapter**: cross-attention layers injected в language model. Image encoder updated, LM frozen.
- **Video adapter**: trained on top of image adapter на paired video-text data. Temporal aggregation across frames.
- **Speech encoder**: self-supervised (masked prediction), speech adapter converts encodings в LM token space.

## Key Results

### Flagship 405B (Instruct) vs Competitors

| Benchmark | Llama 3 405B | GPT-4 (0125) | Claude 3.5 Sonnet | GPT-4o |
|---|---|---|---|---|
| MMLU (5-shot) | 87.3 | 85.1 | 89.9 | 89.1 |
| MMLU-Pro (5-shot CoT) | 73.3 | 64.8 | 77.0 | 74.0 |
| MATH (0-shot CoT) | 73.8 | 64.5 | 71.1 | 76.6 |
| HumanEval (0-shot) | 89.0 | 86.6 | 92.0 | 90.2 |
| GPQA (0-shot CoT) | 51.1 | 41.4 | 59.4 | 53.6 |
| IFEval | 88.6 | 84.3 | 88.0 | 85.6 |
| GSM8K (8-shot CoT) | 96.8 | 94.2 | 96.4 | 96.1 |
| ARC Challenge (0-shot) | 96.9 | 96.4 | 96.7 | 96.7 |

### Per-Size Performance
- **8B**: MMLU 69.4, HumanEval 72.6, MATH 51.9 -- **best-in-class** среди ~8B models (vs Gemma 2 9B: 72.3 MMLU, Mistral 7B: 61.1)
- **70B**: MMLU 83.6, HumanEval 80.5, MATH 68.0 -- outperforms Mixtral 8x22B (76.9 MMLU) и Nemotron-4 340B (82.6 MMLU) при 4-5x fewer params
- **405B**: сопоставима с GPT-4 across the board, ближе к Claude 3.5 Sonnet

### Long Context
- NIH/Multi-needle: **98.1%** (vs GPT-4: 100%, GPT-4o: 100%)
- ZeroSCROLLS/QuALITY: **95.2%** (vs GPT-4: 95.2%)
- InfiniteBench/En.MC: **83.4%**

### Multilingual
- MGSM (0-shot CoT): **91.6** для 405B (= GPT-4o)

## My Notes

- Meta сделала ставку на **simplicity at scale** -- dense Transformer без MoE, DPO вместо PPO. Это прямо противоположно подходу DeepSeek (MoE + GRPO). Оба работают, но у DeepSeek лучше cost-efficiency (671B total, 37B active за $5.5M vs 405B dense за гораздо больший бюджет).
- **15.6T tokens** -- это огромный dataset, и основной вклад в quality идёт именно от data, не от архитектуры. Data mix (50% general, 25% math) -- сильно отличается от "просто web crawl" и от Falcon's 84% web approach. Dedicated math/code pipelines с domain-specific classifiers -- key differentiator.
- **Scaling laws для downstream task prediction** -- полезная методология. Двухступенчатый подход (NLL -> accuracy) через Llama 2 family для калибровки. Extrapolation на 4 порядка -- impressive и practical для planning.
- **Document attention mask** -- subtle но важно. Prevents cross-document attention, что критично при packing multiple documents в один sequence и при long-context extension. Falcon использовал аналогичный подход для conversation trees.
- Выбор **GQA с 8 KV heads для всех размеров** (8B, 70B, 405B) -- aggressive, но оправдано для inference efficiency. 405B с 128 query heads и 8 KV heads = ratio 16:1.
- **Rejection Sampling** для SFT data -- elegant. Генерируется K responses, лучший (по reward model) используется для SFT. Это hybrid между SFT и RLHF -- проще PPO, но incorporates reward signal.
- **405B как teacher для 8B/70B** -- practical application of knowledge distillation. Synthetic data от flagship model используется для SFT smaller models, что значительно улучшает их quality. Это self-reinforcing: лучший flagship -> лучшие small models.
- **Multimodal через compositional approach** (frozen LM + adapters) -- practical, но уступает native multimodal (Gemini). Модели не релизнуты, что говорит о недоработке. Cross-attention layers vs linear projections -- more expensive но лучше quality.
- **Annealing as evaluation tool** -- underappreciated contribution. Можно оценить value нового dataset за один short training run вместо full scaling law experiments.
- **Open release 405B** -- значимо для экосистемы. Это первая по-настоящему frontier open model, enabling downstream fine-tuning, distillation, research на capabilities frontier-level model.
- Markdown removal from training data -- интересный finding. Counter-intuitive для models часто fine-tuned для markdown output, но при web-heavy pretraining markdown formatting скорее noise.
- **93% effective training time** на 16K GPUs -- impressive engineering. Hardware failures и communication issues при таком масштабе обычно съедают 20-30% времени.
- **Vocabulary 128K** -- larger чем большинство models (Falcon: 65K, Llama 2: 32K). 28K дополнительных multilingual tokens при zero impact на English tokenization -- demonstrates that vocabulary expansion можно делать без trade-offs.
- **Code + Math pipelines** с DistilRoberta classifiers, prompt-tuned для domain-specific content -- это key difference от Falcon's approach (84% undifferentiated web). Llama 3 actively curates 25% math/reasoning и 17% code, что объясняет сильные MATH/HumanEval scores.
- **KL divergence token distribution filter** -- novel heuristic для removing documents с outlier token distributions. Catches things other filters miss: machine-generated spam, corrupted text.
- **8B model trained far past compute-optimal** -- explicit trade-off: worse compute efficiency at training time, but better inference efficiency. Small models see disproportionate benefit from longer training.
- **DPO vs PPO choice** -- pragmatic. Meta explicitly acknowledges PPO is harder to scale и less stable. DPO с rejection sampling achieves comparable results с dramatically simpler pipeline. This is vindicated by strong IFEval (88.6) и human preference results.
- **4D parallelism** (TP+PP+CP+DP) -- most comprehensive parallelism strategy среди reviewed papers. Context Parallelism (Ring Attention-like) -- novel addition для 128K context training. Overlap comm с compute for efficiency.
- **Data mix determination** через scaling law experiments -- iterative process: train small models on candidate mix -> predict large model performance -> repeat. Combined с knowledge classifier для category balancing. More principled чем manual mix selection (Falcon, InternLM2).
- **Annealing findings**: 8B benefits significantly (+24% GSM8K), 405B barely changes -- suggests larger models already learn from their broader pretraining и don't need domain-specific annealing. This has implications для training budget allocation.
- **3.8x10^25 FLOPs** -- ~50x more than Llama 2. For comparison: Falcon-180B ~43K PF-days, Llama 3 405B >>100K PF-days. Scale difference explains much of the performance gap between open models of this generation.
- **Multimodal compositional approach** -- frozen LM + adapters is pragmatic: allows multimodal experiments без risking language model quality. Но ceiling is lower than native multimodal (Gemini). Models not released -- honest acknowledgment of incomplete work.
- **ViT-H/14 image encoder** с 600M image-text pairs + cross-entropy + CLIP loss. Cross-attention layers в LM (not linear projection as in LLaVA). More expensive но higher quality alignment between modalities.
- **Document attention mask** -- subtle engineering decision. При packing multiple documents в один training sequence, prevents attention between unrelated documents. Critical для avoiding spurious correlations в continued pretraining at 128K context.
- **Rejection sampling** as SFT augmentation: generate K responses per prompt, select best via reward model. Hybrid SFT/RLHF approach -- simpler than PPO, incorporates reward signal, scales с compute at generation time.
- Llama 3 как **ecosystem enabler**: open 405B model enables distillation, fine-tuning, research на frontier-level capabilities. Community adoption (Llama Guard 3, tool use protocols) creates network effects beyond raw model quality.
- **IsoFLOPs curves flattening** at high compute -- important practical implication: at Meta's scale, model size vs token count tradeoff is forgiving. 402B vs 450B would give nearly identical results. This reduces risk of suboptimal architecture decisions.
- **Heuristic filtering details**: duplicated n-gram coverage ratio для long repeated content (logging/errors), dirty word counting для adult content, KL divergence для outlier token distributions. Each addresses specific failure mode other filters miss.
- **Line-level dedup** aggressive: removes lines appearing >6 times per 30M docs. Removes navigation menus и boilerplate, но also some high-quality repeated text. Empirical evaluation showed net positive despite over-filtering -- "better to remove too much than too little".
