---
title: ML and DL Map of Content
type: concept
status: legacy
last_updated: "2026-04-15"
---

# ML and DL Knowledge Base

Root map of the ML/DL knowledge base. Updated automatically by the wiki-compiler agent.

> [!important] Новый основной вход
> Открывайте [[02 Areas/ML & DL/00 Учебник/_index|интерактивный учебник по NLP и LLM]].
> Старые `Concepts`, `Papers` и `Courses` сохранены как глубокие заметки и
> источники для постепенной миграции.

## Новая навигация

- [[02 Areas/ML & DL/00 Учебник/_index|Учебник]]
- [[02 Areas/ML & DL/01 Справочник/_index|Справочник]]
- [[02 Areas/ML & DL/02 Атлас моделей/_index|Атлас моделей]]
- [[02 Areas/ML & DL/03 Исследовательские линии/_index|Исследовательские линии]]
- [[02 Areas/ML & DL/04 Вопросы/_index|Вопросы]]
- [[02 Areas/ML & DL/05 Источники/_index|Источники]]
- [[02 Areas/ML & DL/06 Практика/_index|Практика]]

**Stats (as of 2026-04-15):** 71 papers · 120 concept articles (7 categories) · 29 course notes (6 courses)

---

## Concept Categories

### [[02 Areas/ML & DL/Concepts/Architectures/_index|Architectures]] (36 concepts)
Model architectures: Transformers, encoder/decoder variants, LLMs, SSM и альтернативы.

Key concepts: Transformer · BERT · GPT-3 · LLaMA · T5 · Mamba · Mistral 7B · RoPE · GQA · MoE · LSTM · RNN

### [[02 Areas/ML & DL/Concepts/Training/_index|Training]] (32 concepts)
Pre-training objectives, fine-tuning, alignment, PEFT, regularization.

Key concepts: Pre-training · SFT · RLHF · PPO · DPO · RLVR · GRPO · LoRA · Reward Model · Distillation · Alignment · Backpropagation

### [[02 Areas/ML & DL/Concepts/Inference/_index|Inference]] (10 concepts)
Prompting, decoding, inference-time optimization.

Key concepts: In-context Learning · Chain of Thought · Flash Attention · Speculative Decoding · Quantization

### [[02 Areas/ML & DL/Concepts/Retrieval/_index|Retrieval]] (7 concepts)
Retrieval-augmented generation, dense retrieval, knowledge-intensive NLP.

Key concepts: RAG · Self-RAG · ColBERT · REALM · Dense Retrieval

### [[02 Areas/ML & DL/Concepts/NLP/_index|NLP]] (25 concepts)
Transformer internals, tokenization, benchmarks, paradigms.

Key concepts: Attention · Self-Attention · Tokenization · BPE · Word2Vec · GloVe · ELMo · Cross-Attention · Language Model · Seq2Seq · Machine Translation

### [[02 Areas/ML & DL/Concepts/Reasoning/_index|Reasoning]] (6 concepts)
LLM reasoning, planning, tool use.

Key concepts: ReAct · Tree of Thoughts · Self-Consistency · Toolformer · RLVR Tool-Use · Tool Use

### [[02 Areas/ML & DL/Concepts/Evaluation/_index|Evaluation]] (4 concepts)
Benchmarks и метрики для оценки LLM.

Key concepts: MMLU · HumanEval · Perplexity · BLEU Score

Full concept index: [[02 Areas/ML & DL/Concepts/_index|Concepts/_index]]

---

## Papers (71 reviewed)

| Paper | Year | Org | Key Result |
|---|---|---|---|
| Karpathy — Unreasonable Effectiveness of RNNs | 2015 | — | Char-level RNN generates Shakespeare, LaTeX, C |
| Adversarial Training | 2017 | Google | VAT semi-supervised; IMDB 5.91% error |
| Attention Is All You Need | 2017 | Google | Transformer architecture; EN-DE 28.4 BLEU |
| BERT | 2018 | Google | GLUE 80.5; bidirectional MLM + NSP |
| GPT 1.0 | 2018 | OpenAI | Pre-train + fine-tune paradigm; SOTA on 9/12 NLU tasks |
| GPT-2 | 2019 | OpenAI | Zero-shot multi-task; 1.5B; WebText |
| Karpathy — Recipe for Training NNs | 2019 | — | 6-step training methodology |
| RoBERTa | 2019 | Meta | GLUE 88.5; no NSP; dynamic masking |
| ColBERT | 2020 | Stanford | Late interaction; 170x faster; MaxSim |
| GPT-3 | 2020 | OpenAI | 175B; in-context learning; TriviaQA 71.2% |
| RAG | 2020 | Meta | NQ 44.5 EM; retrieval + BART + DPR |
| REALM | 2020 | Google | End-to-end retrieval pre-training; +4-16% QA |
| T5 | 2020 | Google | Text-to-text; 11B; C4; GLUE 90.3 |
| DeBERTa | 2021 | Microsoft | Disentangled attention; SuperGLUE 89.9 > human |
| HumanEval | 2021 | OpenAI | Codex; 164 code tasks; pass@k metric |
| LoRA | 2021 | Microsoft | 10,000x fewer params; 0 inference latency |
| MMLU | 2021 | UC Berkeley | 57 tasks benchmark; expert-level ~89.8% |
| Pre-train, Prompt, Predict | 2021 | CMU | Taxonomy: 5 dimensions x prompt methods |
| Revisiting BERT Fine-tuning | 2021 | Stanford | 3 stability fixes for few-sample |
| CoT | 2022 | Google | PaLM 540B GSM8K 57%; emergent at 100B+ |
| Constitutional AI | 2022 | Anthropic | RLAIF; harmless + non-evasive |
| Flash Attention | 2022 | Stanford | IO-aware tiling; 7.6x speedup; linear memory |
| Flan-T5/PaLM | 2022 | Google | 1.8K tasks; MMLU 75.2%; +9.4% vs PaLM |
| Formal Algorithms for Transformers | 2022 | DeepMind | Pseudocode for all Transformer variants |
| InstructGPT | 2022 | OpenAI | RLHF; 1.3B > 175B GPT-3 by humans |
| OPT | 2022 | Meta | 175B open; 1/7 carbon of GPT-3 |
| Speculative Decoding | 2022 | Google | Draft + verify; 2-3x speedup; exact distribution |
| Baichuan 2 | 2023 | Baichuan Inc. | 7B/13B bilingual; 2.6T tokens; strong on Chinese benchmarks |
| DPO | 2023 | Stanford | Direct preference optimization; no RM, no RL |
| Falcon | 2023 | TII Abu Dhabi | 7B/40B/180B; RefinedWeb; web data > curated corpora |
| Flash Attention 2 | 2023 | Stanford | 2x vs FlashAttention; 50-73% peak TFLOPS |
| Gemini | 2023 | Google | Native multimodal; Ultra MMLU 90.04% > human-expert |
| GPT-4 | 2023 | OpenAI | MMLU 86.4%; bar exam 90th%; multimodal |
| Harnessing LLMs | 2023 | Amazon | LLM vs fine-tuned guide; scaling taxonomy |
| LLaMA | 2023 | Meta | 13B > GPT-3 175B; RMSNorm+SwiGLU+RoPE |
| LLaMA 2 | 2023 | Meta | 70B RLHF; GQA; Ghost Attention; ≈ChatGPT |
| Mamba | 2023 | CMU/Princeton | Selective SSM; linear time; 5x throughput |
| Mistral 7B | 2023 | Mistral | 7B > Llama 2 13B; GQA + SWA |
| PaLM 2 | 2023 | Google | Smaller+better data > larger; multilingual first |
| Phi-2 | 2023 | Microsoft | UltraFastBERT; FFF; 0.3% neurons; 78x speedup |
| QLoRA | 2023 | UW | 4-bit + LoRA; 65B on 1 GPU; 99.3% ChatGPT |
| ReAct | 2023 | Princeton/Google | Reasoning + acting; LLM agent paradigm |
| RetNet | 2023 | Microsoft | Retention mechanism; 8.4x faster; 70% less memory |
| RWKV | 2023 | EleutherAI | Parallel training + O(Td) RNN inference; 14B |
| Self-Consistency | 2023 | Google | Majority vote over CoT paths; +17.9% GSM8K |
| Self-RAG | 2023 | UW/Allen AI | Adaptive retrieval + reflection tokens |
| Toolformer | 2023 | Meta | Self-supervised tool use; 6.7B > GPT-3 175B |
| Tree of Thoughts | 2023 | Princeton/DeepMind | BFS/DFS over thoughts; Game of 24: 74% vs CoT 4% |
| DeepSeek-Coder-V2 | 2024 | DeepSeek | 236B MoE code model; > GPT-4 Turbo on code+math; 338 languages |
| DeepSeek-V2 | 2024 | DeepSeek | MLA 93.3% KV cache reduction; 236B/21B active; 5.76x throughput |
| DeepSeek-V3 Technical Report | 2024 | DeepSeek | 671B MoE trained for $5.6M; ≈GPT-4o quality; FP8 training |
| Gemma | 2024 | Google DeepMind | 2B/7B open-weight; > Mistral 7B on 11/18 bench |
| Gemma 2 | 2024 | Google DeepMind | 2B/9B/27B; distillation > next-token prediction for small models |
| GLM-4 | 2024 | Zhipu AI / Tsinghua | 83.3 MMLU; 128K context; autonomous tool use; 10M+ downloads |
| InternLM2 | 2024 | Shanghai AI Lab | COOL RLHF; 200K context; 1.8B-20B; conditional reward model |
| Jamba | 2024 | AI21 Labs | Hybrid Transformer-Mamba + MoE; 8x less KV cache; 256K context |
| LLaMA 3 | 2024 | Meta | 8B/70B/405B; 15.6T tokens; ≈GPT-4; dense Transformer + DPO |
| Mixtral of Experts | 2024 | Mistral | 8x7B SMoE; 47B total, 13B active; ≥ Llama 2 70B |
| Nemotron-4 340B | 2024 | NVIDIA | 98% synthetic alignment data; #1 RewardBench (92.0) |
| Phi-3 | 2024 | Microsoft | 3.8B ≈ Mixtral 8x7B; data quality > model size; runs on iPhone |
| Phi-4 | 2024 | Microsoft | 14B; 40% synthetic data; surpasses teacher GPT-4o on STEM |
| Qwen 2.5 | 2024 | Alibaba | 0.5-72B; 18T tokens; DPO + GRPO; 72B ≈ Llama-3-405B |
| Qwen2 | 2024 | Alibaba | 0.5-72B; 84.2 MMLU; 7T tokens; ~30 languages |
| XGrammar | 2024 | Anonymous (in vLLM) | CFG-based structured generation; 100x per-token speedup |
| Yi | 2024 | 01.AI | 6B/34B bilingual; data quality focus; 76.3 MMLU; 200K context |
| DeepSeek-R1 | 2025 | DeepSeek | Pure RL emergent reasoning; ≈o1; distillation preserves reasoning |
| Kimi k1.5 | 2025 | Moonshot AI | RL scaling; 77.5 AIME; ≈o1; long2short CoT compression |
| Nemotron-Research-Tool-N1 | 2025 | NVIDIA | Pure RL with binary reward; 7B/14B > GPT-4o; SFT not needed |
| Qwen3 | 2025 | Alibaba | Dense + MoE; unified thinking/non-thinking; 36T tokens; 119 languages |
| ToolRL | 2025 | OSU/CISPA/Oxford | Cold-start GRPO > SFT+RL; step-level granular reward; BFCL 58.38% |
| ASTRA | 2026 | Beike/LianjiaTech | SFT→RLVR for tool agents; BFCL-MT 64.25; verifiable arenas |

---

## Recent additions (2026-04-12)

- **ASTRA** (2026) — Beike, SFT→RLVR for tool-use agents, verifiable multi-turn arenas
- **ToolRL** (2025) — reward design for tool-use RL, cold-start GRPO > SFT+RL
- **Nemotron-Research-Tool-N1** (2025) — NVIDIA, pure RL with binary reward beats GPT-4o
- New concept: RLVR (Reinforcement Learning with Verifiable Rewards)
- Updated concept: GRPO — added tool-use applications (ASTRA, ToolRL, adaptive batch filling)

## Previous additions (2026-04-07)

- **Karpathy — Unreasonable Effectiveness of RNNs** (2015) — char-level RNN
- **Karpathy — Recipe for Training Neural Networks** (2019) — training methodology
- **Gemma** (2024) — Google DeepMind open-weight
- **Mixtral of Experts** (2024) — Mistral SMoE
- **Qwen2** (2024) — Alibaba multilingual LLM
- New concepts: nanoGPT, LoRA, DPO, Constitutional AI, PEFT, Flash Attention, Speculative Decoding, KV-Cache, ColBERT, REALM, Self-RAG, ReAct, Toolformer, Tree of Thoughts, Self-Consistency, MMLU, HumanEval, Mistral 7B, Mixtral, Mamba, RWKV, RetNet, Gemma, Qwen2

---

## Courses (6 courses, 29 notes)

| Course | Notes | Status |
|---|---|---|
| [[02 Areas/ML & DL/Courses/Stanford CS224N/_index\|Stanford CS224N]] | 15 lectures | Full course |
| [[02 Areas/ML & DL/Courses/Stanford CS336/_index\|Stanford CS336]] | 4 topics | In progress |
| [[02 Areas/ML & DL/Courses/MIT 6.S191/_index\|MIT 6.S191]] | 4 topics | In progress |
| [[02 Areas/ML & DL/Courses/Lena Voita NLP/_index\|Lena Voita NLP]] | 3 topics | In progress |
| [[02 Areas/ML & DL/Courses/SHAD LLM/_index\|SHAD LLM]] | 1 week | Week 1 done |
| [[02 Areas/ML & DL/Courses/MIPT NLP/_index\|MIPT NLP]] | 1 topic (ANCE) | In progress |
| [[02 Areas/ML & DL/Courses/Ранжирование\|Ранжирование]] | 1 note | Standalone |

---

## Timeline

→ Full timeline: [[02 Areas/ML & DL/Timeline|Timeline.md]]

2015 Karpathy RNNs · 2017 Transformer, Adversarial Training · 2018 BERT, GPT 1.0 · 2019 GPT-2, RoBERTa, Karpathy Recipe · 2020 GPT-3, T5, RAG, ColBERT, REALM, DeBERTa · 2021 LoRA, MMLU, HumanEval, Pre-train Prompt Predict, Revisiting BERT FT · 2022 InstructGPT, CoT, OPT, Flan-T5/PaLM, Formal Algorithms, Constitutional AI, Flash Attention, Speculative Decoding · 2023 LLaMA, GPT-4, PaLM 2, LLaMA 2, Mistral 7B, DPO, QLoRA, Flash Attention 2, Mamba, RWKV, RetNet, Phi-2, ReAct, Toolformer, ToT, Self-Consistency, Self-RAG, Harnessing LLMs, Baichuan 2, Falcon, Gemini · 2024 Gemma, Gemma 2, Mixtral, Qwen2, Qwen 2.5, DeepSeek-V2, DeepSeek-V3, DeepSeek-Coder-V2, LLaMA 3, Jamba, GLM-4, InternLM2, Nemotron-4, Phi-3, Phi-4, XGrammar, Yi · 2025 DeepSeek-R1, Kimi k1.5, Qwen3, ToolRL, Nemotron-Tool-N1 · 2026 ASTRA

---

## Quality status (2026-04-07)

- Papers reviewed: 71/71
- Concept articles: 104 (35 Architectures, 26 Training, 18 NLP, 10 Inference, 7 Retrieval, 5 Reasoning, 3 Evaluation)
- Categories: 7 (Architectures, Training, Inference, Retrieval, NLP, Reasoning, Evaluation)
- Courses: 6 (Stanford CS224N 15, Stanford CS336 4, MIT 6.S191 4, Lena Voita NLP 3, SHAD LLM 1, MIPT NLP 1) + Ранжирование standalone
- Phase 1 (paper rewrites): DONE
- Phase 2 (concept rewrites): DONE
- Phase 3 (indexes + quality): DONE
- Known gaps: Contrastive Learning, Hard Negatives, RetroMAE (MIPT NLP -- нет статьи-источника)

---

*Last updated: 2026-04-12*
