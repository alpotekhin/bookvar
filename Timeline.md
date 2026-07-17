---
title: ML & DL Timeline
type: timeline
last_updated: "2026-04-07"
---

# ML & DL Timeline

Chronological view of all 47 papers in the knowledge base with key quantitative results.

---

## 2015

- **The Unreasonable Effectiveness of Recurrent Neural Networks** → [[Papers/Karpathy — Unreasonable Effectiveness of RNNs]] — char-level RNN/LSTM генерирует Shakespeare, LaTeX, C-код Linux kernel. Выпуск [char-rnn](https://github.com/karpathy/char-rnn).

---

## 2017

- **Attention Is All You Need** → [[Papers/Attention Is All You Need]] — **Transformer**: первая архитектура без RNN/CNN; WMT EN-DE **28.4 BLEU**, EN-FR **41.8 BLEU**. Фундамент всех современных LLM.
- **Adversarial Training Methods for Semi-Supervised Text Classification** → [[Papers/Adversarial Training Methods for Semi-Supervised Text Classification]] — VAT на word embeddings в LSTM; IMDB **5.91%** error; semi-supervised SOTA.

---

## 2018

- **BERT: Pre-training of Deep Bidirectional Transformers** → [[Papers/BERT]] — bidirectional MLM + NSP; GLUE **80.5** (+7.7 vs GPT); SOTA на 11 NLP задач. Установил paradigm pre-train → fine-tune.

---

## 2019

- **GPT-2: Language Models are Unsupervised Multitask Learners** → [[Papers/GPT 2.0]] — 1.5B decoder-only; zero-shot multi-task generalization; WebText 40GB.
- **RoBERTa: A Robustly Optimized BERT Pretraining Approach** → [[Papers/RoBERTa]] — GLUE **88.5** (+8 vs BERT); no NSP; dynamic masking. BERT был значительно недообучен.
- **A Recipe for Training Neural Networks** → [[Papers/Karpathy — Recipe for Training Neural Networks]] — 6-этапный рецепт отладки и обучения нейросетей. Один из самых цитируемых практических постов.

---

## 2020

- **GPT-3: Language Models are Few-Shot Learners** → [[Papers/GPT 3.0]] — **175B**; in-context learning; TriviaQA **71.2%** few-shot. Decoder-only era begins.
- **T5: Exploring the Limits of Transfer Learning** → [[Papers/T5]] — text-to-text unification; encoder-decoder; C4 750GB; T5-11B: GLUE **~90.3**.
- **RAG: Retrieval-Augmented Generation** → [[Papers/Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks]] — DPR + BART end-to-end; NQ **44.5 EM**; SOTA на 3 open-QA tasks.
- **ColBERT: Efficient and Effective Passage Search** → [[Papers/ColBERT]] — late interaction MaxSim; **170x** быстрее BERT ranker; **14,000x** меньше FLOPs.
- **REALM: Retrieval-Augmented Language Model Pre-Training** → [[Papers/REALM]] — end-to-end retriever + MLM; **+4-16%** absolute на Open-domain QA.

---

## 2021

- **DeBERTa: Decoding-enhanced BERT with Disentangled Attention** → [[Papers/DeBERTa]] — disentangled attention + EMD; SuperGLUE **89.9 > human 89.8**. Первая модель, превзошедшая человека.
- **LoRA: Low-Rank Adaptation of Large Language Models** → [[Papers/LoRA]] — **10,000x** fewer params; 3x less VRAM; **0** inference latency. Де-факто стандарт PEFT.
- **MMLU: Measuring Massive Multitask Language Understanding** → [[Papers/MMLU]] — **57 задач** multiple choice; GPT-3 43.9%; expert **~89.8%**. Де-факто стандарт оценки LLM.
- **HumanEval: Evaluating Large Language Models Trained on Code** → [[Papers/HumanEval]] — Codex-12B: **28.8%** pass@1; **164** рукописные задачи; метрика pass@k.
- **Pre-train, Prompt, and Predict** → [[Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing]] — таксономия: 5 измерений x prompt methods; 100+ работ.
- **Revisiting Few-sample BERT Fine-tuning** → [[Papers/Revisiting Few-sample BERT Fine-tuning]] — 3 fix'а: debiased Adam + re-init + longer training. Устраняет нестабильность few-sample.

---

## 2022

- **InstructGPT: Training LMs to Follow Instructions** → [[Papers/InstructGPT]] — RLHF (SFT → RM → PPO); **1.3B > 175B GPT-3** по оценке людей; hallucination **2x ниже**. Основа ChatGPT.
- **Chain-of-Thought Prompting Elicits Reasoning** → [[Papers/COT]] — reasoning steps в промпте; PaLM 540B GSM8K **57%**; emergent ability >= 100B.
- **OPT: Open Pre-trained Transformer Language Models** → [[Papers/OPT]] — 175B open; **1/7 carbon** vs GPT-3; training logbook опубликован.
- **Flan-T5 / Flan-PaLM: Scaling Instruction-Finetuned LMs** → [[Papers/Flan-T5-PaLM]] — 1.8K tasks; MMLU **75.2%**; +9.4% vs PaLM 540B. Open Flan-T5 checkpoints.
- **Formal Algorithms for Transformers** → [[Papers/Formal Algorithms for Transformers]] — pseudocode для всех Transformer вариантов; ~50 строк покрывают всё.
- **Constitutional AI: Harmlessness from AI Feedback** → [[Papers/Constitutional AI]] — RLAIF; "конституция" из ~16 правил; harmless + non-evasive assistant.
- **FlashAttention: Fast and Memory-Efficient Exact Attention** → [[Papers/Flash Attention]] — IO-aware tiling в SRAM; **7.6x** speedup attention; **линейная** память.
- **Speculative Decoding** → [[Papers/Speculative Decoding]] — draft model + target verification; **2-3x** speedup; exact output distribution.

---

## 2023

- **LLaMA: Open and Efficient Foundation Language Models** → [[Papers/LLaMA]] — 7-65B; RMSNorm + SwiGLU + RoPE; **13B > GPT-3 175B** на большинстве бенчмарков.
- **GPT-4 Technical Report** → [[Papers/GPT 4.0]] — multimodal; MMLU **86.4%**; bar exam **90th percentile**; predictable scaling.
- **PaLM 2 Technical Report** → [[Papers/PaLM 2]] — smaller + better data > PaLM-540B; Polish NLU **94%**; multilingual first-class.
- **Llama 2: Open Foundation and Fine-Tuned Chat Models** → [[Papers/LLaMA 2]] — 70B; 2T tokens; GQA; Ghost Attention; iterative RLHF. **Llama 2-Chat ~ ChatGPT**.
- **Harnessing the Power of LLMs in Practice** → [[Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] — practical guide: LLM vs fine-tuned; emergent/inverse/U-shape scaling.
- **Mistral 7B** → [[Papers/Mistral 7B]] — **7B > Llama 2 13B** на всех бенчмарках; GQA + SWA; Apache 2.0.
- **DPO: Direct Preference Optimization** → [[Papers/DPO]] — alignment без RM и RL; binary CE loss; on par with PPO-RLHF.
- **QLoRA: Efficient Finetuning of Quantized LLMs** → [[Papers/QLoRA]] — 4-bit + LoRA; **65B на 1 GPU** (48GB); Guanaco **99.3%** от ChatGPT.
- **FlashAttention-2** → [[Papers/Flash Attention 2]] — **2x** vs FlashAttention; 50-73% peak TFLOPs на A100.
- **Mamba: Linear-Time Sequence Modeling** → [[Papers/Mamba]] — selective SSM; **linear time**; Mamba-3B ~ Transformer-6B+; **5x** throughput.
- **RWKV: Reinventing RNNs for the Transformer Era** → [[Papers/RWKV]] — parallel training + **O(Td)** inference; **14B** — крупнейшая dense RNN.
- **RetNet: Retentive Network** → [[Papers/RetNet]] — retention mechanism; **8.4x** faster decoding; **70%** less memory.
- **Phi-2 / UltraFastBERT** → [[Papers/Phi-2]] — fast feedforward networks (FFF); **0.3%** нейронов; CPU **78x** speedup.
- **ReAct: Synergizing Reasoning and Acting** → [[Papers/ReAct]] — reasoning traces + actions в промпте; основа LLM-агентов (LangChain, AutoGPT).
- **Toolformer: Language Models Can Teach Themselves to Use Tools** → [[Papers/Toolformer]] — self-supervised tool use; GPT-J **6.7B > GPT-3 175B** на math.
- **Tree of Thoughts: Deliberate Problem Solving** → [[Papers/Tree of Thoughts]] — BFS/DFS по дереву мыслей; Game of 24: **74%** vs CoT **4%** (GPT-4).
- **Self-Consistency Improves Chain of Thought Reasoning** → [[Papers/Self-Consistency]] — majority vote по CoT paths; GSM8K **+17.9%** на PaLM-540B.
- **Self-RAG: Learning to Retrieve, Generate, and Critique** → [[Papers/Self-RAG]] — adaptive retrieval + reflection tokens; **7B/13B > ChatGPT** на QA/fact-check.

---

## 2024

- **Gemma: Open Models Based on Gemini Research** → [[Papers/Gemma]] — 2B/7B open-weight; > Mistral 7B на 11/18 бенчмарков; GSM8K **46.4** vs Mistral 35.4.
- **Mixtral of Experts** → [[Papers/Mixtral of Experts]] — 8x7B SMoE; 47B total, **13B active**; >= Llama 2 70B + GPT-3.5; Apache 2.0.
- **Qwen2 Technical Report** → [[Papers/Qwen2]] — 0.5-72B; **84.2 MMLU**; 7T tokens; ~30 языков; Qwen2-72B > Llama-3-70B.

---

*47 papers total. Last updated: 2026-04-07*
