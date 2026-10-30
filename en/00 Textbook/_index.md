---
title: Machine Learning and Language Models
type: textbook-chapter
status: active
locale: en
translation_status: current
canonical_revision: "94a06ce389fdf27af3290fdaa9447abaeb3aade37ef1ebfb7dcd9c259560557f"
translation_reviewed: "2026-10-30"
translation_of: "00 Учебник/_index.md"
last_updated: 2026-10-30
last_verified: 2026-09-15
---

# Machine Learning and Language Models

For a first reading, start with [[00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|vectors, matrices, and tensors]]. See [[00 Учебник/00 Как пользоваться учебником|How to use this textbook]] for other routes.

The contents below match the core course and independent modules in the canonical textbook. **EN** marks a chapter with a separate English text. Unmarked chapters open the canonical-language edition; original course excerpts retain their source language. A page marked partial still has known omissions, which are stated on that page. This contents page is complete; the English edition of the whole book is not.

Original lectures, books, slides, and assignments are collected in the [[05 Источники/Курсы|course library]]. Chapters link to the specific materials used in their explanations.

<a id="curriculum"></a>
<a id="полный-маршрут"></a>

## Core course

The core course runs from mathematics and classical models through neural networks, the Transformer, and language-model training. The specialist modules below can be read independently after their stated prerequisites.


### I. Mathematical and machine-learning foundations

- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|Vectors, matrices, and tensors]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/02 Производная и градиент|Derivatives and gradients]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм|Probability, likelihood, and logarithms]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/04 Функции потерь|Loss functions]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/05 Train, validation и test|Train, validation, and test]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание|Generalization, regularization, and evaluation]]


### II. Classical machine learning

- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/00 Карта модуля|Module map and sources]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline|The learning problem and prediction pipeline]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация|Linear regression and classification]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/03 Деревья решений|Decision trees]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting|Bagging, random forests, and gradient boosting]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога|Evaluation, cross-validation, and threshold selection]]

<a id="iv-нейронные-сети"></a>

### III. Neural networks

- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|The neuron and multilayer perceptron]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации|Activation functions]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|Computational graphs and backpropagation]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|SGD, Momentum, Adam, and learning-rate schedules]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections|Initialization, normalization, and residual connections]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация|Dropout and regularization]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований|Data augmentation: Mixup, CutMix, and transformation policies]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|CNNs: from convolution to ResNet]]

<a id="v-текст-до-transformer"></a>

### IV. Text before the Transformer

- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|Representing text numerically]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель|N-gram language models]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction|Text probability and next-token prediction]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings|Word2Vec, GloVe, and distributional representations]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece, and Unigram]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|RNNs and backpropagation through time]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU|LSTM and GRU]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора|Seq2Seq and the fixed-vector bottleneck]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|Attention for translation]]

<a id="vi-transformer-bert-и-gpt"></a>

### V. Transformer, BERT, and GPT

- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention: Q, K, and V]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Masking, heads, and tensor shapes]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|Positional information]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|The complete encoder–decoder Transformer]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three architectural patterns]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa, and DeBERTa]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1: generative pre-training]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2: zero-shot through language]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3: in-context learning]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5: the text-to-text Transformer]] · **EN**

<a id="vii-анатомия-современной-llm"></a>

### VI. Anatomy of a modern LLM

- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA as a baseline architecture]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block|The modern decoder block]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|Pre-norm, RMSNorm, SwiGLU, and residual connections]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE|RoPE]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating|Dense FFNs: dimensional expansion and SwiGLU]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|Transformer from scratch: shapes, parameters, and cost]]

<a id="ix-обучение-llm"></a>
<a id="viii-обучение-llm"></a>

### VII. Language-model pre-training

After the decoder architecture, the course turns to the corpus, training objective, and choice of model size under a limited training budget.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|Data and pre-training]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Data collection, cleaning, and mixtures]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса|Deduplication, PII, and corpus quality]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|The pre-training objective]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Scaling laws]]


### VIII. Post-training and reasoning

A pretrained model already predicts the next token. Post-training changes the examples, quality signal, and optimization procedure used to shape its behavior.

- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники|Post-training map and sources]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT and instruction data]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01a Fine-tuning и PEFT — adapters, LoRA, QLoRA и soft prompts|Fine-tuning and PEFT]] · **EN original**
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data|Preference data]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|Reward modeling]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|Policy gradients and PPO for LLMs]]
- [[00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF|Alignment goals, HHH, Constitutional AI, and RLAIF]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|DPO]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR and verifiers]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|GRPO and DeepSeek-R1]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation|Reasoning distillation]]
- [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Test-time compute]]

<a id="x-инференс-и-serving"></a>
<a id="ix-инференс-и-serving"></a>

### IX. Generation and evaluation

Token probabilities become generated answers; model quality is then evaluated on tasks with explicit checks for test-data contamination.

- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Decoding and selecting the next token]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Model evaluation and contamination]]

<a id="углублённые-модули"></a>

## Specialist modules

Each module has its own chapter sequence. Its last chapter returns to the module selection instead of treating the next independent subject as a required continuation.


### Clustering, dimensionality reduction, and kernels

Prerequisites: linear algebra, probability, and evaluation. Read this module after classical machine learning; Transformer architecture is not required.

- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения|Clustering and its limitations]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian|Spectral clustering and the graph Laplacian]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/07 PCA, t-SNE и UMAP|PCA, t-SNE, and UMAP]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM|Gaussian mixtures and EM]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/09 Kernels, SVM и random Fourier features|Kernels, SVMs, and random Fourier features]]


### Generative image models

Start after CNNs and backpropagation. The sequence moves from latent representations and VAEs to GANs, diffusion models, and the use of synthetic images.

- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|Autoencoders and VAEs]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/09 GAN и DCGAN — генератор и дискриминатор|GAN and DCGAN: generator and discriminator]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|Diffusion, latent diffusion, and flow matching]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции|Synthetic images and sim-to-real transfer]]


<a id="контрастивное-и-маскированное-предобучение"></a>

### Contrastive and masked pre-training

Prerequisites: neural networks, data augmentation, and self-attention. SimCLR and MoCo learn representations by contrasting examples; MAE learns by reconstructing masked patches. MDLM and LLaDA use masking to train a generative model of text.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений|Contrastive pre-training: SimCLR]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41g Momentum contrast — MoCo, очередь и key encoder|MoCo: the queue and momentum encoder]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41h Masked autoencoders — MAE и восстановление патчей|MAE: reconstructing masked patches]]
- [[00 Учебник/11 Pre-training и Scaling/41j Диффузионные языковые модели — MDLM и LLaDA|Masked diffusion: MDLM and LLaDA]]


### Data selection and partial labels

Prerequisites: loss functions, data splits, and augmentation. The module examines which examples to label, select, or generate, and how to test whether those choices help.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting|Example selection: EL2N, GraNd, and forgetting]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|Active learning: requesting labels]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding|Geometric selection: k-center, coresets, and herding]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|Partial labels: pseudo-labeling and FixMatch]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества|Data selection: cost, shards, and quality checks]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|Synthetic data and curricula]]


### Efficient attention, MoE, and alternatives to the Transformer

Continue the study of LLM architecture with K/V storage, long histories, expert selection, and recurrent state. Read the dense-FFN chapter before MoE.

- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|MHA, MQA, and GQA]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA and KV-cache compression]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание|Long context: extension, sparsity, and evaluation]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts: routing, capacity, and serving]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|Mamba, RWKV, RetNet, and hybrid architectures]]

<a id="viii-вычислительные-основы-ml-систем"></a>
<a id="vii-вычислительные-основы-ml-систем"></a>

### Computational foundations of ML systems

Prerequisites: tensor shapes, matrix multiplication, and backpropagation. The expert-related calculations in the arithmetic chapter also require MoE.

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/01 Модель как часть системы|The model as part of a system]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPUs, CUDA, and the memory hierarchy]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Performance measurement and roofline]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Transformer and MoE arithmetic]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Numerical formats and mixed precision]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipelines, padding, and packing]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML workloads]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|GPU kernels and Triton: from program to measurement]]


### Distributed training

Start after GPUs, numerical formats, and performance measurement in the ML-systems module. Read MoE before expert parallelism.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision|Distributed training and mixed precision]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|Processes, collectives, and DDP]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload|Gradient checkpointing and offload]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism|Tensor and sequence parallelism]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44d Pipeline parallelism|Pipeline parallelism]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor|ZeRO, FSDP2, DeviceMesh, and DTensor]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism|Expert and hybrid parallelism]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints|Networks, storage, and distributed checkpoints]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration|Fault tolerance and fleet orchestration]]


### Inference and serving

Prerequisites: decoding, GPUs, and roofline analysis. KV-cache calculations also use MHA/MQA/GQA from the efficient-architecture module. The sequence then covers memory, request scheduling, kernels, compression, and distributed serving.

- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|LLM inference physics: prefill, decode, and roofline]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV cache, batching, and PagedAttention]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Scheduling: continuous batching, chunked prefill, and prefix caching]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines: vLLM, SGLang, TensorRT-LLM, and FlashInfer]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Quantizing language models]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload|KV-cache compression and offload]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank|Pruning, distillation, and low-rank compression]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Speculative decoding]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving|Parallelism and collective operations]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|Disaggregated prefill and decode serving]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLOs, and inference operations]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning|Queueing and capacity planning]]

<a id="xi-эксплуатация-и-оценивание"></a>
<a id="x-эксплуатация-и-оценивание"></a>

### Operating ML systems

Prerequisites: data splits, evaluation, and the training pipeline. This module covers model releases, service observability, device constraints, and errors across different groups of examples.

- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow|The ML workflow]]
- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps|MLOps: from an artifact to an observable service]]
- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment|Edge and federated deployment]]
- [[00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем|Security and privacy of ML systems]]
- [[00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем|Robustness of ML systems]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Несбалансированная классификация|Imbalanced classification]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Responsible systems|Responsible ML systems]]

<a id="xii-retrieval-и-rag"></a>
<a id="xi-retrieval-и-rag"></a>

### Retrieval and RAG

Prerequisites: vector representations, BERT, and evaluation. The sequence covers embedding training, retrieval, reranking, and generating an answer from retrieved context.

- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|Embeddings and metric learning]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid|Retrieval: from BM25 to dense and hybrid methods]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction|Reranking: cross-encoders and late interaction]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер|RAG: the complete pipeline]]

<a id="xiii-мультимодальные-модели"></a>
<a id="xii-мультимодальные-модели"></a>

### Multimodal models

Start after CNNs and the Transformer. Follow visual representations into language-model answers, then study documents, video, audio, and evaluation. Image generation has a separate module.

- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Multimodal models]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64a Connectors и fusion|Connectors and fusion]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции|Resolution, tiling, and spatial positions]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Training VLMs: alignment, instruction tuning, and data]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding|Documents, OCR, and visual grounding]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|Video, audio, and omni models]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|Evaluation, failure modes, and VLM serving]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|Unified multimodal sequences and Chameleon]]

<a id="xiv-инструменты-и-агенты"></a>
<a id="xiii-инструменты-и-агенты"></a>

### Tools and agents

Prerequisites: generation, post-training, and language-model evaluation. Study tool calls and context management before memory, planning, applications, and security.

- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля|Module map and sources]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|Tool use: from a function call to an action]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|Agent harnesses and context engineering]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Agent memory, planning, and orchestration]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|Evaluating agent systems]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Coding, web, and computer-use agents]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Formal proofs and mathematical agents]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Scientific search and discovery agents]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Agent-system security]]

<a id="iii-рекомендательные-системы"></a>
<a id="xiv-рекомендательные-системы"></a>

### Recommender systems

Matrix factorization requires linear algebra, loss functions, and data splits. Neural recommenders also require MLPs and backpropagation; sequential Transformer recommenders require self-attention.

- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/00 Карта модуля|Module map and complete sources]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline|Recommendation tasks, data, and pipelines]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/02 Collaborative filtering и matrix factorization|Collaborative filtering and matrix factorization]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели|Ranking and neural models]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки|Sequential recommendations and features]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация|Evaluation and operations]]

<a id="другие-режимы"></a>

## Reference sections

- [[02 Areas/ML & DL/01 Справочник/_index|ML, DL, and LLM topic reference]]
- [[02 Areas/ML & DL/02 Атлас моделей/_index|Models and families]]
- [[02 Areas/ML & DL/02 Атлас моделей/Timeline|Timeline of models and technologies]]
- [[02 Areas/ML & DL/03 Исследовательские линии/_index|Research overviews]]
- [[02 Areas/ML & DL/04 Вопросы/_index|Questions linked to explanations]]
- [[02 Areas/ML & DL/05 Источники/_index|Courses, papers, and original materials]]
- [[02 Areas/ML & DL/06 Практика/_index|Practice]]
