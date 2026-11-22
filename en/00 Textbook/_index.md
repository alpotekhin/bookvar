---
title: Machine Learning and Language Models
type: textbook-chapter
status: active
locale: en
translation_status: current
canonical_revision: "8b45d042501fa68c95cd15849ef51bd61a7761574fb7edcc405799359e378135"
translation_reviewed: "2026-11-14"
translation_of: "00 Учебник/_index.md"
last_updated: 2026-11-14
last_verified: 2026-09-15
---

# Machine Learning and Language Models

For a first reading, start with [[00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|vectors, matrices, and tensors]]. Jump to the [[#curriculum|core course]] for the chapter list, or see [[00 Учебник/00 Как пользоваться учебником|How to use this textbook]] for other routes.

The contents below match the core course and independent modules in the canonical textbook. **EN** marks a chapter with a separate English text. Unmarked chapters open the canonical-language edition; original course excerpts retain their source language. A page marked partial still has known omissions, which are stated on that page. This contents page is complete; the English edition of the whole book is not.

<a id="curriculum"></a>
<a id="полный-маршрут"></a>

## Core course

The course starts with mathematics and neural networks. It then covers language models, the Transformer, how to train and evaluate an LLM, and how to connect one to retrieval and tools.

### I. Mathematical and machine-learning foundations

- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|Vectors, matrices, and tensors]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/02 Производная и градиент|Derivatives and gradients]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм|Probability, likelihood, and logarithms]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/04 Функции потерь|Loss functions]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/05 Train, validation и test|Train, validation, and test]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание|Generalization, regularization, and evaluation]]

<a id="iv-нейронные-сети"></a>
<a id="iii-neural-networks"></a>

### II. Neural networks

- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|The neuron and multilayer perceptron]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации|Activation functions]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|Computational graphs and backpropagation]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|SGD, Momentum, Adam, and learning-rate schedules]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections|Initialization, normalization, and residual connections]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация|Dropout and regularization]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований|Data augmentation: Mixup, CutMix, and transformation policies]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|CNNs: from convolution to ResNet]]

<a id="v-текст-до-transformer"></a>
<a id="iv-text-before-the-transformer"></a>

### III. Tokenization and language models

- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|Representing text numerically]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель|N-gram language models]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction|Text probability and next-token prediction]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings|Word2Vec, GloVe, and distributional representations]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece, and Unigram]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/03 Морфология и разметка последовательностей|Morphology and sequence labeling]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/04 Синтаксический разбор|Syntactic parsing]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|RNNs and backpropagation through time]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU|LSTM and GRU]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора|Seq2Seq with a fixed context]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|Attention for translation]]

<a id="vi-transformer-bert-и-gpt"></a>
<a id="v-transformer-bert-and-gpt"></a>

### IV. Transformer, BERT, and T5

- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-attention: Q, K, and V]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Masking, heads, and tensor shapes]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|Positional information]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|The complete encoder–decoder Transformer]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Three Transformer architectures]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa, and DeBERTa]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03a Распознавание именованных сущностей|Named entity recognition]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03b Классификация документов|Document classification]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03c Извлечение информации из документов|Information extraction from documents]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5: the text-to-text Transformer]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/08 Машинный перевод|Machine translation]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/09 Суммаризация текста|Text summarization]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/10 Ответы на вопросы|Question answering]] · **EN**

<a id="vii-анатомия-современной-llm"></a>
<a id="vi-anatomy-of-a-modern-llm"></a>

### V. From GPT-1 to LLaMA

- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1: generative pre-training]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2: tasks as text continuations]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3: in-context learning]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA as a baseline architecture]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block|The modern decoder block]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|Pre-norm, RMSNorm, SwiGLU, and residual connections]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE|RoPE]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating|Dense FFNs: dimensional expansion and SwiGLU]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|Transformer from scratch: shapes, parameters, and cost]]

<a id="efficient-attention-moe-and-alternatives-to-the-transformer"></a>

### VI. Long-context attention and MoE

- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|MHA, MQA, and GQA]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA and KV-cache compression]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание|Long context: extension, sparsity, and evaluation]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts: routing, capacity, and serving]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|Mamba, RWKV, RetNet, and hybrid architectures]]

<a id="ix-обучение-llm"></a>
<a id="viii-обучение-llm"></a>

### VII. Language-model pre-training

After the decoder architecture, the course turns to the corpus, training objective, and choice of model size under a limited training budget.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|Data and pre-training]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Data collection, cleaning, and mixtures]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса|Deduplication, PII, and corpus quality]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|The pre-training objective]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Scaling laws]]
- [[00 Учебник/11 Pre-training и Scaling/41j Диффузионные языковые модели — MDLM и LLaDA|Masked diffusion: MDLM and LLaDA]]

### VIII. Decoding

- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Decoding and selecting the next token]] · **EN**

<a id="viii-post-training-and-reasoning"></a>

### IX. Post-training and reasoning

A pretrained model already predicts the next token. Post-training changes the examples, quality signal, and optimization procedure used to shape its behavior.

- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники|Language model post-training]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT: learning from demonstrations]]
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
<a id="ix-generation-and-evaluation"></a>

### X. Evaluating language models

- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Model evaluation and contamination]]

<a id="xii-retrieval-и-rag"></a>
<a id="xi-retrieval-и-rag"></a>
<a id="retrieval-and-rag"></a>

### XI. Retrieval and RAG

Prerequisites: vector representations, BERT, and evaluation. The sequence covers embedding training, retrieval, reranking, and generating an answer from retrieved context.

- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|Embeddings and metric learning]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid|Retrieval: from BM25 to dense and hybrid methods]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction|Reranking: cross-encoders and late interaction]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер|RAG: the complete pipeline]]

<a id="xiii-мультимодальные-модели"></a>
<a id="xii-мультимодальные-модели"></a>
<a id="multimodal-models"></a>

### XII. Multimodal models

Start after CNNs and the Transformer. Follow visual representations into language-model answers, then study documents, video, audio, and evaluation. Image generation has a separate module.

- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Multimodal models]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64a Connectors и fusion|Connectors and fusion]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции|Resolution, tiling, and spatial positions]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Training vision-language models]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding|Reading documents: OCR, fields, and coordinates]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|How models understand video and answer in speech]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64ea Распознавание речи|Automatic speech recognition]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64eb Синтез речи|Text-to-speech synthesis]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|Evaluating VLMs and measuring serving latency]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|Chameleon: text and images in one sequence]] · **EN**

<a id="xiv-инструменты-и-агенты"></a>
<a id="xiii-инструменты-и-агенты"></a>
<a id="tools-and-agents"></a>

### XIII. Tools and agents

Prerequisites: generation, post-training, and language-model evaluation. Study tool calls and context management before memory, planning, applications, and security.

- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля|How agents use tools]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|How a function call becomes an action]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|How agents manage context]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Agent memory, planning, and orchestration]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|Evaluating agent systems]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Coding, web, and computer-use agents]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Formal proofs and mathematical agents]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Scientific discovery agents]] · **EN**
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Agent system security]] · **EN**

<a id="углублённые-модули"></a>

## Specialist modules

Each module has its own chapter sequence. Its last chapter returns to the module selection instead of treating the next independent subject as a required continuation.


<a id="ii-classical-machine-learning"></a>

### Classical machine learning

- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/00 Карта модуля|Module map and sources]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline|The learning problem and prediction pipeline]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация|Linear regression and classification]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/03 Деревья решений|Decision trees]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting|Bagging, random forests, and gradient boosting]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога|Evaluation, cross-validation, and threshold selection]]

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

### Data selection and partial labels

Prerequisites: loss functions, data splits, and augmentation. The module examines which examples to label, select, or generate, and how to test whether those choices help.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting|Example selection: EL2N, GraNd, and forgetting]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|Active learning: requesting labels]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding|Geometric selection: k-center, coresets, and herding]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|Partial labels: pseudo-labeling and FixMatch]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества|Data selection: cost, shards, and quality checks]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|Synthetic data and curricula]]

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
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs|Graph compilation: PyTorch, JAX, and CUDA Graphs]]

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
<a id="operating-ml-systems"></a>

### ML-system operations

Prerequisites: data splits, evaluation, and the training pipeline. This module covers model releases, service observability, device constraints, and errors across different groups of examples.

- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow|The ML workflow]]
- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps|MLOps: from an artifact to an observable service]]
- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment|Edge and federated deployment]]
- [[00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем|Security and privacy of ML systems]]
- [[00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем|Robustness of ML systems]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Несбалансированная классификация|Imbalanced classification]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Responsible systems|Responsible ML systems]]

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
- [[05 Источники/Курсы|Course library]]
- [[02 Areas/ML & DL/06 Практика/_index|Practice]]

<a id="gpt-decoder-route"></a>

## From GPT-1 to LLaMA

Does `An animal rests.` follow from `A cat sleeps.`? A fine-tuned
GPT-1 classifier supplies the answer. With GPT-2, we can score labels
as continuations after `Relation:`. With GPT-3 and LLaMA, we can put
labeled examples before the pair to show which answers are allowed.
This is a worked example of how to set up the same task; it does
not reproduce the original experiments.

All four models learned to predict the next token during pretraining.
The decoder changed its normalization, positions, and attention mask
over time. Model size and the amount of training data changed separately.

| Chapter | How it answered and what changed |
| --- | --- |
| [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training|GPT-1 (2018)]] | A [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training#Textual entailment|separate head scores three classes]] from the final token state. Labeled sentence pairs fine-tune both the head and the decoder.<br><br>117 million parameters, 512-token context. Pretraining on BooksCorpus; LayerNorm follows residual addition. |
| [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language|GPT-2 (2019)]] | Compare the [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language#gpt2-relation-prefix|probabilities of labels as text]] after `Relation:`. The weights stay fixed.<br><br>Up to 1.5 billion parameters, 1,024-token context. WebText and byte-level BPE; LayerNorm moves before attention and the FFN. |
| [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3 (2020)]] | Put [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-relation-shots|zero, one, or several labeled examples]] before the test pair. To switch tasks, change the prompt.<br><br>Up to 175 billion parameters, 2,048-token context. Eight models each saw 300 billion tokens; full and local attention layers alternate. |
| [[en/00 Textbook/07 Anatomy of a modern LLM/01 LLaMA as a baseline architecture|LLaMA v1 (2023)]] | As in GPT-3, score the label as a continuation. The [[en/00 Textbook/07 Anatomy of a modern LLM/01 LLaMA as a baseline architecture#gpt3-to-llama1|same text splits into different tokens]].<br><br>7 to 65 billion parameters, still a 2,048-token context. Full attention in every layer, RMSNorm, RoPE, and SwiGLU. The models saw 1 to 1.4 trillion tokens. |

The [[#gpt-decoder-route-comparison|illustrated comparison]] follows
one sentiment classification task. It starts with how each model
produces an answer, then compares the decoder and the pretraining data.
Read the four examples in order:
[[#gpt-route-gpt1|GPT-1]] →
[[#gpt-route-gpt2|GPT-2]] →
[[#gpt-route-gpt3|GPT-3]] →
[[#gpt-route-llama|LLaMA]].

[[#gpt-decoder-route-sentiment|One review across all four models]] ·
[[#gpt-decoder-route-architecture|Decoder diagrams]] ·
[[#gpt-decoder-route-depth-width|Depth, width, and parameter count]] ·
[[#gpt-decoder-route-cache|Memory during generation]] ·
[[#gpt-decoder-route-data|Corpora and training length]] ·
[[#gpt-decoder-route-training-budget|Training and inference compute]] ·
[[#gpt-decoder-route-evidence|Experimental results]] ·
[[#gpt-decoder-route-bpb|Comparing loss across tokenizers]] ·
[[#gpt-decoder-route-calibration|Calibrating label scores]]

<a id="gpt-decoder-route-comparison"></a>

## Comparing GPT-1, GPT-2, GPT-3, and LLaMA

[[#gpt-decoder-route|Back to the four chapters]] · [[#curriculum|Core course]]

In the figure below, `book` is the prediction above the input `this`: the decoder can read `this` and the preceding tokens. Next we use the same kind of decoder to classify the review `i liked it`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ntp-review-2026-10/d2l-gpt-decoder-original.svg|The decoder predicts book above the input this. The book row on the right can see the start marker, I, like, and this, but not book.]]

*Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola, [Dive into Deep Learning, Figure 11.9.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#fig-gpt-decoder-only), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original figure, unmodified. The target row for `book` can see the column for `this`, but not the one for `book` itself. The figure shows a full causal mask; a local GPT-3 layer can also exclude positions to the left.*

<a id="gpt-decoder-route-sentiment"></a>

### One review, four generations

<a id="gpt-route-gpt1"></a>

#### GPT-1 (2018): a separate sentiment head

Put the review `i liked it` between the special tokens `Start` and
`Extract`. That is the `Classification`
branch in [Figure 1 of the GPT-1 paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=4).
Causal attention lets the final position read the whole review. A linear
head maps the `Extract` state to two scores, one per class. Fine-tuning
on SST-2 updates both the head and the decoder. The label stays out of
the review and is used to compute the training loss.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt1-pretraining-2026-11-10/cs224n-decoder-classifier.png|The final state h_T receives information from every input position. Only h_T goes to the linear head, which distinguishes two sentiment classes.]]

*Stanford CS224N, [Lecture 9: Pretraining, slide 41](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf#page=41), [course resource reuse notice](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/#logistics). Right-hand diagram with the original labels.*

In the diagram, the last input $w_T$ corresponds to `Extract`, and $h_T$ is its state at the decoder output. Only $h_T$ feeds the `Linear` box. The faces above it are sentiment classes, not tokens appended to the review.

In a [separate experiment](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=8),
the authors dropped the classification head and skipped fine-tuning.
They appended `very` to the review and compared the probabilities of
`positive` and `negative` as continuations. This time the label came
from the vocabulary. GPT-1 already used a vocabulary head during
pretraining. The next Stanford slide shows the prediction above each
input: the state after `w₅` goes through the same projection `A, b`
as the earlier states. There is no separate sentiment output.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-14/cs224n-vocabulary-head.png|Each input position predicts the next token. The final state h_T passes through the same vocabulary head A, b as the earlier states.]]

*Stanford CS224N, John Hewitt, [Lecture 9: Pretraining, slide 42](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf#page=42), [course resource reuse notice](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/#logistics). Right-hand diagram with the original labels. The bias `b` is part of the general diagram; the released GPT-2 code uses a bias-free output projection.*

The slide also describes fine-tuning a language model as a generator.
GPT-2's zero-shot experiments skip that step. To change the task,
change the prefix and keep the pretrained output projection.

<a id="gpt-route-gpt2"></a>

#### GPT-2 (2019): the label as a text continuation

Replace `very` with a prompt ending in
`Review: i liked it` and then `Sentiment:` on a new line. If the next
tokens ` positive` and ` negative` have probabilities 0.18 and 0.07,
choose the positive class. These are illustrative values. They do not
sum to one because the language model assigns probabilities across the
whole vocabulary. The GPT-2 report used this approach for tasks such as
[[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language#Translation and answers through prefixes|translation from a prefix]].
[Zhao et al., §2 and Appendix A, Table 4](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=2)
later tested GPT-2 on sentiment classification.

One way to measure what the prefix itself does is to keep the model
and the task fixed. In [Table 4 of the GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=6),
the authors generate text after a news article with and without
`TL;DR:`. Their procedure samples 100 tokens, choosing between
the two most probable tokens at each step, and takes the first
three generated sentences as the summary. Removing the cue drops
ROUGE-2 F1 from 8.27 to 4.03. The metric compares adjacent word pairs
with the reference summary; it does not check whether the summary
preserves the article's facts.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt2-2026-10-22/gpt2-output-projection-2.png|GPT-2 compares a 768-dimensional state with each of the 50,257 rows in its token embedding table.]]

*Jay Alammar, [The Illustrated GPT-2, “Model Output”](https://jalammar.github.io/illustrated-gpt2/#model-output), [original PNG](https://jalammar.github.io/images/gpt2/gpt2-output-projection-2.png), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Unmodified figure.*

Alammar shows the first position, where generation starts. In the review prompt we take the state of the final token in `Sentiment:`. For small GPT-2 this is a vector of 768 numbers. Its dot product with each row of the table gives one of 50,257 logits. We compare two, for ` positive` and ` negative`. The others enter the softmax denominator but cannot win when the choice is restricted to those two labels.

<a id="gpt-route-gpt3"></a>

#### GPT-3 (2020): examples in the prompt

Keep the same review at the end, but place
two labeled examples before it: `Well acted.` with `positive` and
`Too slow.` with `negative`. The decoder reads the labels before the
final `Sentiment:`; there is no optimizer step. Keep the same instruction
at the start in all three settings, for example
`Classify the review as positive or negative.` Keep one labeled review
for one-shot and remove both for zero-shot. GPT-2 had used examples in some
experiments. The [GPT-3 paper, §2 and Figure 2.1](https://arxiv.org/html/2005.14165v4#S2.F1)
systematically varied their number with the weights held fixed.
The [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#Translation: direction and parallel examples|worked translation example]]
shows the same three settings.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-13/lu22-figure2-prompt-order.png|Three versions of one prompt. Four labeled examples change order and the test input stays last. The same language model gives a separate prediction for each prompt.]]

*Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp,
[Fantastically Ordered Prompts and Where to Find Them, §2](https://arxiv.org/html/2104.08786v2#S2),
[Figure 2, p. 2 of v2](https://arxiv.org/pdf/2104.08786v2#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure from the paper, not redrawn.*

In the figure, `Train 1`–`Train 4` are four labeled reviews written
into the prompt. Each row on the left is a separate run of the same
model (`PLM`). The first uses the order `1, 2, 3, 4`; the second
uses `3, 4, 2, 1`. The `Test` block always comes last with its label
left blank. The [[#gpt-decoder-route-prompt-order|results below]]
show how the predictions changed when the authors reordered the examples.

In [GPT-3's entailment experiment on ANLI R3](https://arxiv.org/html/2005.14165v4#A8),
the 175B model scored 34.5% on the test set with no examples,
35.1% with one, and 40.2% with fifty. It chose among three labels
written as text ([sample prompt, Appendix G](https://arxiv.org/html/2005.14165v4#A7)).
Adding fifty examples improved accuracy by 5.7 percentage points,
but the score was still close to the 33.3% expected from uniform
guessing. The figure above shows a sentiment prompt; ANLI used a
different template.

<a id="gpt-route-demonstration-controls"></a>

A higher score with demonstrations does not necessarily mean the model
used the connection between each review and its label. Min et al. kept
the demonstration inputs and randomly sampled each label from the allowed
set. In our prompt, `Well acted.` might get `negative`, while
`Too slow.` might keep its original label. The model still sees
sample reviews, the label vocabulary, and the format of a pair, but
the pairings may be wrong. The test review and its reference answer
stay the same.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-continuity-2026-11-10/min-figure1-controls.png|Two panels show macro-F1 for classification and accuracy for multiple choice. In the rightmost group, GPT-3 scores almost as well with random as with gold labels; both beat the no-demonstration baseline.]]

*Sewon Min et al., [Rethinking the Role of Demonstrations, Figure 1, p. 11048](https://aclanthology.org/2022.emnlp-main.759.pdf#page=1),
[§§3–4](https://arxiv.org/html/2202.12837v2#S3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Both original
panels, with their axes, legends, and labels unchanged.*

Compare the three rightmost bars in the upper panel. GPT-3 175B gets
about 41% macro-F1 with no demonstrations, 58% with gold labels, and
55% with random labels. The lower panel reports multiple-choice accuracy
instead. Each panel averages three datasets. Both use the channel
method: score how likely the test text is given each candidate label.
Our `Sentiment:` example goes the other way and scores the label given
the text. With random labels GPT-3 kept most of the gain. A task in
which the labels can only be interpreted from the prompt needs a
different test.

<a id="gpt-route-llama"></a>

#### LLaMA v1 (2023): the same prompt format, a different decoder

Base LLaMA also takes labeled examples in the prompt. Pan et al. asked
whether GPT-3 and LLaMA could infer a new but consistent mapping from
classes to labels. In the middle row of the figure, the positive
example is labeled `0`, so the reference answer for `i liked it`
is also `0`. Swap `0` and `1` in every demonstration and the
reference answer becomes `1`. The weights and test review stay the
same. Unlike the random labels in Min et al., these pairings are
consistent, and knowing what `positive` means is not enough: the
prompt tells the model which digit to use. The final column gives
the reference answer, not a guaranteed prediction.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/pan-icl-figure1.png|One review in three prompts: random word labels, consistent numeric labels, and correct word labels. The last column shows the reference answer.]]

*Jane Pan, Tianyu Gao, Howard Chen, and Danqi Chen, [What In-Context Learning “Learns” In-Context, Figure 1](https://aclanthology.org/2023.findings-acl.527.pdf#page=1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original figure from the paper, not redrawn. See the [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-label-mapping|experimental setup and results]].*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/pan-icl-figure2.png|Bottom panels for GPT-3 and LLaMA: accuracy rises from 8 to 32 examples with numeric labels, but not with random word labels.]]

*Pan et al., [Figure 2, p. 8300](https://aclanthology.org/2023.findings-acl.527.pdf#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Six original panels with their shared legend.*

In the bottom row, each model stays fixed as the number of examples
changes. On the left, GPT-3 175B goes from about 45% accuracy at 8
examples to 60% at 32 with numeric labels (`Abstract`). In the middle,
LLaMA-65B goes from about 48% to 57%. Neither model improves
with random labels (`Random`). The points average over
16 classification datasets. Compare the change within each panel:
the authors used different evaluation sample sizes for GPT-3 and
LLaMA. The two gains are consistent with extra examples helping
the models infer which digit goes with which class. The experiment
does not isolate the effect of RoPE or SwiGLU.

This prompt format survives the change in architecture. GPT-3 and
LLaMA v1 both have a 2,048-token window, but the original LLaMA
returns to full causal attention in every layer. It changes the
normalization, positional encoding, and FFN. The diagrams below
trace those changes; the [[#gpt-decoder-route-training-budget|compute comparison]]
shows why the authors trained smaller models for longer.

<a id="gpt-decoder-route-architecture"></a>

### What changed inside the decoder

| Model / context | Change inside the block |
| --- | --- |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение#gpt1-decoder-architecture|GPT-1]]<br>512 | Full causal attention, learned absolute positions, LayerNorm after residual addition. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-pre-norm|GPT-2]]<br>1,024 | LayerNorm moves to the input of attention and the FFN. A separate LayerNorm follows the stack. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-full-local-attention|GPT-3]]<br>2,048 | Full and local attention layers alternate. A local layer masks some of the distant tokens on the left. |
| [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-comparison|LLaMA v1]]<br>2,048 | Full causal attention in every layer again. RMSNorm replaces LayerNorm, RoPE replaces the position table, and SwiGLU replaces the GELU FFN. |

In Xiong et al.’s figure, the gray arrow runs straight from the input up to the addition in both panels. In GPT-1’s ordering (left), the sum still goes through LayerNorm. In GPT-2’s ordering (right), the sum goes straight to the next sublayer, and LayerNorm is on the input to attention. GPT-3 kept this ordering. LLaMA changed the normalization operation itself.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-pre-post-norm.png|Left: LayerNorm after addition. Right: LayerNorm on the attention and FFN branches before addition.]]

*Ruibin Xiong et al., [On Layer Normalization in the Transformer Architecture, Figure 1, p. 2](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Both panels and the original caption are from the paper. The diagram compares the placement of normalization rather than the full GPT architecture.*

GPT-3 changed which tokens a single attention layer could read. Find
the `on` row in the two matrices below. The left matrix has four ones:
the query can attend to `The`, `cat`, `sat`, and itself. In the middle
matrix there are three. The window in the drawing holds three positions
including the current one, so it excludes `The`. The final `the` is
to the right of the query and is masked in both cases. A one marks an
allowed connection, not the weight attention assigns to it.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/long-context-review-2026-09/mistral-sliding-window.png|In the on row, the full mask allows four positions. The local mask allows three and excludes The. The right panel shows information passing through several local layers.]]

*Albert Q. Jiang et al., [Mistral 7B, Figure 1](https://arxiv.org/html/2310.06825v1#S2.F1),
[original PNG](https://arxiv.org/html/2310.06825v1/swa.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), unmodified.
The three-position window illustrates Mistral's masking; the GPT-3 paper does not specify its window width.*

[GPT-3, §2.1](https://arxiv.org/html/2005.14165v4#S2.SS1) alternates
full and local masks. If `cat` read `The` in a full-attention layer,
`on` can receive information about `The` through `cat` in the next
local layer even though it cannot attend to `The` directly. In the
first LLaMA, the direct connection is available in every layer, as in
the left matrix. Both models still have a 2,048-token context. The
context length sets how many tokens fit in the input; the mask sets
which of them a position can attend to in one layer.

<a id="gpt-decoder-route-attention-pairs"></a>

The same figure lets us count allowed query–key pairs across a layer.
For five tokens, the full mask allows $1+2+3+4+5=15$ pairs per head.
With a three-position window, the local mask allows $1+2+3+3+3=12$.
At the model's full context length $T$, the full mask gives $T(T+1)/2$:
$131\,328$ pairs at $T=512$ for GPT-1 and $524\,800$ at $T=1024$ for
GPT-2. Doubling the input length almost quadruples the pair count.
A full layer in GPT-3 or LLaMA allows $2\,098\,176$ pairs at $T=2048$.
A local GPT-3 layer allows fewer, but the report does not give the band
width needed to count them exactly. These are allowed pairs across a
pass over the whole input, not a measure of speed or matrix storage.

In the Stanford CS336 diagram, the token state enters at the bottom.
One branch goes directly to the first `Add`. The other goes through
`Norm` and attention. GPT-2 already used this order. In the first LLaMA,
`Norm` stands for RMSNorm, which rescales the vector without subtracting
its mean. Inside the attention sublayer, the label `w/ RoPE` refers to
how the query and key are computed.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/cs336-prenorm-block.png|The lower branch has normalization and causal attention with RoPE. The upper branch has normalization and a position-wise FFN. Each branch has its own residual addition.]]

*Stanford CS336 Staff, [Assignment 1: Basics, Figure 2, p. 13](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=13),
“A pre-norm Transformer block,” [MIT](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/LICENSE).
Original diagram from the assignment.*

<a id="gpt-decoder-route-positions"></a>

The [GPT-2 implementation](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L147-L165)
adds row $p_i$ of the learned table `wpe` to the token embedding at
position $i$ before the first block. In GPT-2 Small that table
contains $1024\times768=786\,432$ parameters. Move two tokens from
positions $(0,1)$ to $(3,4)$ and the block receives $p_3$ and $p_4$
in place of $p_0$ and $p_1$. The table does not constrain those pairs
to encode the same distance.

[The first LLaMA](https://arxiv.org/html/2302.13971v1#S2.SS2) removes
the table. Each layer computes a query $q$ and key $k$, then rotates
their coordinates in pairs. The top panel in the RoFormer figure
shows one pair: it rotates $(x_1,x_2)$ by $m\theta_1$, where $m$ is
the position and $\theta_1$ is the fixed frequency for that pair.
The bottom panel repeats the operation across positions and
coordinate pairs.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png|Top: a query or key coordinate pair rotates by mθ₁. Bottom: the operation repeats for each pair with its own frequency.]]

*Jianlin Su et al., [RoFormer, Figure 1 and §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2),
[original PDF page](https://arxiv.org/pdf/2104.09864v5#page=5),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Complete figure and original caption, unmodified.*

Take one coordinate pair with $q=k=(1,0)$ before rotation and
$\theta_1=1$ radian per position. Put the key at 0 and the query at 1,
so their rotation angles are 0 and 1. The dot product of these unit
vectors is the cosine of the angle between them, $\cos(1)\approx0.5403$.
At positions $(3,4)$ both vectors have turned by three more radians,
so the angle between them is still one radian. Keep the key at 0
and move the query to 2, and the dot product becomes
$\cos(2)\approx-0.4161$. This is one pair's contribution before
scaling and softmax. Adding a prefix can change $q$ and $k$ themselves
as well as the set of available keys, so the model need not give the
same answer. The [LLaMA implementation](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L115-L143)
rotates queries and keys before comparing them; it leaves values $V$ unrotated.

The upper branch (`Position-Wise Feed-Forward`) processes each token separately. In the left panel below, the input goes through $W$, GELU, and the output matrix $W_o$. In the center, the same input goes through two projections. One passes through SiLU; the two results are multiplied element by element, and $W_o$ projects back to the decoder width. This is where the third matrix in LLaMA's FFN comes from.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/tajima-figure1-ffn.svg|Left: a GELU FFN with two matrices. Center: SwiGLU with two input projections, Swish on one of them, element-wise multiplication, and an output matrix. Right: the later SwiMGLU variant.]]

*Yukito Tajima, Nakamasa Inoue, Yusuke Sekikawa, Ikuro Sato, and Rio Yokota, [Masked Gated Linear Unit, Figure 1](https://arxiv.org/html/2506.23225v1#S1.F1), [original SVG](https://arxiv.org/html/2506.23225v1/fig_swimlu_v2.svg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Unmodified. The right panel shows the authors' later method, which LLaMA does not use.*

[GPT-3 6.7B](https://arxiv.org/html/2005.14165v4#S2.T1) and [LLaMA-7B](https://arxiv.org/html/2302.13971v1#S2.T2) both have a decoder width of 4,096. GPT-3 expands to 16,384 coordinates, whereas LLaMA uses 11,008 in each of its two branches. The two GPT-3 matrices contain about 134.2 million weights, and the three LLaMA matrices contain about 135.3 million, a difference of less than 1% per FFN excluding biases. The [[en/00 Textbook/07 Anatomy of a modern LLM/01 LLaMA as a baseline architecture#gpt-llama-ffn|LLaMA chapter]] derives the width from the matrix count and shows how the original code rounds it.

Even the same 2,048-position context can hold different amounts of text:
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-tokenization|one string takes different numbers of tokens in different tokenizers]]. For a model with separate encoder and decoder stacks, see [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5]].

<a id="gpt-decoder-route-depth-width"></a>

### Where the billions of parameters come from

In Alammar's figure, the number next to the top block gives the depth:
12 blocks for GPT-2 Small and 48 for Extra Large. The number under each
stack is the width of a single token state, 768 or 1,600. Both models
accept up to 1,024 tokens. A taller stack takes the state through more
blocks; it does not let the model read more text.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-sizes-hyperparameters-3.png|Four GPT-2 stacks with 12, 24, 36, and 48 blocks. The token-state widths underneath are 768, 1,024, 1,280, and 1,600.]]

*Jay Alammar, [The Illustrated GPT-2, “Transformers for Language Modeling”](https://jalammar.github.io/illustrated-gpt2/#transformers-for-language-modeling), [original PNG](https://jalammar.github.io/images/gpt2/gpt2-sizes-hyperparameters-3.png), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Unmodified figure.*

Let $d$ be the state width. A GPT-2 block uses three projections for
$Q$, $K$, and $V$, plus the attention output projection, for $4d^2$
weights. The FFN projects from $d$ to $4d$ and back, adding $8d^2$.
Counting only those matrices gives $12Ld^2$ weights for $L$ blocks.
Extra Large has four times as many blocks as Small, and its matrices
grow along both dimensions. That takes the stack from about 85 million
weights to 1.47 billion, a factor of 17.4. The full parameter count
grows by a smaller factor, about 12.5: the token table grows only with
$d$ because the vocabulary size stays fixed.
The [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language#What changed from GPT-1|full GPT-2 count]]
also includes positions, biases, and normalization.

GPT-1, GPT-2 Small, and GPT-3 Small all have 12 blocks of width 768.
Most of the difference between GPT-1 and GPT-2 Small comes from
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-parameter-budget|the larger vocabulary]].
GPT-3 175B increases the stack to 96 blocks of width 12,288.
The stack matrices alone account for about 174 billion weights.
The local mask changes which queries can see which keys but does not
change the size of the matrices. Both GPT-3 models have the same
2,048-token window ([§2.1 and Table 2.1](https://arxiv.org/html/2005.14165v4#S2.SS1)).
The LLaMA-7B FFN has three matrices instead of two, with an inner
width of 11,008. Together with the attention projections they give
$4d^2+3d\cdot11008$ weights per block, about 202 million at $d=4096$
([[en/00 Textbook/07 Anatomy of a modern LLM/01 LLaMA as a baseline architecture#Shapes and parameters of the original models|LLaMA parameter count]]).

<a id="gpt-decoder-route-cache"></a>

### What grows in memory during generation

In the Hugging Face figure, the model has already produced `will` after `I am sure this project`. To predict `be`, it computes a query for `will` alone. The keys and values for the seven earlier positions, including the beginning-of-sequence token, are already cached. The model adds an eighth key/value pair for `will`. The head has one query but eight K/V pairs. The next step needs a query for `be`, so there is no reason to store the query for `will`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/hf-kv-cache.png|The step for will uses one query, seven cached key/value pairs, and one new key/value pair. The output token is be.]]

*Rémi Ouazan Reboul, Arthur Zucker, and Luc Georges, [Continuous batching, KV-cache](https://huggingface.co/blog/continuous_batching#kv-cache), [original PNG](https://huggingface.co/datasets/huggingface/documentation-images/resolve/c8c3cf72b3b724ce734259e4a086d30618a9198b/blog/continuous_batching/kv_cache.png), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Unmodified. $A$ is the width of one head.*

The released [GPT-2 implementation](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L69-L112) concatenates the new K/V with the states for earlier positions. The [first LLaMA implementation](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L115-L143) writes them into a preallocated buffer. LLaMA stores the key after applying RoPE; rotation does not change its width.

For one position in one layer, the cache stores $H_{KV}$ keys and the same number of values, each with $A$ entries. At $b$ bytes per entry, $L$ layers and $T$ cached positions take $2LH_{KV}ATb$ bytes. LLaMA-7B has 32 layers with 32 heads of width 128. At two bytes per entry, that gives $2\cdot32\cdot32\cdot128\cdot2=524\,288$ bytes per token, or 512 KiB. A 2,048-position cache therefore holds exactly 1 GiB.

| Model | K/V per token (16-bit) | K/V at full context |
| --- | ---: | ---: |
| GPT-1 (12 layers, 12 heads of width 64) | 36 KiB | 18 MiB / 512 tokens |
| GPT-2 Small (12 layers, 12 heads of width 64) | 36 KiB | 36 MiB / 1,024 tokens |
| GPT-3 6.7B (32 layers, 32 heads of width 128) | 512 KiB | 1 GiB / 2,048 tokens |
| LLaMA v1 7B (32 layers, 32 heads of width 128) | 512 KiB | 1 GiB / 2,048 tokens |

Configurations: [GPT-1, §4.1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=5), [GPT-2, Table 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4), [GPT-3, Table 2.1](https://arxiv.org/html/2005.14165v4#S2.T1), [LLaMA, Table 2](https://arxiv.org/html/2302.13971v1#S2.T2). The table counts only the K/V data for one sequence if every layer retains every position. It is an estimate from the configurations, not a memory measurement of the original implementations. It excludes weights, temporary buffers, and reservations for other requests.

From GPT-1 to GPT-2 Small, the context length doubled but the byte count per token would stay the same. The GPT-3 and LLaMA models in the table have the same context length and K/V shapes. A local GPT-3 layer could nevertheless discard older positions outside its band, since its next query cannot use their keys. A full layer cannot do that. The GPT-3 paper gives neither the band width nor its serving cache layout, so the 1 GiB figure assumes full retention. The first LLaMA uses full attention. A later model, [Llama 2 70B](https://arxiv.org/abs/2307.09288), reduces the number of K/V heads through [[00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|grouped-query attention (GQA)]].

<a id="gpt-decoder-route-data"></a>

### From books to training mixtures

GPT-1 and GPT-2 used the same loss to learn to continue text. In a novel,
that often means carrying on a story. On a web page, it can mean answering
a question or giving the translation that follows a phrase. The GPT-2
authors tested whether this kind of pretraining would let the model solve
tasks from a text prefix alone. For GPT-3 and LLaMA, the researchers set
each source's share of the training mixture independently of its size.

| Model | What it read during training |
| --- | --- |
| [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/04 GPT-1 — generative pre-training#Stage one: what BooksCorpus teaches|GPT-1]] | The authors sampled contiguous 512-token sequences from BooksCorpus. Neighboring paragraphs from a novel could fit into one sequence; the whole novel could not. [Paper, §4.1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=5). |
| [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language#WebText and byte-level BPE|GPT-2]] | WebText was built from outbound links on Reddit. After cleaning, the version used in the paper contained just over 8 million documents, around 40 GB of text. [Paper, §2.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3). |
| [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#Data and sampling|GPT-3]] | The mixture combined WebText2 with filtered Common Crawl, two book corpora, and English Wikipedia. Over 300 billion training tokens, the authors report 0.44 passes through Common Crawl and 2.9 through WebText2. [Paper, Table 2.2](https://arxiv.org/html/2005.14165v4#S2.T2). |
| [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#llama-pretraining-data|LLaMA v1]] | CCNet-filtered English Common Crawl made up 67% of the mixture and C4 another 15%. The rest came from GitHub, Wikipedia, books, arXiv, and Stack Exchange. The 7B and 13B models each processed one trillion tokens; 33B and 65B each processed 1.4 trillion. [Paper, Table 1 and Figure 1](https://arxiv.org/html/2302.13971v1#S2). |

WebText's 40 GB measures the collected text. GPT-3's 300 billion counts
tokens seen during training, including repeats: a passage counts every
time it goes through the model. The 2,048 positions measure what fits
in a single pass. GPT-3 13B and LLaMA 13B therefore had the same context
length even though LLaMA processed about 3.3 times as many tokens over
pretraining. They used different tokenizers, so the ratio does not tell
us how many words each model read.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/llama1-training-loss.png|LLaMA training-loss curves plotted against billions of tokens processed. The 7B and 13B curves end at 1,000; 33B and 65B end at 1,400.]]

*Hugo Touvron et al., [LLaMA, Figure 1, p. 3](https://arxiv.org/pdf/2302.13971v1#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original plot from the paper.*

Find 300 billion, halfway between 200 and 400, on the orange 13B curve.
Follow that curve right to one trillion: the next-token prediction loss
falls while the number of weights stays fixed. GPT-3 does not appear
in this figure. Following one curve tells us what happened as LLaMA
trained longer, not how the two families compare.

<a id="gpt-decoder-route-training-budget"></a>

#### GPT-3, Chinchilla, and LLaMA: what stayed fixed

All eight GPT-3 models processed 300 billion training tokens.
Going from 13B to 175B therefore increased both the weight count and
the cost of processing those 300 billion tokens. The series compares
sizes at the same training length; it cannot tell us which size is
best for a fixed number of operations.

[Hoffmann et al., §3.2](https://arxiv.org/html/2203.15556v1#S3.SS2)
ran the complementary experiment. They trained several model sizes
at the same compute budget and adjusted the number of tokens for each.
If $C$ operations are available for $N$ parameters and $D$ tokens,
then $C\approx6ND$ means doubling $N$ leaves room for half as many
tokens. At $C=6\cdot10^{19}$, a 1-billion-parameter model can process
about 10 billion tokens; a 2-billion-parameter model can process
about 5 billion. To choose between them, compare the final losses.
The equation sets the budget and does not predict the loss.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scaling-isoflop-2026-11-09/training_flop_vs_model_size.png|Training operations on the horizontal axis, parameters on the vertical axis. Near 10¹⁹ FLOP, the upper points have higher loss than some of the lower ones.]]

*Tamay Besiroglu, Ege Erdil, Matthew Barnett, and Josh You,
[Chinchilla Scaling: A replication attempt, Figure 2](https://arxiv.org/html/2404.10102v2#S2.F2),
[original PNG](https://arxiv.org/html/2404.10102v2/training_flop_vs_model_size.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The replication
authors' figure, unmodified. They recovered the points from Figure 4
of Hoffmann et al., not from the original training logs.*

To compare runs at one budget, read vertically. Around $10^{19}$ FLOP,
the light dots near the top show higher loss for some larger models.
The replication authors flagged several of those points as outliers,
so that column alone is a poor way to choose a model size.
Hoffmann et al. tested the approach at a larger scale by training
[Chinchilla](https://arxiv.org/html/2203.15556v1#S4) on 1.4 trillion
tokens at roughly Gopher's training budget. Chinchilla had 70 billion
weights, versus Gopher's 280 billion. Its LAMBADA accuracy rose from
74.5% to 77.4%. The authors also changed the optimizer, tokenizer,
and the proportions of sources in the training mixture.

The [LLaMA authors](https://arxiv.org/html/2302.13971v1#S1) chose model
sizes with inference cost in mind. If a service can run a 13B model,
training it longer does not increase the number of weights it has to
load. In the next figure, each point on the orange LLaMA 13B curve
is a checkpoint saved after a different number of training tokens.
Here the authors measure task accuracy rather than the loss on
text continuation.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/llama1-downstream-training.png|Accuracy on six tasks throughout LLaMA pretraining. For 13B, HellaSwag rises from about 71% at 250 billion tokens to 79% at one trillion, while SIQA fluctuates around 50%.]]

*Hugo Touvron et al., [LLaMA, §3.7, Figure 2](https://arxiv.org/html/2302.13971v1#S3.SS7), [PDF, p. 8](https://arxiv.org/pdf/2302.13971v1#page=8), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All six panels and the caption are from the paper. Open the PDF to read the axes at full size.*

In the top middle panel (HellaSwag), LLaMA 13B scores about 71% after 250 billion tokens and 79% after one trillion. In the bottom left panel (SIQA), its curve fluctuates around 50%, without a comparable gain. The horizontal dashed line is Chinchilla's score, not a series of GPT-3 measurements. The experiment compares earlier and later checkpoints of one LLaMA model. Comparing it with GPT-3 calls for scores on the same task and attention to differences in the evaluation protocol.

<a id="gpt-decoder-route-repeated-data"></a>

#### New tokens or another pass through the corpus

GPT-3's authors report 2.9 passes through WebText2 and 0.44 through
Common Crawl. For LLaMA at 1.4 trillion training tokens, the Common
Crawl count reached 1.10 passes and Wikipedia reached 2.45. These are
averages for each source, not the number of times any one document
appeared. Total tokens alone cannot tell us how often a model read
the same text. Muennighoff et al. ran a separate experiment to measure
the effect of repetition.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-13/muennighoff-jmlr-figure10.png|Left: measured loss at fixed training length. For the 8.7B model, the 25% unique-token run is close to the 100% run. Right: predicted loss for longer training on a corpus of fixed size.]]

*Niklas Muennighoff et al., [Scaling Data-Constrained Language Models, JMLR 26(53), 2025](https://jmlr.org/papers/v26/24-1000.html),
[§6, Figure 10, p. 14](https://jmlr.org/papers/volume26/24-1000/24-1000.pdf#page=14),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Both panels and the legend are from the paper, not redrawn.*

Follow the bottom row of dots on the left. Each model has 8.7 billion
parameters and processed about 178 billion C4 tokens. At 100%, the
model makes one pass over a large corpus. At 25%, it makes four passes
over about 44 billion tokens. The final loss on held-out text was
only 0.5% higher than for one pass. The solid curves on the left are
fits to the experiment. The solid curves on the right are predictions
for a fixed corpus and more training tokens from additional passes.
The experiment repeats the whole C4 corpus; it does not measure the
effect of repeating Wikipedia in LLaMA's mixture.

<a id="gpt-decoder-route-evidence"></a>

### What the experiments show

GPT-1's authors checked the value of pretraining by training the
same Transformer on the same labeled examples from random weights.
On SST-2, the sentiment classification task, accuracy was 91.3%
with pretraining and 84.0% without it
([Table 5, §5](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=8)).
Both runs trained on labeled reviews. The table also reports mean
scores of 74.7 and 59.9 across eight tasks that use different
metrics: accuracy, F1, and correlation.

LAMBADA asks the model to predict the last word of a passage. The weights
can stay fixed while we change the prompt or the rule for choosing the
word. GPT-2 scored
52.66% without filtering and 63.24% with a stop-word filter
([§3.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=5)).
The [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/05 GPT-2 — zero-shot through language#gpt2-gpt3-lambada|GPT-2 example]]
shows which candidate the filter removes and why restricting answers
to words from the passage is a different rule.
GPT-3 175B scored 76.2% with plain continuation and 86.4% with
15 demonstrations. In the latter setting the authors also changed
the format, from completing a passage to
[[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#lambada-gpt3-fill-blank|filling in a blank]].
One demonstration in the new format scored 72.5%, below plain
continuation. The gap between 86.4% and 76.2% therefore includes the
effect of changing the format
([GPT-3, §3.1.2, Table 3.2 and Figure 3.2](https://arxiv.org/html/2005.14165v4#S3.SS1.SSS2)).

[LLaMA, Table 3](https://arxiv.org/html/2302.13971v1#S2.T3) compares
GPT-3 175B and LLaMA 13B without demonstrations. On HellaSwag, where
the model chooses a plausible continuation, GPT-3 gets 78.9% and
LLaMA gets 79.2%. On PIQA, where it chooses a solution to an everyday
physical problem, the order reverses: 81.0% and 80.1%. These are
reported results, not an experiment that varies only model size.
Even the [[en/00 Textbook/06 Encoder Decoder and Encoder-Decoder/06 GPT-3 — in-context learning#gpt-llama-completion-scoring|rule for ranking HellaSwag endings]]
differs: GPT-3 normalizes scores by tokens, LLaMA by characters.
The 0.3-point gap does not isolate the effect of RoPE.

The models differ in size by about 13.5 times. The rough pretraining
compute estimate $6ND$ accounts for both the number of weights $N$
and the number of tokens processed $D$. For GPT-3 175B and 300 billion
tokens, that gives about $3.15\cdot10^{23}$ FLOP. For LLaMA 13B
and one trillion tokens, it gives $7.8\cdot10^{22}$ FLOP. The gap
is about fourfold rather than 13.5-fold because the smaller model
processed more tokens. These estimates do not measure GPU time.
The [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#training-and-inference-budget|training and inference budget example]]
shows when the extra training of a smaller model can pay for itself at inference.

<a id="gpt-decoder-route-bpb"></a>

#### When perplexities are not directly comparable

GPT-3 and the original LLaMA have the same context length but
[[en/00 Textbook/07 Anatomy of a modern LLM/01 LLaMA as a baseline architecture#gpt-llama-tokenization|different tokenizers]].
So “3 nats per token” can describe different losses on the same text.
The denominator changed: a token may cover more of the original bytes.

Take four UTF-8 bytes: `cats`. Suppose the first tokenizer splits
the string into `cat` and `s`. The model assigns probability 1/2
to the first token and 1/4 to the second given the first. The product
is $1/8$, a loss of 3 bits for the string. A second toy model splits
it into four tokens, one per byte, with conditional probabilities
1/2, 1/2, 4/5, and 5/8. Their product is also $1/8$. The perplexities
differ: $2^{3/2}\approx2.83$ for the first model and
$2^{3/4}\approx1.68$ for the second. Both use $3/4=0.75$ bits per byte.
These probabilities are illustrative, not measurements from GPT or LLaMA.

For a corpus, sum the negative log conditional probabilities of the
scored tokens and divide by the number of bytes in the original text:

$$
\begin{aligned}
\mathrm{BPB}&=\frac{\sum_{t=1}^{T}-\log_2 p(x_t\mid x_{<t})}{B}\\
&=\frac{T}{B}\log_2(\mathrm{PPL}).
\end{aligned}
$$

Here $T$ is the number of scored tokens and $B$ is the number of
bytes in the corresponding UTF-8 text. The sum uses the single
segmentation returned by the tokenizer. In the GPT-2 report, Radford
et al. use the same procedure with the canonical unit of each
benchmark: bytes, characters, or words
([§3.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4)).
[Gao et al., §§3.1–3.2](https://arxiv.org/html/2101.00027v1#S3)
chose bytes and applied the metric to GPT-2 and GPT-3.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-14/pile-gpt2-gpt3-figure2.png|Eight GPT-2 and GPT-3 points on The Pile. Parameters are on the horizontal logarithmic axis and bits per byte on the vertical axis. The leftmost point is about 1.23 and the rightmost about 0.72.]]

*Leo Gao et al., [The Pile, Figure 2](https://arxiv.org/html/2101.00027v1#S3.F2),
[original PNG](https://arxiv.org/html/2101.00027v1/gpt3_zeroshot_pile_scalinglaw.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Unmodified figure.*

Each point is one model. The endpoints are GPT-2 Small at 1.2253
bits per byte and GPT-3 Davinci at 0.7177
([Table 2](https://arxiv.org/html/2101.00027v1#S3.T2)).
Over 1,000 bytes, that difference would amount to about 508 bits
of negative log probability. The plot compares prediction on
documents from The Pile, not the accuracy of answers to questions.
The authors scored each document separately; the GPT-3 sizes on
the axis are their assumed mapping from API names to paper models.
There is no LLaMA point. To add one, we would need to score the
same text with consistent rules for context, document boundaries,
and which tokens enter the sum. BPB supplies a common unit,
but it does not resolve those protocol differences.

<a id="gpt-decoder-route-prompt-order"></a>

#### What changes when the examples change order

Lu et al. fixed four labeled examples and tried all $4!=24$
permutations. For each order, they placed the same four demonstrations
before each new text and measured classification accuracy. The model
weights, examples, and template stayed fixed. The four positions on
the left of each plot are GPT-2 models; the other four are GPT-3 models.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-11/lu22-figure1.png|Box plots of accuracy across four-example prompt orders. GPT-2 1.5B varies widely on SST-2; GPT-3 175B varies little on SST-2 but still varies on Subj.]]

*Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp,
[Fantastically Ordered Prompts and Where to Find Them, §2](https://aclanthology.org/2022.acl-long.556/),
[Figure 1, p. 1 of v2](https://arxiv.org/pdf/2104.08786v2#page=1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Both plots are from the paper, not redrawn.*

On the top plot, for SST-2 sentiment, the whiskers for GPT-2 1.5B run
from about 48% to 87% accuracy; for GPT-3 175B, from about 92% to 95%.
The line in each box is the median across orders, not an average over
different sets of examples. The bottom plot measures how well the
same models distinguish subjective statements from objective ones.
For GPT-3 175B the whiskers there span about 56–85%. GPT-3 175B was less sensitive to order than GPT-2 1.5B on SST-2,
but still showed a wide spread on Subj.

Accuracy alone does not tell us which label became more common.
In a separate experiment, Zhao et al. gave GPT-3 2.7B four labeled reviews before each test
review. On the chart, `P` and `N` are the four demonstration labels
in the order the model reads them. Compare `N N P P` and `P P N N` in
the middle group. Both prompts have two examples of each class.
With the positive examples last, the model answers `Positive` for
about three quarters of the test reviews. With the negative examples
last, the share drops to about one tenth. A bar shows the fraction of
SST-2 validation reviews classified as `Positive`, not accuracy or
the model's confidence on a single review.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/zhao21c-figure4.png|GPT-3 2.7B answers Positive more often for N N P P than for P P N N, although both prompts have two labels of each class.]]

*Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh,
[Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://proceedings.mlr.press/v139/zhao21c.html),
ICML 2021, [Figure 4, §4](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original plot from the paper.*

<a id="gpt-decoder-route-calibration"></a>

#### Correcting the prompt's preference for a label

Each dot in the next figure is one of 25 randomly sampled test reviews.
Its position is the score for `Positive`; its color is the true class,
red for negative and green for positive. The vertical line is at 0.5.
Red dots to its right are false positives when the higher-scoring label wins.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/zhao21c-figure5.png|Positive scores for 25 reviews. Several negative reviews lie above the 0.5 threshold and are misclassified as positive.]]

*Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh,
[Calibrate Before Use, ICML 2021, PMLR 139](https://proceedings.mlr.press/v139/zhao21c.html),
[Figure 5, PDF p. 5](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original plot, with its labels unchanged.*

For this prompt the authors report 67% accuracy at the 0.5 threshold
and 94% at 0.68, a threshold selected using validation labels.
Without those labels, we need another way to set the threshold.

To measure the preference, keep the examples, their order, and the
template fixed. In the prompt above, `Review: i liked it` becomes
`Review: N/A`, and `Sentiment:` is still blank. In a separate two-shot
example in [§5](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=5),
GPT-3 assigned `Positive` 61.8% of the probability after normalizing over
the two labels. This is a measurement on `N/A`, not a threshold fitted
to the plot. Change the examples or their order and measure the baseline again.

Let $p_y$ be the share of probability for label $y$ among the allowed
labels on the real review, and $b_y>0$ its share on `N/A`. For
classification, contextual calibration ranks the ratios $p_y/b_y$.
On `N/A` itself they all equal one. Suppose a real review gets scores
of 0.60 and 0.40. It would be classified as `Positive` before the
correction. With baseline scores of 0.618 and 0.382, the ratios are
$0.60/0.618\approx0.971$ and $0.40/0.382\approx1.047$, so `Negative`
wins. For two classes this raises the threshold from 0.5 to 0.618:
$p_+/b_+>(1-p_+)/(1-b_+)$ holds exactly when $p_+>b_+$.

In their experiments the authors averaged the baseline over `N/A`,
`[MASK]`, and the empty string. With four demonstrations, mean SST-2
accuracy went from 64.9% to 73.8% for GPT-2 XL and from 59.1% to 79.9%
for GPT-3 2.7B. With eight demonstrations, GPT-3 175B slipped from
95.6% to 95.3%. Each comparison held the weights, template, and
sets of examples fixed
([Table 1](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=7),
[Appendix Table 4, p. 12](https://arxiv.org/pdf/2102.09690v2#page=12)).
The correction changes the label decision. It does not tell us
how often predictions with a score of 0.8 are correct.

<a id="gpt-decoder-route-end"></a>

[[#gpt-decoder-route|Back to the four chapters]] · [[#curriculum|Core course]]
