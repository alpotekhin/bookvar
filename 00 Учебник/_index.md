---
title: Машинное обучение и языковые модели
type: textbook-chapter
status: active
last_updated: 2026-10-29
---

# Машинное обучение и языковые модели

Для чтения с нуля начните с
[[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|векторов, матриц и тензоров]].
Другие маршруты описаны на странице
[[02 Areas/ML & DL/00 Учебник/00 Как пользоваться учебником|«Как пользоваться учебником»]].

Оригинальные лекции, книги, слайды и задания собраны в
[[05 Источники/Курсы|библиотеке курсов]]; ссылки на конкретные материалы
приведены рядом с соответствующими объяснениями.

<a id="полный-маршрут"></a>

## Основной курс

Курс проходит от математических оснований и классических моделей к нейронным сетям, Transformer и обучению языковой модели. Углублённые модули ниже можно читать отдельно после указанных предпосылок.


### I. Математические и ML-основания

- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|Векторы, матрицы и тензоры]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/02 Производная и градиент|Производная и градиент]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм|Вероятность, правдоподобие и логарифм]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/04 Функции потерь|Функции потерь]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/05 Train, validation и test|Train, validation и test]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание|Переобучение, регуляризация и оценивание]]


### II. Классическое машинное обучение

- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/00 Карта модуля|Карта модуля и источники]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline|Задача обучения и predictive pipeline]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация|Линейная регрессия и классификация]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/03 Деревья решений|Деревья решений]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting|Bagging, random forest и gradient boosting]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога|Оценивание, кросс-валидация и выбор порога]]

<a id="iv-нейронные-сети"></a>

### III. Нейронные сети

- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|Нейрон и MLP]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации|Функции активации]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|Вычислительный граф и backpropagation]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|SGD, Momentum, Adam и расписания шага]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections|Инициализация, нормализация и residual connections]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация|Dropout и регуляризация]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований|Аугментация данных — Mixup, CutMix и политики преобразований]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|CNN — от свёртки до ResNet]]

<a id="v-текст-до-transformer"></a>

### IV. Текст до Transformer

- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|Представление текста числами]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель|N-граммная языковая модель]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction|Вероятность текста и next-token prediction]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings|Word2Vec, GloVe и распределительные представления]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece и Unigram]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|RNN и BPTT]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU|LSTM и GRU]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора|Seq2Seq и bottleneck фиксированного вектора]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|Attention для перевода]]

<a id="vi-transformer-bert-и-gpt"></a>

### V. Transformer, BERT и GPT

- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-Attention — Q, K, V]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Маскирование, головы и формы тензоров]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|Позиционная информация]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Полный encoder-decoder Transformer]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Три архитектурных паттерна]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa и DeBERTa]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1 — генеративное предобучение]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2 — zero-shot через язык]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3 — in-context learning]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5 — text-to-text Transformer]]

<a id="vii-анатомия-современной-llm"></a>

### VI. Анатомия современной LLM

- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA как базовая архитектура]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block|Современный decoder block]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|Pre-norm, RMSNorm, SwiGLU и residual]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE|RoPE]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating|Dense FFN — расширение размерности и SwiGLU]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|Transformer с нуля — формы, параметры и стоимость]]

<a id="ix-обучение-llm"></a>
<a id="viii-обучение-llm"></a>

### VII. Предобучение языковой модели

После устройства декодера разбираются корпус, обучающая цель и выбор размера модели при ограниченном бюджете.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|Данные и предобучение]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Сбор, очистка и смеси данных]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса|Дедупликация, PII и контроль качества корпуса]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|Целевая функция предобучения]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Scaling laws]]


### VIII. Post-training и рассуждения

Предобученная модель уже предсказывает следующий токен. Здесь меняются обучающие данные, сигнал качества и способ оптимизации поведения.

- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники|Карта post-training и источники]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT и instruction data]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01a Fine-tuning и PEFT — adapters, LoRA, QLoRA и soft prompts|Fine-tuning и PEFT]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data|Preference data]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|Reward modeling]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|Policy gradient и PPO для LLM]]
- [[00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF|Цели alignment, HHH, Constitutional AI и RLAIF]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|DPO]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR и verifiers]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|GRPO и DeepSeek-R1]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation|Reasoning distillation]]
- [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Test-time compute]]

<a id="x-инференс-и-serving"></a>
<a id="ix-инференс-и-serving"></a>

### IX. Генерация и оценивание

Вероятности токенов превращаются в ответ; затем качество модели проверяется на задачах с контролем загрязнения тестовых данных.

- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Декодирование и выбор следующего токена]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей и контаминация]]

## Углублённые модули

В каждом модуле свой порядок глав. В конце модуля переход ведёт обратно к выбору темы, а не в следующий независимый курс.


### Кластеризация, понижение размерности и ядровые методы

Нужны линейная алгебра, вероятность и оценивание. Модуль можно читать после классических алгоритмов; устройство Transformer для него не требуется.

- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения|Кластеризация и её ограничения]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/06a Спектральная кластеризация и графовый Laplacian|Спектральная кластеризация и graph Laplacian]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/07 PCA, t-SNE и UMAP|PCA, t-SNE и UMAP]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/08 Gaussian mixture и EM|Gaussian mixture и EM]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/09 Kernels, SVM и random Fourier features|Kernels, SVM и random Fourier features]]


### Генеративные модели изображений

Начните после CNN и обратного распространения. Последовательность идёт от скрытого представления и VAE к GAN, диффузионной генерации и использованию синтетических изображений.

- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE|Autoencoder и VAE]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/09 GAN и DCGAN — генератор и дискриминатор|GAN и DCGAN — генератор и дискриминатор]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64g Диффузионные и flow-модели изображений|Diffusion, latent diffusion и flow matching]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции|Синтетические изображения и перенос из симуляции]]


### Контрастивное и маскированное предобучение

Нужны нейронные сети, аугментации и self-attention. SimCLR и MoCo учат представления через сравнение примеров, MAE — через восстановление скрытых патчей. MDLM и LLaDA используют маскирование для обучения генеративной модели текста.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41f Контрастивное предобучение — SimCLR и проверка представлений|Контрастивное предобучение: SimCLR]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41g Momentum contrast — MoCo, очередь и key encoder|MoCo: очередь и momentum encoder]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41h Masked autoencoders — MAE и восстановление патчей|MAE: восстановление скрытых патчей]]
- [[00 Учебник/11 Pre-training и Scaling/41j Диффузионные языковые модели — MDLM и LLaDA|Диффузионные языковые модели: MDLM и LLaDA]]


### Отбор данных и частичная разметка

Нужны функции потерь, разделение данных и аугментации. Здесь решается, какие примеры размечать, отбирать или создавать и как проверить пользу такого выбора.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting|Отбор примеров: EL2N, GraNd и forgetting]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|Active learning: запрос меток]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор: k-center, coreset и herding]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|Частичная разметка: pseudo-labeling и FixMatch]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных: стоимость, шарды и проверка качества]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|Синтетические данные и учебные программы]]


### Эффективное внимание, MoE и альтернативы Transformer

Продолжение анатомии LLM: хранение K/V, обработка длинной истории, выбор экспертов и модели с рекуррентным состоянием. Перед MoE нужна глава о dense FFN.

- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|MHA, MQA и GQA]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA и сжатие KV-cache]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание|Длинный контекст — расширение, разреженность и оценивание]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts — routing, capacity и serving]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|Mamba, RWKV, RetNet и гибридные архитектуры]]

<a id="viii-вычислительные-основы-ml-систем"></a>
<a id="vii-вычислительные-основы-ml-систем"></a>

### Вычислительные основы ML-систем

Нужны формы тензоров, матричное умножение и backpropagation. Для экспертной части расчётов в главе об арифметике потребуется устройство MoE.

- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/01 Модель как часть системы|Модель как часть системы]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти|GPU, CUDA и иерархия памяти]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline|Измерение производительности и roofline]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE|Арифметика Transformer и MoE]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision|Численные форматы и mixed precision]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing|Data pipeline, padding и packing]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки|Profiling ML-нагрузки]]
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению|GPU kernels и Triton — от программы к измерению]]


### Распределённое обучение

Начните после GPU, численных форматов и измерения производительности из модуля ML-систем. Для expert parallelism предварительно прочитайте MoE.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision|Distributed training и mixed precision]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP|Processes, collectives и DDP]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44b Gradient checkpointing и offload|Gradient checkpointing и offload]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism|Tensor и sequence parallelism]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44d Pipeline parallelism|Pipeline parallelism]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44e ZeRO, FSDP2, DeviceMesh и DTensor|ZeRO, FSDP2, DeviceMesh и DTensor]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism|Expert и hybrid parallelism]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints|Сеть, хранилище и распределённые checkpoints]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44h Fault tolerance и fleet orchestration|Отказоустойчивость и управление кластером]]


### Инференс и обслуживание запросов

Нужны декодирование, GPU и roofline. Для расчётов KV-кэша используйте MHA/MQA/GQA из модуля эффективных архитектур. Далее идут память, расписание запросов, ядра, сжатие и распределённое обслуживание.

- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|Физика LLM inference — prefill, decode и roofline]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache, пакетирование и PagedAttention]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Scheduling — continuous batching, chunked prefill и prefix caching]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention|FlashAttention]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей|Квантизация языковых моделей]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57a KV-cache compression и offload|KV-cache compression и offload]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57b Сжатие моделей — pruning, distillation и low-rank|Pruning, distillation и low-rank compression]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование|Спекулятивное декодирование]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving|Параллелизм и коллективные операции]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|Раздельное обслуживание prefill и decode]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLO и эксплуатация inference]]
- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58c Queueing и capacity planning|Queueing и capacity planning]]

<a id="xi-эксплуатация-и-оценивание"></a>
<a id="x-эксплуатация-и-оценивание"></a>

### Эксплуатация ML-систем

Нужны разделение данных, оценивание и обучающий конвейер. Модуль посвящён выпуску модели, наблюдению за сервисом, ограничениям устройств и ошибкам на разных группах данных.

- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/01 ML workflow|ML workflow]]
- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/02 MLOps|MLOps — от артефакта до наблюдаемого сервиса]]
- [[02 Areas/ML & DL/00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment|Edge и federated deployment]]
- [[00 Учебник/19 Deployment, Reliability и MLOps/04 Безопасность и приватность ML-систем|Безопасность и приватность ML-систем]]
- [[00 Учебник/19 Deployment, Reliability и MLOps/05 Устойчивость ML-систем|Устойчивость ML-систем]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Несбалансированная классификация|Несбалансированная классификация]]
- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59a Responsible systems|Ответственные ML-системы]]

<a id="xii-retrieval-и-rag"></a>
<a id="xi-retrieval-и-rag"></a>

### Retrieval и RAG

Нужны векторные представления, BERT и оценивание. Последовательно разбираются обучение embeddings, поиск, повторное ранжирование и сборка ответа с найденным контекстом.

- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|Embeddings и metric learning]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid|Retrieval — от BM25 до dense и hybrid]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction|Reranking — cross-encoder и late interaction]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер|RAG — полный конвейер]]

<a id="xiii-мультимодальные-модели"></a>
<a id="xii-мультимодальные-модели"></a>

### Мультимодальные модели

Начните после CNN и Transformer. Модуль проходит путь от зрительного представления к языковому ответу, затем рассматривает документы, видео, аудио и оценивание. Генерация изображений вынесена в отдельный модуль.

- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Мультимодальные модели]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64a Connectors и fusion|Connectors и fusion]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции|Разрешение, tiling и пространственные позиции]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Обучение VLM — alignment, instruction tuning и данные]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding|Документы, OCR и visual grounding]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|Видео, аудио и omni-модели]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|Оценивание, отказы и serving VLM]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|Единая мультимодальная последовательность и Chameleon]]

<a id="xiv-инструменты-и-агенты"></a>
<a id="xiii-инструменты-и-агенты"></a>

### Инструменты и агенты

Нужны генерация, post-training и оценивание языковых моделей. Сначала разбираются вызовы инструментов и управление контекстом, затем память, планирование, приложения и безопасность.

- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля|Карта модуля и источники]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|Tool use — от вызова функции к действию]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|Agent harness и context engineering]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Память, планирование и оркестрация агентов]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|Оценивание агентных систем]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Coding, web и computer-use agents]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Формальные доказательства и математические агенты]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Агенты научного поиска и discovery]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Безопасность агентных систем]]

<a id="iii-рекомендательные-системы"></a>
<a id="xiv-рекомендательные-системы"></a>

### Рекомендательные системы

Для факторизации матрицы достаточно линейной алгебры, функций потерь и разделения данных. Нейронные модели требуют MLP и backpropagation, последовательные Transformer-модели — self-attention.

- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/00 Карта модуля|Карта модуля и полные источники]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/01 Задача рекомендации, данные и pipeline|Задача рекомендации, данные и конвейер]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/02 Collaborative filtering и matrix factorization|Совместная фильтрация и факторизация матрицы]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели|Ранжирование и нейронные модели]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/04 Последовательные рекомендации и признаки|Последовательные рекомендации и признаки]]
- [[02 Areas/ML & DL/00 Учебник/02 Рекомендательные системы/05 Оценивание и эксплуатация|Оценивание и эксплуатация]]

<a id="другие-режимы"></a>

## Справочные разделы

- [[02 Areas/ML & DL/01 Справочник/_index|Предметный справочник ML, DL и LLM]]
- [[02 Areas/ML & DL/02 Атлас моделей/_index|Модели и семейства]]
- [[02 Areas/ML & DL/02 Атлас моделей/Timeline|Хронология моделей и технологий]]
- [[02 Areas/ML & DL/03 Исследовательские линии/_index|Обзоры направлений]]
- [[02 Areas/ML & DL/04 Вопросы/_index|Вопросы со ссылками на ответы]]
- [[02 Areas/ML & DL/05 Источники/_index|Курсы, статьи и исходные материалы]]
- [[02 Areas/ML & DL/06 Практика/_index|Практика]]
