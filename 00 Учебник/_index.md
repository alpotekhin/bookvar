---
title: Машинное обучение и языковые модели
type: textbook-chapter
status: active
last_updated: 2026-11-14
---

# Машинное обучение и языковые модели

Начните с [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|векторов, матриц и тензоров]].
В [[#полный-маршрут|оглавлении]] главы стоят в порядке чтения. [[02 Areas/ML & DL/00 Учебник/00 Как пользоваться учебником|Как пользоваться учебником]]
поможет выбрать главу, если основы уже знакомы.

<a id="полный-маршрут"></a>

## Основной курс

За математикой и нейросетями следуют языковые модели и Transformer. Затем разбираем, как обучить LLM, получить и оценить ответ, подключить поиск и инструменты.

### I. Математические и ML-основания

- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры|Векторы, матрицы и тензоры]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/02 Производная и градиент|Производная и градиент]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм|Вероятность, правдоподобие и логарифм]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/04 Функции потерь|Функции потерь]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/05 Train, validation и test|Train, validation и test]]
- [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание|Переобучение, регуляризация и оценивание]]

<a id="iv-нейронные-сети"></a>
<a id="iii-нейронные-сети"></a>

### II. Нейронные сети

- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP|Нейрон и MLP]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации|Функции активации]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|Вычислительный граф и backpropagation]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|SGD, Momentum, Adam и расписания шага]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections|Инициализация, нормализация и остаточные связи]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация|Dropout и регуляризация]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований|Аугментация данных — Mixup, CutMix и политики преобразований]]
- [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet|CNN — от свёртки до ResNet]]

<a id="v-текст-до-transformer"></a>
<a id="iv-текст-до-transformer"></a>

### III. Токенизация и языковые модели

- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|Представление текста числами]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель|N-граммная языковая модель]]
- [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction|Вероятность текста и next-token prediction]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings|Word2Vec, GloVe и распределительные представления]]
- [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram|BPE, WordPiece и Unigram]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT|RNN и BPTT]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU|LSTM и GRU]]
- [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора|Seq2Seq с фиксированным контекстом]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|Attention для перевода]]

<a id="vi-transformer-bert-и-gpt"></a>
<a id="v-transformer-bert-и-gpt"></a>

### IV. Transformer, BERT и T5

- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V|Self-Attention — Q, K, V]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Маскирование, головы и формы тензоров]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация|Позиционная информация]]
- [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Полный Transformer: энкодер и декодер]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|Три архитектуры Transformer]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa|BERT, RoBERTa и DeBERTa]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5 — text-to-text Transformer]]

<a id="vii-анатомия-современной-llm"></a>
<a id="vi-анатомия-современной-llm"></a>

### V. От GPT-1 к LLaMA

- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1 — генеративное предобучение]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2 — задачи как продолжение текста]]
- [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3 — in-context learning]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA как базовая архитектура]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block|Современный decoder block]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual|Pre-norm, RMSNorm, SwiGLU и residual]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE|RoPE]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating|Dense FFN — расширение размерности и SwiGLU]]
- [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/05 Transformer с нуля — формы, параметры и стоимость|Transformer с нуля — формы, параметры и стоимость]]

<a id="эффективное-внимание-moe-и-альтернативы-transformer"></a>

### VI. Внимание с длинным контекстом и MoE

- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|MHA, MQA и GQA]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache|MLA и сжатие KV-cache]]
- [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание|Длинный контекст — расширение, разреженность и оценивание]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving|Mixture of Experts — routing, capacity и serving]]
- [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры|Mamba, RWKV, RetNet и гибридные архитектуры]]

<a id="ix-обучение-llm"></a>
<a id="viii-обучение-llm"></a>

### VII. Предобучение языковой модели

После устройства декодера разбираются корпус, обучающая цель и выбор размера модели при ограниченном бюджете.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/01 Данные и pre-training|Данные и предобучение]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных|Сбор, очистка и смеси данных]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса|Дедупликация, PII и контроль качества корпуса]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction|Целевая функция предобучения]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws|Законы масштабирования]]
- [[00 Учебник/11 Pre-training и Scaling/41j Диффузионные языковые модели — MDLM и LLaDA|Диффузионные языковые модели: MDLM и LLaDA]]

### VIII. Декодирование

- [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена|Декодирование и выбор следующего токена]]

<a id="viii-post-training-и-рассуждения"></a>

### IX. Дообучение и рассуждения

Предобученная модель уже предсказывает следующий токен. Здесь меняются обучающие данные, сигнал качества и способ оптимизации поведения.

- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/00 Карта модуля и источники|Постобучение языковых моделей]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data|SFT: обучение на демонстрациях]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01a Fine-tuning и PEFT — adapters, LoRA, QLoRA и soft prompts|Fine-tuning и PEFT]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data|Данные предпочтений]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling|Модель награды]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM|Градиент стратегии и PPO]]
- [[00 Учебник/12 Post-training и Alignment/00a Цели alignment — HHH, Constitutional AI и RLAIF|HHH, Constitutional AI и RLAIF]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO|DPO]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR: проверяемая награда]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1|GRPO и DeepSeek-R1]]
- [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation|Дистилляция рассуждений]]
- [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Вычисления во время ответа]]

<a id="x-инференс-и-serving"></a>
<a id="ix-инференс-и-serving"></a>
<a id="ix-генерация-и-оценивание"></a>

### X. Оценивание языковых моделей

- [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей и контаминация]]

<a id="xii-retrieval-и-rag"></a>
<a id="xi-retrieval-и-rag"></a>
<a id="retrieval-и-rag"></a>

### XI. Поиск и RAG

Нужны векторные представления, BERT и оценивание. Последовательно разбираются обучение embeddings, поиск, повторное ранжирование и сборка ответа с найденным контекстом.

- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning|Эмбеддинги и метрическое обучение]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid|Лексический, плотный и гибридный поиск]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction|Переранжирование документов]]
- [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер|RAG: от источника до проверяемого ответа]]

<a id="xiii-мультимодальные-модели"></a>
<a id="мультимодальные-модели"></a>

### XII. Мультимодальные модели

Начните после CNN и Transformer. Модуль проходит путь от зрительного представления к языковому ответу, затем рассматривает документы, видео, аудио и оценивание. Генерация изображений вынесена в отдельный модуль.

- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64 Мультимодальные модели|Визуальные токены: ViT, CLIP и SigLIP]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64a Connectors и fusion|Как подключить визуальный энкодер к LLM]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции|Разрешение, плитки и позиции изображения]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Как обучают мультимодальные модели]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding|Чтение документов: OCR, поля и координаты]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|Как модели понимают видео и отвечают голосом]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|Как оценивать качество и задержку VLM]]
- [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64h Единая мультимодальная последовательность и Chameleon|Chameleon: текст и изображения в одной последовательности]]

<a id="xiv-инструменты-и-агенты"></a>
<a id="инструменты-и-агенты"></a>

### XIII. Инструменты и агенты

Нужны генерация, post-training и оценивание языковых моделей. Сначала разбираются вызовы инструментов и управление контекстом, затем память, планирование, приложения и безопасность.

- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/00 Agent Harness и Context Engineering — карта модуля|Как агент работает с инструментами]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию|Как вызов функции становится действием]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/66 Agent harness и context engineering|Как агент управляет контекстом]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Память, планирование и оркестрация агентов]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/68 Оценивание агентных систем|Оценивание агентных систем]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Программирующие агенты и работа с интерфейсами]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Формальные доказательства и математические агенты]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Агенты научного поиска и discovery]]
- [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Безопасность агентных систем]]

## Углублённые модули

В каждом модуле свой порядок глав. В конце модуля переход ведёт обратно к выбору темы, а не в следующий независимый курс.


<a id="ii-классическое-машинное-обучение"></a>

### Классическое машинное обучение

- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/00 Карта модуля|Карта модуля и источники]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline|Задача обучения и predictive pipeline]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация|Линейная регрессия и классификация]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/03 Деревья решений|Деревья решений]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting|Bagging, random forest и gradient boosting]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога|Оценивание, кросс-валидация и выбор порога]]

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

### Отбор данных и частичная разметка

Нужны функции потерь, разделение данных и аугментации. Здесь решается, какие примеры размечать, отбирать или создавать и как проверить пользу такого выбора.

- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting|Отбор примеров: EL2N, GraNd и forgetting]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров|Active learning: запрос меток]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41d Геометрический отбор — k-center, coreset и herding|Геометрический отбор: k-center, coreset и herding]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41e Обучение с частичной разметкой — pseudo-labeling и FixMatch|Частичная разметка: pseudo-labeling и FixMatch]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества|Отбор данных: стоимость, шарды и проверка качества]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы|Синтетические данные и учебные программы]]

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
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/09 Компиляция графов — PyTorch, JAX и CUDA Graphs|Компиляция графов: PyTorch, JAX и CUDA Graphs]]

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
- [[05 Источники/Курсы|Библиотека учебных курсов]]
- [[02 Areas/ML & DL/06 Практика/_index|Практика]]

<a id="gpt-decoder-route"></a>

## От GPT-1 к LLaMA

Следует ли `An animal rests.` из `A cat sleeps.`? В GPT-1 ответ выдаёт
дообученный классификатор. В GPT-2 его можно искать среди продолжений
текста после `Relation:`. В GPT-3 и LLaMA перед парой можно поставить
размеченные примеры, чтобы показать, какие ответы допустимы. Это
учебный пример постановки одной задачи четырьмя способами; он не
воспроизводит результаты оригинальных экспериментов.

При предобучении все четыре модели предсказывали следующий токен.
Внутри декодера при этом менялись нормализация, позиции и маска
внимания; отдельно менялись размер модели и объём обучающих данных.

| Глава | Как получали ответ и что изменилось |
| --- | --- |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение|GPT-1 (2018)]] | [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение#Логическое следование|Отдельная голова оценивает три класса]] по состоянию последнего токена. И голову, и декодер дообучают на размеченных парах.<br><br>117 млн параметров, окно 512 токенов. Предобучение на BooksCorpus; LayerNorm стоит после остаточного сложения. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык|GPT-2 (2019)]] | После `Relation:` сравнивают [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-relation-prefix|вероятности текстовых меток]]. Веса не меняются.<br><br>До 1,5 млрд параметров, окно 1024 токена. Веб-корпус WebText, байтовый BPE; LayerNorm переходит на вход внимания и FFN. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning|GPT-3 (2020)]] | Перед тестовой парой ставят [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-relation-shots|ноль, один или несколько размеченных примеров]]. Чтобы сменить задачу, меняют запрос.<br><br>До 175 млрд параметров, окно 2048 токенов. Восемь моделей обработали по 300 млрд токенов; полные слои внимания чередуются с локальными. |
| [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура|LLaMA v1 (2023)]] | Как и у GPT-3, метку оценивают как продолжение запроса. [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt3-to-llama1|Тот же текст разбивается на другие токены]].<br><br>От 7 до 65 млрд параметров при том же окне в 2048. Полное внимание во всех слоях, RMSNorm, RoPE и SwiGLU. Модели обработали от 1 до 1,4 трлн токенов. |

В [[#gpt-decoder-route-comparison|разборе с рисунками]] определяем
тональность одного отзыва. Сначала сравниваем, как модели получают
ответ, затем возвращаемся к устройству декодера и данным предобучения.
Читать по порядку:
[[#gpt-route-gpt1|GPT-1]] →
[[#gpt-route-gpt2|GPT-2]] →
[[#gpt-route-gpt3|GPT-3]] →
[[#gpt-route-llama|LLaMA]].

[[#gpt-decoder-route-sentiment|Один отзыв во всех четырёх моделях]] ·
[[#gpt-decoder-route-architecture|Схемы декодеров]] ·
[[#gpt-decoder-route-depth-width|Глубина, ширина и число весов]] ·
[[#gpt-decoder-route-cache|Память при генерации]] ·
[[#gpt-decoder-route-data|Корпуса и объём обучения]] ·
[[#gpt-decoder-route-training-budget|Затраты на обучение и вывод]] ·
[[#gpt-decoder-route-evidence|Результаты опытов]] ·
[[#gpt-decoder-route-bpb|Как сравнивать потери при разных токенизаторах]] ·
[[#gpt-decoder-route-calibration|Калибровка меток]]

<a id="gpt-decoder-route-comparison"></a>

## Сравнение GPT-1, GPT-2, GPT-3 и LLaMA

[[#gpt-decoder-route|К четырём главам]] · [[#полный-маршрут|Основной курс]]

На схеме ниже `book` стоит над входом `this`: для прогноза декодер читает `this` и предыдущие токены. Следом разберём, как тем же декодером определить тональность отзыва `i liked it`.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ntp-review-2026-10/d2l-gpt-decoder-original.svg|Декодер предсказывает book над входным this. В строке book справа доступны начало последовательности, I, like и this, но не book.]]

*Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola, [Dive into Deep Learning, рис. 11.9.6](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#fig-gpt-decoder-only), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Рисунок без изменений. Справа в строке цели `book` открыт столбец `this`, но закрыт столбец `book`. На рисунке полная причинная маска; локальный слой GPT-3 может отсечь и часть позиций слева.*

<a id="gpt-decoder-route-sentiment"></a>

### Один отзыв, четыре поколения

<a id="gpt-route-gpt1"></a>

#### GPT-1 (2018): отдельная голова для тональности

Возьмём отзыв `i liked it` и поставим вокруг него служебные токены
`Start` и `Extract`. На
[рисунке 1 статьи GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=4)
это ветка `Classification`. Причинное внимание даёт последней
позиции доступ ко всему отзыву. Из состояния `Extract` линейная
голова получает две оценки, по одной на каждый класс. При
дообучении на SST-2 обновляют и голову, и декодер. В самом отзыве
метки нет: она нужна для вычисления ошибки.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt1-pretraining-2026-11-10/cs224n-decoder-classifier.png|Последнее состояние h_T получает информацию от всех входных позиций. Только h_T подаётся в линейную голову, которая различает два класса тональности.]]

*Stanford CS224N, [Lecture 9: Pretraining, слайд 41](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf#page=41), [условия использования слайдов](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/#logistics). Правая схема без изменения обозначений.*

На рисунке последняя входная позиция $w_T$ соответствует `Extract`, а $h_T$ — её состоянию на выходе декодера. Стрелка к `Linear` идёт только от $h_T$. Смайлики над ним обозначают классы, а не токены, которые модель дописывает к отзыву.

В [отдельном опыте](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=8)
авторы обошлись без этой головы и без дообучения. К отзыву приписали
`very`, после чего сравнили вероятности двух продолжений:
`positive` и `negative`. В этом опыте метка уже была частью
словаря. Такую языковую голову GPT-1 уже использовала при предобучении.
На соседнем слайде над каждым входом стоит прогноз следующего
токена: состояние после `w₅` идёт в ту же проекцию `A, b`, что и
состояния предыдущих позиций. На схеме нет отдельного выхода для
тональности.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-14/cs224n-vocabulary-head.png|Над каждым входом показан прогноз следующего токена. Последнее состояние h_T проходит через языковую голову A, b, как и предыдущие состояния.]]

*Stanford CS224N, John Hewitt, [Lecture 9: Pretraining, слайд 42](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture9-pretraining.pdf#page=42), [условия использования слайдов](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/#logistics). Правая схема с исходными подписями. Смещение `b` относится к общей схеме; у GPT-2 в выпущенном коде выходная проекция без смещения.*

Слайд описывает и случай, когда языковую модель дообучают как
генератор. В zero-shot опытах GPT-2 этот этап пропускают: чтобы
сменить задачу, меняют префикс, а обученная проекция остаётся прежней.

<a id="gpt-route-gpt2"></a>

#### GPT-2 (2019): метка как продолжение текста

Вместо `very` допишем
к `Review: i liked it` строку `Sentiment:`. Если вероятности
следующих токенов ` positive` и ` negative` равны 0,18 и 0,07,
выбираем положительный класс. Это условные числа: сумма не равна
единице, поскольку выход языковой модели охватывает весь словарь.
В отчёте GPT-2 этот подход применили, например, к
[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#Перевод и ответ через префикс|переводу по текстовому префиксу]].
На тональности GPT-2 позже проверили
[Zhao et al., §2 и приложение A, табл. 4](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=2).

Эффект самого префикса можно проверить, оставив модель и задачу
прежними. В [таблице 4 отчёта GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=6)
авторы сравнивают два способа начать генерацию после газетной статьи:
с маркером `TL;DR:` и без него. В описанном протоколе модель дописывает
100 токенов, выбирая на каждом шаге из двух самых вероятных, а первые
три сгенерированных предложения идут в резюме. Если убрать маркер,
ROUGE-2 F1 падает с 8,27 до 4,03. Эта метрика сопоставляет пары
соседних слов с эталонным резюме; она не проверяет, сохранила ли
модель факты исходной статьи.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt2-2026-10-22/gpt2-output-projection-2.png|Состояние одной позиции ширины 768 сравнивается с каждой из 50 257 строк токенной таблицы GPT-2.]]

*Jay Alammar, [The Illustrated GPT-2, «Model Output»](https://jalammar.github.io/illustrated-gpt2/#model-output), [оригинал PNG](https://jalammar.github.io/images/gpt2/gpt2-output-projection-2.png), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Рисунок без изменений.*

У Аламмара показана первая позиция, с которой начинается генерация. В нашем запросе берём состояние последнего токена строки `Sentiment:`. У малой GPT-2 это вектор из 768 чисел. Скалярное произведение с каждой строкой таблицы даёт один из 50 257 логитов. Нас интересуют два из них — для ` positive` и ` negative`. Остальные участвуют в нормировке softmax, но в выборе между двумя метками не участвуют.

<a id="gpt-route-gpt3"></a>

#### GPT-3 (2020): примеры в запросе

Оставим тот же отзыв последним, а перед ним
поставим два готовых примера: `Well acted.` с меткой `positive` и
`Too slow.` с меткой `negative`. Декодер читает эти метки слева
от последнего `Sentiment:`; оптимизатор не запускается. При сравнении
трёх режимов инструкция в начале запроса остаётся прежней, например
`Classify the review as positive or negative.` Один размеченный отзыв
даёт one-shot, а запрос без размеченных отзывов — zero-shot.
GPT-2 уже получала примеры в отдельных опытах. В
[статье GPT-3, §2 и рис. 2.1](https://arxiv.org/html/2005.14165v4#S2.F1)
число примеров меняли систематически, сохраняя веса.
[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#Перевод: направление и параллельные примеры|Те же режимы на примере перевода]].

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-13/lu22-figure2-prompt-order.png|Слева три варианта одного запроса. Четыре размеченных примера переставлены, а тестовый текст остаётся последним. Одна языковая модель даёт отдельный ответ на каждый запрос.]]

*Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel и Pontus Stenetorp,
[Fantastically Ordered Prompts and Where to Find Them, §2](https://arxiv.org/html/2104.08786v2#S2),
[рис. 2, с. 2 версии v2](https://arxiv.org/pdf/2104.08786v2#page=2),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Рисунок из статьи, без перерисовки.*

На схеме `Train 1`–`Train 4` — четыре пары «отзыв и метка» в тексте
запроса. Каждая строка слева — отдельный запуск той же модели (`PLM`).
В первой строке порядок `1, 2, 3, 4`, во второй — `3, 4, 2, 1`.
Блок `Test` всегда последний и остаётся без метки.
[[#gpt-decoder-route-prompt-order|На графиках ниже]] показано, насколько
ответы менялись от такой перестановки.

В [опыте GPT-3 на логическом следовании, ANLI R3](https://arxiv.org/html/2005.14165v4#A8)
модель 175B на тестовой выборке набрала 34,5% без примеров, 35,1%
с одним и 40,2% с пятьюдесятью. Ответ выбирали из трёх текстовых
меток ([пример запроса, приложение G](https://arxiv.org/html/2005.14165v4#A7)).
Добавление пятидесяти примеров дало 5,7 процентного пункта, но
точность осталась близка к 33,3% случайного равновероятного выбора.
На схеме выше показан запрос для тональности; в ANLI авторы
использовали другой шаблон.

<a id="gpt-route-demonstration-controls"></a>

Прибавка от примеров ещё не показывает, использовала ли модель
связь между каждым отзывом и его меткой. Min и соавторы оставили
демонстрации на месте, но вместо правильной метки каждого примера
выбирали случайную из допустимых. В нашем запросе `Well acted.` может
получить `negative`, а `Too slow.` сохранить ту же метку. Модель
по-прежнему видит примеры отзывов, возможные метки и формат пары;
соответствия между отзывами и метками больше не обязаны быть верными.
Тестовый отзыв и его эталон остаются прежними.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-continuity-2026-11-10/min-figure1-controls.png|Две панели: macro-F1 для классификации и точность для выбора ответа. У GPT-3 в правой группе два столбца с демонстрациями близки и выше столбца без примеров.]]

*Sewon Min et al., [Rethinking the Role of Demonstrations, рис. 1, с. 11048](https://aclanthology.org/2022.emnlp-main.759.pdf#page=1),
[§§3–4](https://arxiv.org/html/2202.12837v2#S3),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Обе панели
из статьи; оси, легенды и подписи не изменены.*

Сравним три правых столбца верхней панели. У GPT-3 175B без примеров
около 41% macro-F1, с правильными метками около 58%, со случайными
около 55%. Нижняя панель показывает другую метрику, точность выбора
ответа. На каждой панели усреднены результаты по трём наборам данных.
В обоих случаях авторы оценивают вероятность тестового текста при
каждой метке (*channel method*). В примере с `Sentiment:` наоборот:
мы оцениваем метку после текста. При случайных метках GPT-3 сохранила
большую часть прибавки. Для задачи, в которой правильное соответствие
меток можно узнать только из запроса, нужна отдельная проверка.

<a id="gpt-route-llama"></a>

#### LLaMA v1 (2023): тот же способ задать задачу, другой декодер

Базовая LLaMA тоже принимает размеченные примеры в запросе. Pan et al.
проверили, могут ли GPT-3 и LLaMA восстановить из примеров новое,
но согласованное соответствие между классом и меткой.
В средней строке рисунка положительному примеру соответствует `0`,
поэтому эталон для `i liked it` тоже `0`. Если во всех демонстрациях
обменять `0` и `1`, эталоном станет `1`. Веса и новый отзыв не
меняются. В отличие от случайных меток в опыте Min, здесь пары
согласованы, а знания о значении слова `positive` недостаточно:
из запроса нужно узнать, какая цифра ему соответствует. Столбец
справа показывает эталон, а не гарантированный ответ модели.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/pan-icl-figure1.png|Один отзыв в трёх запросах: случайные словесные метки, согласованные цифровые метки и правильные словесные метки. Справа показан эталон.]]

*Jane Pan, Tianyu Gao, Howard Chen и Danqi Chen, [What In-Context Learning “Learns” In-Context, рис. 1](https://aclanthology.org/2023.findings-acl.527.pdf#page=1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Оригинальный рисунок из статьи, без перерисовки. [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-label-mapping|Постановка и результаты сравнения]].*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/pan-icl-figure2.png|Нижние графики GPT-3 и LLaMA: точность с числовыми метками растёт от 8 к 32 демонстрациям. Со случайными словесными метками роста нет.]]

*Pan et al., [рис. 2, с. 8300](https://aclanthology.org/2023.findings-acl.527.pdf#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Шесть оригинальных панелей с общей легендой.*

Нижний ряд показывает, что происходит при добавлении примеров одной
и той же модели. Слева у GPT-3 175B при переходе от 8 к 32 примерам
точность с числовыми метками (`Abstract`) выросла примерно с 45%
до 60%. В центре у LLaMA-65B — с 48% до 57%. В этих же панелях
точность со случайными метками (`Random`) не растёт. Это средние
по 16 наборам классификации; сравнивать надо изменение внутри панели:
авторы оценивали GPT-3 и LLaMA на выборках разного размера.
Прибавка у обеих моделей согласуется с тем, что дополнительные
примеры помогали восстановить соответствие между классами и цифрами.
Опыт не показывает, какую роль здесь сыграли RoPE или SwiGLU.

Сам формат запроса пережил смену архитектуры. У GPT-3 и LLaMA v1
одно окно на 2048 токенов, но первая LLaMA снова использует полную
причинную маску во всех слоях. Она по-другому нормализует состояния,
задаёт позиции и вычисляет FFN. Ниже эти изменения разобраны по
порядку; [[#gpt-decoder-route-training-budget|сравнение затрат]]
отдельно показывает, зачем авторы дольше обучали меньшие модели.

<a id="gpt-decoder-route-architecture"></a>

### Что менялось внутри декодера

| Модель / окно | Изменение в блоке |
| --- | --- |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение#gpt1-decoder-architecture|GPT-1]]<br>512 | Полное причинное внимание, обучаемые абсолютные позиции, LayerNorm после сложения. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-pre-norm|GPT-2]]<br>1024 | LayerNorm переходит на вход внимания и FFN. В конце стека появляется отдельная LayerNorm. |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt3-full-local-attention|GPT-3]]<br>2048 | Полные слои внимания чередуются с локальными. В локальном слое часть дальних токенов слева закрыта. |
| [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-comparison|LLaMA v1]]<br>2048 | Снова полное причинное внимание во всех слоях. RMSNorm вместо LayerNorm, RoPE вместо таблицы позиций, SwiGLU вместо FFN с GELU. |

На рисунке Xiong et al. прямой путь от входа к сумме идёт снизу вверх по серой стрелке. В порядке GPT-1 сумма ещё проходит через LayerNorm (слева). В порядке GPT-2 она уже идёт дальше, а нормализация стоит на входе в вычисление внимания (справа). GPT-3 сохраняет этот порядок; LLaMA ставит на место LayerNorm другой оператор.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-pre-post-norm.png|Слева LayerNorm после сложения, справа LayerNorm в ветвях внимания и FFN до сложения.]]

*Ruibin Xiong et al., [On Layer Normalization in the Transformer Architecture, рис. 1, с. 2](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Обе панели и оригинальная подпись из статьи. Схема показывает положение нормализации, а не полную архитектуру GPT.*

Между GPT-2 и GPT-3 изменилось то, какие токены доступны вниманию
в одном слое. Возьмём строку `on` на двух матрицах ниже. Слева стоят
четыре единицы: запрос читает `The`, `cat`, `sat` и себя. В средней
матрице осталось три: связь с `The` закрыта, потому что окно на рисунке
вмещает три позиции вместе с текущей. Последнее `the` находится справа
от запроса и закрыто в обоих случаях. Единицы здесь показывают
разрешённые связи, а не веса внимания.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/long-context-review-2026-09/mistral-sliding-window.png|В строке on полная маска разрешает четыре позиции, локальная три: дальний токен The закрыт. Справа показано, как информация передаётся через несколько локальных слоёв.]]

*Albert Q. Jiang et al., [Mistral 7B, рис. 1](https://arxiv.org/html/2310.06825v1#S2.F1),
[оригинал PNG](https://arxiv.org/html/2310.06825v1/swa.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), без изменений.
Три позиции взяты из учебной схемы Mistral; ширину окна GPT-3 авторы не указали.*

В [GPT-3, §2.1](https://arxiv.org/html/2005.14165v4#S2.SS1) полная и локальная
маски чередуются. Если `cat` прочитал `The` в полном слое, то в следующем
локальном слое `on` может получить сведения о `The` через состояние
`cat`, хотя напрямую обратиться к `The` не может. В первой LLaMA
прямая связь снова разрешена во всех слоях, как слева на рисунке.
При этом окно обеих моделей рассчитано на 2048 токенов. Длина окна
задаёт, сколько токенов можно подать; маска задаёт, кто кого прочитает
внутри одного слоя.

<a id="gpt-decoder-route-attention-pairs"></a>

По тому же рисунку можно посчитать, сколько пар «запрос — ключ»
разрешено во всём слое. Для пяти токенов полная маска даёт
$1+2+3+4+5=15$ пар на голову, локальная с окном три позиции —
$1+2+3+3+3=12$. Если подать вход на всё окно модели, для полной
маски получится $T(T+1)/2$: $131\,328$ пар при $T=512$ у GPT-1 и
$524\,800$ при $T=1024$ у GPT-2. Почти вчетверо больше пар, хотя
длина входа удвоилась. Полный слой GPT-3 или LLaMA при $T=2048$
допускает $2\,098\,176$ пар. У локального слоя GPT-3 их меньше;
точное число из отчёта не получить без ширины полосы. Это подсчёт
разрешённых связей за проход по всему входу, а не замер скорости
или размера матрицы в памяти.

На схеме Stanford CS336 состояние токена идёт снизу вверх.
Сначала оно раздваивается: один путь ведёт прямо к нижнему сложению
`Add`, другой проходит через `Norm` и внимание. Такой порядок был
уже в GPT-2. У первой LLaMA в этой ветви `Norm` означает RMSNorm:
она масштабирует вектор без вычитания среднего. Внутри внимания
надпись `w/ RoPE` относится к вычислению запроса и ключа.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/cs336-prenorm-block.png|В нижней ветви норма и причинное внимание с RoPE, в верхней норма и FFN. Обе ветви возвращаются в свой Add.]]

*Stanford CS336 Staff, [Assignment 1: Basics, рис. 2, с. 13](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=13),
«A pre-norm Transformer block», [MIT](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/LICENSE).
Схема из задания, без перерисовки.*

<a id="gpt-decoder-route-positions"></a>

В [коде GPT-2](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L147-L165)
к вектору токена на позиции $i$ прибавляют строку $p_i$ из обучаемой
таблицы `wpe` и только потом подают сумму в первый блок. У младшей
GPT-2 таблица содержит $1024\times768=786\,432$ параметра. Если
перенести два токена с позиций $(0,1)$ на $(3,4)$, на вход попадут
уже $p_3$ и $p_4$ вместо $p_0$ и $p_1$. Таблица не связывает эти пары
условием, что между токенами одинаковое расстояние.

[Первая LLaMA](https://arxiv.org/html/2302.13971v1#S2.SS2) убирает эту
таблицу. В каждом слое она сначала вычисляет запрос $q$ и ключ $k$,
затем поворачивает пары их координат. На верхней панели RoFormer
показана одна такая пара: вектор $(x_1,x_2)$ поворачивается на
$m\theta_1$, где $m$ — позиция, а $\theta_1$ — фиксированная частота
для этих двух координат. На нижней панели тот же расчёт повторяется
для других пар координат и других позиций.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png|Верхняя панель показывает поворот двух координат запроса или ключа на угол mθ₁. Внизу тот же расчёт применяется к каждой паре координат с её частотой.]]

*Jianlin Su et al., [RoFormer, рис. 1 и §3.2](https://arxiv.org/html/2104.09864v5#S3.SS2),
[исходная страница PDF](https://arxiv.org/pdf/2104.09864v5#page=5),
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
Полный рисунок и авторская подпись, без изменений.*

Возьмём только одну пару координат: до поворота $q=k=(1,0)$,
частота $\theta_1=1$ радиан на позицию. При ключе на позиции 0
и запросе на позиции 1 углы равны 0 и 1. Скалярное произведение
двух единичных векторов равно косинусу угла между ними:
$\cos(1)\approx0.5403$. На позициях $(3,4)$ оба вектора дополнительно
повёрнуты на 3 радиана, угол между ними по-прежнему 1.
Если оставить ключ на позиции 0, а запрос перенести на 2,
получится $\cos(2)\approx-0.4161$. Это вклад одной пары координат
до масштабирования и softmax. При добавлении нового префикса
могут измениться сами $q$ и $k$ и набор доступных ключей, поэтому
из примера не следует, что ответ модели останется прежним.
В [коде LLaMA](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L115-L143)
поворот стоит перед сравнением запросов с ключами, а значения $V$
через него не проходят.

Верхняя ветвь (`Position-Wise Feed-Forward`) обрабатывает каждый токен отдельно. На рисунке ниже слева вход проходит через матрицу $W$, активацию GELU и выходную матрицу $W_o$. В центре тот же вход поступает в две проекции. SiLU действует на одну из них; затем результаты перемножаются поэлементно, и $W_o$ возвращает вектор к ширине декодера. Так в FFN LLaMA появляется третья матрица.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/tajima-figure1-ffn.svg|Слева FFN с GELU и двумя матрицами. В центре SwiGLU с двумя входными проекциями, Swish на одной из них, поэлементным произведением и выходной матрицей. Справа более поздний SwiMGLU.]]

*Yukito Tajima, Nakamasa Inoue, Yusuke Sekikawa, Ikuro Sato и Rio Yokota, [Masked Gated Linear Unit, рис. 1](https://arxiv.org/html/2506.23225v1#S1.F1), [оригинал SVG](https://arxiv.org/html/2506.23225v1/fig_swimlu_v2.svg), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Рисунок без изменений. Правая панель показывает отдельный метод авторов, которого нет в LLaMA.*

У [GPT-3 6.7B](https://arxiv.org/html/2005.14165v4#S2.T1) и [LLaMA-7B](https://arxiv.org/html/2302.13971v1#S2.T2) ширина декодера равна 4096. GPT-3 расширяет вектор до 16 384 координат, а LLaMA — до 11 008 в каждой из двух ветвей. В первом случае две матрицы дают около 134,2 млн весов, во втором три — около 135,3 млн: разница меньше 1% на FFN, если не считать смещения. [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-ffn|Разбор в главе LLaMA]] выводит эту ширину из числа матриц и показывает, как её округляет исходный код.

Даже одинаковое окно в 2048 позиций не вмещает одинаковый объём текста:
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-tokenization|одна строка в разных токенизаторах]] занимает разное число позиций. Для сравнения с отдельными стеками энкодера и декодера см. [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer|T5]].

<a id="gpt-decoder-route-depth-width"></a>

### Откуда берутся миллиарды параметров

На рисунке Аламмара число у верхнего блока показывает глубину стека:
у GPT-2 Small 12 блоков, у Extra Large 48. Число под стеком — ширина
состояния одного токена, 768 или 1600. Обе модели принимают до 1024
токенов. Более высокий стек означает, что состояние проходит через
большее число блоков, а не что модель читает более длинный текст.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-sizes-hyperparameters-3.png|Четыре стека GPT-2: 12, 24, 36 и 48 блоков. Под ними подписана ширина состояния токена: 768, 1024, 1280 и 1600.]]

*Jay Alammar, [The Illustrated GPT-2, «Transformers for Language Modeling»](https://jalammar.github.io/illustrated-gpt2/#transformers-for-language-modeling), [оригинал PNG](https://jalammar.github.io/images/gpt2/gpt2-sizes-hyperparameters-3.png), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Рисунок без изменений.*

Пусть $d$ — ширина состояния. В блоке GPT-2 три проекции для $Q$, $K$ и
$V$ и одна выходная проекция внимания дают $4d^2$ весов. FFN сначала
расширяет состояние с $d$ до $4d$, а затем возвращает к $d$: ещё $8d^2$.
Если считать только эти матрицы, стек из $L$ блоков содержит $12Ld^2$
весов. При переходе от Small к Extra Large блоков становится вчетверо
больше, а каждая матрица растёт в обеих размерностях. Получается
около 85 млн весов против 1,47 млрд, разница в 17,4 раза. Для всей модели
разница меньше, около 12,5 раза: токенная таблица растёт только с $d$,
а число строк словаря остаётся прежним.
[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#Что изменилось относительно GPT-1|Полный подсчёт GPT-2]]
учитывает также позиции, смещения и нормы.

У GPT-1, GPT-2 Small и GPT-3 Small по 12 блоков ширины 768.
При переходе от GPT-1 к GPT-2 Small прибавку в весах
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-parameter-budget|в основном даёт словарь]], а у GPT-3 175B по сравнению с GPT-3 Small
растёт сам стек: 96 блоков ширины 12 288. Только матрицы стека дают
около 174 млрд весов. Локальная маска меняет допустимые пары запроса
и ключа, но не размеры этих матриц. Окно у обеих GPT-3 одинаковое,
2048 токенов ([§2.1 и табл. 2.1](https://arxiv.org/html/2005.14165v4#S2.SS1)).
В LLaMA-7B в FFN три матрицы вместо двух, и внутренний вектор имеет
11 008 координат. На блок приходится $4d^2+3d\cdot11008$ весов
в этих матрицах и проекциях внимания, около 202 млн при $d=4096$
([[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#Формы и параметры исходных моделей|подсчёт по формам LLaMA]]).

<a id="gpt-decoder-route-cache"></a>

### Что растёт в памяти при генерации

На рисунке Hugging Face модель уже выдала `will` после `I am sure this project`. Чтобы получить следующий токен `be`, она вычисляет запрос только для `will`. Ключи и значения для семи предыдущих позиций, включая начало последовательности, уже есть в кэше. К ним добавляется восьмая пара, для `will`. У головы один запрос, но восемь пар K/V. На следующем шаге понадобится запрос для `be`, а запрос для `will` хранить не нужно.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/hf-kv-cache.png|На шаге с токеном will показаны один запрос, семь сохранённых пар K/V и одна новая пара. Выходной токен — be.]]

*Rémi Ouazan Reboul, Arthur Zucker и Luc Georges, [Continuous batching, KV-cache](https://huggingface.co/blog/continuous_batching#kv-cache), [оригинал PNG](https://huggingface.co/datasets/huggingface/documentation-images/resolve/c8c3cf72b3b724ce734259e4a086d30618a9198b/blog/continuous_batching/kv_cache.png), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Рисунок без изменений. $A$ обозначает ширину одной головы.*

В опубликованной [реализации GPT-2](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py#L69-L112) новый K/V присоединяют к состояниям предыдущих позиций. В [первой LLaMA](https://github.com/meta-llama/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L115-L143) их записывают в заранее выделенный буфер. Ключ в LLaMA уже повёрнут RoPE; размер вектора от поворота не меняется.

Для одной позиции в каждом слое сохраняем $H_{KV}$ ключей и столько же значений, по $A$ чисел в каждом. Если число занимает $b$ байт, то для $L$ слоёв и $T$ сохранённых позиций нужно $2LH_{KV}ATb$ байт. Например, у LLaMA-7B 32 слоя, в каждом 32 головы ширины 128. При двух байтах на число получаем $2\cdot32\cdot32\cdot128\cdot2=524\,288$ байт на токен, или 512 КиБ. За 2048 позиций набирается ровно 1 ГиБ.

| Модель | K/V на один токен (16 бит) | K/V на полное окно |
| --- | ---: | ---: |
| GPT-1 (12 слоёв, 12 голов по 64) | 36 КиБ | 18 МиБ / 512 токенов |
| GPT-2 Small (12 слоёв, 12 голов по 64) | 36 КиБ | 36 МиБ / 1024 токена |
| GPT-3 6.7B (32 слоя, 32 головы по 128) | 512 КиБ | 1 ГиБ / 2048 токенов |
| LLaMA v1 7B (32 слоя, 32 головы по 128) | 512 КиБ | 1 ГиБ / 2048 токенов |

Конфигурации: [GPT-1, §4.1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=5), [GPT-2, табл. 2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4), [GPT-3, табл. 2.1](https://arxiv.org/html/2005.14165v4#S2.T1), [LLaMA, табл. 2](https://arxiv.org/html/2302.13971v1#S2.T2). В таблице посчитано только содержимое K/V для одной последовательности, если хранить все позиции во всех слоях. Это общий расчёт по конфигурациям, а не измерение памяти исторических реализаций. Веса, временные буферы и резерв под другие запросы сюда не входят.

От GPT-1 к младшей GPT-2 удвоилось окно, но на один токен уходило бы столько же байт. У выбранных GPT-3 и LLaMA совпадают и окно, и размеры K/V. При этом локальный слой GPT-3 мог бы хранить меньше старых позиций: к ключам за пределами полосы он больше не обращается. В полном слое такой экономии нет. Статья GPT-3 не сообщает ширину полосы или устройство кэша при обслуживании запросов, поэтому 1 ГиБ в таблице соответствует хранению всей истории. Первая LLaMA использует полное внимание; сокращение числа голов K/V появится у [Llama 2 70B](https://arxiv.org/abs/2307.09288), в [[00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA|групповом внимании (GQA)]].

<a id="gpt-decoder-route-data"></a>

### От книжного корпуса к смесям данных

GPT-1 и GPT-2 учили продолжать текст одной и той же функцией потерь. Но
в романах модель часто продолжает повествование, а на веб-странице ей
может встретиться вопрос с ответом или фраза с переводом. В GPT-2 авторы
проверяли, поможет ли такое предобучение решать задачи по одному
текстовому префиксу. Для GPT-3 и LLaMA исследователи задавали долю каждого
источника в обучающей смеси отдельно от размера его корпуса.

| Модель | Какие тексты попадали в обучение |
| --- | --- |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение#Первый этап: чему учит BooksCorpus|GPT-1]] | Из BooksCorpus выбирали непрерывные отрезки по 512 токенов: соседние абзацы романа могли попасть в один отрезок, но весь роман в окно не помещался. [Статья, §4.1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=5). |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#WebText и байтовый BPE|GPT-2]] | WebText собирали по внешним ссылкам с Reddit. После очистки в использованной версии осталось чуть больше 8 млн документов, около 40 ГБ текста. [Статья, §2.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=3). |
| [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#Данные и режим семплирования|GPT-3]] | К WebText2 добавили отфильтрованный Common Crawl, два книжных корпуса и английскую Wikipedia. За 300 млрд токенов обучения авторы насчитали 0,44 прохода по Common Crawl и 2,9 по WebText2. [Статья, табл. 2.2](https://arxiv.org/html/2005.14165v4#S2.T2). |
| [[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#llama-pretraining-data|LLaMA v1]] | 67% смеси отвели английской части Common Crawl после CCNet, ещё 15% — C4. Остальное составили GitHub, Wikipedia, книги, arXiv и Stack Exchange. Модели 7B и 13B обработали по триллиону токенов, 33B и 65B — по 1,4 трлн. [Статья, табл. 1 и рис. 1](https://arxiv.org/html/2302.13971v1#S2). |

40 ГБ у WebText — размер собранного текста. Счётчик GPT-3 в 300 млрд
токенов включает повторно прочитанные токены: один и тот же отрывок
учитывается каждый раз, когда попадает в обучение. А 2048 — число позиций,
доступных за один проход. Поэтому GPT-3 13B и LLaMA 13B могли иметь
одинаковое окно, хотя вторая обработала примерно в 3,3 раза больше
токенов за всё предобучение. Из-за разных токенизаторов это не значит,
что она прочитала в 3,3 раза больше слов.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/llama1-training-loss.png|Функция потерь при предобучении LLaMA. По горизонтали миллиарды обработанных токенов. Кривые 7B и 13B заканчиваются на 1000, кривые 33B и 65B — на 1400.]]

*Hugo Touvron et al., [LLaMA, рис. 1, с. 3](https://arxiv.org/pdf/2302.13971v1#page=3), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Оригинальный график из статьи.*

Найдите на оранжевой кривой 13B отметку 300 млрд, между 200 и 400.
Если идти по той же кривой вправо до триллиона, значение функции потерь
снижается. Число весов модели при этом не меняется.
На графике нет GPT-3: снижение потерь вдоль одной кривой показывает
эффект продолжения обучения LLaMA, но не разницу между двумя семействами.

<a id="gpt-decoder-route-training-budget"></a>

#### GPT-3, Chinchilla и LLaMA: что оставляли постоянным

У восьми моделей GPT-3 было по 300 млрд токенов предобучения.
Поэтому переход от 13B к 175B увеличивал и число весов, и затраты
на обработку тех же 300 млрд токенов. По этой серии можно сравнить
размеры при одинаковой длине обучения, но нельзя выбрать лучшую
модель за фиксированное число операций.

[Hoffmann et al., §3.2](https://arxiv.org/html/2203.15556v1#S3.SS2)
поставили обратный опыт: при одном бюджете обучали несколько размеров,
а число токенов подбирали под каждый размер. Если $C$ операций
распределить между $N$ весами и $D$ токенами, то при $C\approx6ND$
удвоение $N$ оставляет вдвое меньше токенов. Например, при
$C=6\cdot10^{19}$ модель на 1 млрд весов обработает около 10 млрд
токенов, а модель на 2 млрд весов — около 5 млрд. Чтобы выбрать
между ними, нужно сравнить потери после обучения: формула задаёт
бюджет, но не говорит, какая модель выучит язык лучше.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/scaling-isoflop-2026-11-09/training_flop_vs_model_size.png|По горизонтали операции обучения, по вертикали параметры. В столбце около 10¹⁹ FLOP у верхних точек потери выше, чем у части нижних.]]

*Tamay Besiroglu, Ege Erdil, Matthew Barnett и Josh You,
[Chinchilla Scaling: A replication attempt, рис. 2](https://arxiv.org/html/2404.10102v2#S2.F2),
[оригинальный PNG](https://arxiv.org/html/2404.10102v2/training_flop_vs_model_size.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Рисунок
авторов репликации без изменений. Точки восстановлены из рис. 4
работы Hoffmann et al., а не из журнала обучения.*

На графике сравнение при одном бюджете идёт по вертикали. Около
$10^{19}$ FLOP вверху видны светлые точки: там у более крупных
моделей выше потери. Авторы репликации отнесли несколько таких
точек к выбросам, поэтому по этому столбцу не стоит подбирать
оптимальный размер. Чтобы проверить сам принцип в крупном масштабе,
Hoffmann et al. обучили [Chinchilla](https://arxiv.org/html/2203.15556v1#S4)
на 1,4 трлн токенов при примерно том же бюджете, что и Gopher.
У Chinchilla было 70 млрд весов против 280 млрд у Gopher.
На LAMBADA результат вырос с 74,5% до 77,4%. Помимо размера и
числа токенов авторы изменили оптимизатор, токенизатор и пропорции
источников в обучающей смеси.

[Авторы LLaMA](https://arxiv.org/html/2302.13971v1#S1) выбирали
размер с учётом затрат на вывод. Если для сервиса выбрана модель
13B, её можно дольше обучать, не увеличивая число весов, которые
придётся загружать при запуске. На следующем графике каждая точка
оранжевой кривой относится к весам той же LLaMA 13B, сохранённым
после разного числа токенов обучения. Здесь измеряют точность
на задачах, а не потерю при продолжении текста.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/llama-2026-10-22/llama1-downstream-training.png|Точность четырёх моделей LLaMA на шести задачах по мере обучения. У LLaMA 13B результат на HellaSwag растёт примерно с 71% на 250 млрд токенов до 79% на триллионе; на SIQA линия колеблется около 50%.]]

*Hugo Touvron et al., [LLaMA, §3.7, рис. 2](https://arxiv.org/html/2302.13971v1#S3.SS7), [PDF, с. 8](https://arxiv.org/pdf/2302.13971v1#page=8), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Шесть панелей и подпись из статьи. Чтобы рассмотреть оси, откройте страницу PDF в полном размере.*

На верхней средней панели (HellaSwag) у LLaMA 13B около 71% после 250 млрд токенов и 79% после триллиона. На нижней левой (SIQA) та же кривая колеблется около 50% и не показывает сопоставимого роста. Горизонтальный пунктир — результат Chinchilla, а не промежуточные измерения GPT-3. Этот опыт позволяет сравнить ранние и поздние веса одной LLaMA. Для прямого сравнения с GPT-3 нужны результаты на одинаковой задаче и оговорки о различиях протокола.

<a id="gpt-decoder-route-repeated-data"></a>

#### Новые токены или ещё один проход по корпусу

У GPT-3 один проход по источнику не был нормой: по WebText2 авторы
насчитали 2,9 прохода, по Common Crawl — 0,44. У LLaMA после 1,4 трлн
токенов счётчик Common Crawl дошёл до 1,10, а Wikipedia — до 2,45.
Это средние по каждому источнику, а не число прочтений любого документа.
При сравнении таких смесей по общему числу токенов нельзя узнать,
сколько раз модели видели один и тот же текст. Muennighoff et al.
выделили эффект повторов отдельным экспериментом.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-13/muennighoff-jmlr-figure10.png|Слева измеренные потери при фиксированной длине обучения: у модели 8,7 млрд параметров результат при 25% неповторяющихся токенов близок к результату при 100%. Справа прогноз для более долгого обучения на корпусе фиксированного размера.]]

*Niklas Muennighoff et al., [Scaling Data-Constrained Language Models, JMLR 26(53), 2025](https://jmlr.org/papers/v26/24-1000.html),
[§6, рис. 10, с. 14](https://jmlr.org/papers/volume26/24-1000/24-1000.pdf#page=14),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Обе панели и легенда из статьи, без перерисовки.*

Слева проследите нижний ряд точек: у каждой модели 8,7 млрд параметров,
каждая обработала около 178 млрд токенов C4. При 100% это один проход
по большому корпусу; при 25% корпус содержит около 44 млрд токенов
и его проходят четыре раза. В этом опыте итоговая потеря на отложенном
тексте выросла всего на 0,5% относительно одного прохода. Сплошные линии
слева подогнаны по эксперименту. Справа сплошные линии уже показывают
прогноз: корпус фиксирован, а число обработанных токенов растёт за счёт
новых проходов. Сравнение относится к повторению всего корпуса C4;
оно не измеряет эффект повторов Wikipedia в смеси LLaMA.

<a id="gpt-decoder-route-evidence"></a>

### Что показали эксперименты

У GPT-1 есть прямой способ проверить, помогло ли предобучение:
обучить тот же Transformer на той же разметке, но начать со
случайных весов. Для SST-2, той самой задачи о тональности,
точность составила 91,3% с предобучением и 84,0% без него
([таблица 5, §5](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf#page=8)).
Разница в 7,3 пункта получена после обучения на размеченных отзывах в
обоих случаях. Средние баллы 74,7 и 59,9 относятся к восьми задачам,
у которых разные метрики: точность, F1 и корреляция.

В LAMBADA нужно предсказать последнее слово отрывка. Не меняя веса,
можно менять и подсказку, и правило выбора слова. GPT-2 получила
52,66% без фильтра и 63,24% после исключения стоп-слов
([§3.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=5)).
В [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык#gpt2-gpt3-lambada|разборе GPT-2]]
на одном отрывке видно, какое слово убирает фильтр и почему выбор
только из слов самого отрывка был бы другим ограничением.
У GPT-3 175B получилось 76,2% с обычным продолжением и 86,4% с
15 размеченными примерами. В последнем случае авторы также сменили
формат: вместо дописывания отрывка модель
[[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#lambada-gpt3-fill-blank|заполняла пропуск]].
С одним примером в этом формате получилось 72,5%: меньше, чем при
обычном продолжении. Поэтому разность 86,4 и 76,2 включает эффект
формата, а не только добавления примеров
([GPT-3, §3.1.2, табл. 3.2 и рис. 3.2](https://arxiv.org/html/2005.14165v4#S3.SS1.SSS2)).

В [таблице 3 LLaMA](https://arxiv.org/html/2302.13971v1#S2.T3)
GPT-3 175B и LLaMA 13B сравнивают без демонстраций. На HellaSwag
нужно выбрать правдоподобное продолжение: 78,9% у GPT-3 и 79,2%
у LLaMA. На PIQA, где нужно выбрать решение бытовой физической
задачи, порядок обратный: 81,0% и 80,1%. Это результаты из
публикаций, а не эксперимент, в котором меняли только размер модели.
На HellaSwag различалось даже [[00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning#gpt-llama-completion-scoring|правило ранжирования концовок]]:
GPT-3 нормировала оценку по токенам, LLaMA — по символам.
Из разницы в 0,3 пункта нельзя выделить вклад RoPE.

Размеры моделей при этом различаются примерно в 13,5 раза.
Грубая оценка вычислений предобучения $6ND$ учитывает и число весов
$N$, и число обработанных токенов $D$. Для GPT-3 175B при 300 млрд
токенов получается около $3.15\cdot10^{23}$ FLOP; для LLaMA 13B при
триллионе — $7.8\cdot10^{22}$ FLOP. Разрыв около четырёх раз, а не
13,5: меньшая модель обработала больше токенов.
Это оценка, а не замер времени на GPU.
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#training-and-inference-budget|Отдельный расчёт]] показывает, когда более долгое обучение меньшей модели окупается на выводе.

<a id="gpt-decoder-route-bpb"></a>

#### Когда перплексии нельзя сравнивать напрямую

У GPT-3 и первой LLaMA одинаковое контекстное окно, но
[[00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура#gpt-llama-tokenization|разные токенизаторы]].
Поэтому «3 ната на токен» у двух моделей не обязательно означают
одинаковые потери на одном и том же тексте. Единица в знаменателе
тоже изменилась: один токен может покрывать больше исходных байтов.

Возьмём четыре байта UTF-8: `cats`. Пусть первый токенизатор делит
строку на `cat` и `s`. Модель даёт первому токену вероятность
1/2, а второму после первого — 1/4. Итого $1/8$, или 3 бита на весь
текст. У второй учебной модели четыре токена, по одному на байт,
с условными вероятностями 1/2, 1/2, 4/5 и 5/8. Их произведение
тоже $1/8$. Перплексия при этом различается: $2^{3/2}\approx2.83$
у первой модели и $2^{3/4}\approx1.68$ у второй. Обе потратили
по $3/4=0.75$ бита на байт. Это условный расчёт, не прогон GPT или LLaMA.

Для корпуса складывают отрицательные логарифмы условных вероятностей
всех оцениваемых токенов и делят на число байтов исходного текста:

$$
\begin{aligned}
\mathrm{BPB}&=\frac{\sum_{t=1}^{T}-\log_2 p(x_t\mid x_{<t})}{B}\\
&=\frac{T}{B}\log_2(\mathrm{PPL}).
\end{aligned}
$$

Здесь $T$ — число оцениваемых токенов, $B$ — число байтов
соответствующего UTF-8-текста. Сумма берётся по одному разбиению,
которое вернул токенизатор. В статье GPT-2 таким же способом
приводят оценки к единицам конкретного набора: байтам, символам
или словам ([Radford et al., §3.1](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf#page=4)).
[Gao et al., §3.1–3.2](https://arxiv.org/html/2101.00027v1#S3)
выбрали байт и применили это правило к GPT-2 и GPT-3.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-14/pile-gpt2-gpt3-figure2.png|Восемь точек GPT-2 и GPT-3 на The Pile: по горизонтали число параметров в логарифмическом масштабе, по вертикали потери в битах на байт. У крайней левой точки около 1,23, у правой около 0,72.]]

*Leo Gao et al., [The Pile, рис. 2](https://arxiv.org/html/2101.00027v1#S3.F2),
[оригинальный PNG](https://arxiv.org/html/2101.00027v1/gpt3_zeroshot_pile_scalinglaw.png),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Рисунок без изменений.*

Каждая точка — одна модель. Крайние точки соответствуют GPT-2 Small
с 1,2253 бита на байт и GPT-3 Davinci с 0,7177
([табл. 2](https://arxiv.org/html/2101.00027v1#S3.T2)).
В пересчёте на 1000 байтов разница в потерях составляет около 508 бит.
График сравнивает качество предсказания текста The Pile.
Авторы считали
потери отдельно по документам; размеры моделей GPT-3 на оси взяты
по их предположению о соответствии имён API моделям статьи.
Точки на рисунке не показывают LLaMA. Чтобы добавить её, потребовалось
бы заново оценить тот же текст с согласованными правилами:
как задавать контекст, где начинать новый документ и какие токены
включать в сумму. BPB приводит к общей единице измерения,
но сам по себе этих различий не устраняет.

<a id="gpt-decoder-route-prompt-order"></a>

#### Что меняет порядок примеров

Lu et al. взяли четыре фиксированных примера и перебрали все $4!=24$
порядка. Для каждого порядка они ставили те же четыре примера перед
каждым новым текстом и считали долю верных ответов. Веса, состав
демонстраций и шаблон запроса оставались прежними. Четыре левые
позиции на рисунке относятся к GPT-2, остальные к GPT-3.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-route-2026-11-11/lu22-figure1.png|Две диаграммы размаха по порядкам четырёх примеров. На SST-2 у GPT-2 1.5B большой разброс, у GPT-3 175B малый. На Subj у GPT-3 175B разброс сохраняется.]]

*Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel и Pontus Stenetorp,
[Fantastically Ordered Prompts and Where to Find Them, §2](https://aclanthology.org/2022.acl-long.556/),
[рис. 1, с. 1 версии v2](https://arxiv.org/pdf/2104.08786v2#page=1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Обе панели из статьи, без перерисовки.*

На верхней панели, для тональности SST-2, у GPT-2 1.5B усы идут
примерно от 48 до 87% точности; у GPT-3 175B — примерно от 92 до 95%.
Линия внутри коробки показывает медиану по порядкам, а не среднее
по разным наборам демонстраций. На нижней панели та же GPT-3 175B
решает другую задачу, отличая субъективные высказывания от объективных.
Там усы охватывают примерно 56–85%. На SST-2 разброс у GPT-3 175B меньше, чем у GPT-2 1.5B;
на Subj у GPT-3 175B он по-прежнему велик.

По точности нельзя узнать, в какую сторону сместились ответы.
В другом опыте Zhao и коллег GPT-3 2.7B получала четыре размеченных отзыва
перед тестовым. На графике `P` и `N` — метки этих четырёх примеров
в порядке чтения. Сравните `N N P P` и `P P N N` в средней группе:
в обоих запросах по два примера каждого класса. Когда положительные
примеры стоят последними, модель выбирает `Positive` примерно для
трёх четвертей тестовых отзывов. Когда последними стоят отрицательные,
доля падает примерно до одной десятой. Высота столбца показывает долю
ответов `Positive` на валидационной выборке SST-2, а не точность
и не уверенность модели в одном ответе.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/zhao21c-figure4.png|GPT-3 2.7B чаще отвечает Positive при порядке N N P P, чем при P P N N, хотя в обоих запросах по две метки каждого класса.]]

*Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein и Sameer Singh,
[Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://proceedings.mlr.press/v139/zhao21c.html),
ICML 2021, [рис. 4, §4](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=4),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
График из статьи, без перерисовки.*

<a id="gpt-decoder-route-calibration"></a>

#### Как учесть склонность запроса к одной метке

На следующем графике каждый кружок — один из 25 случайно выбранных
тестовых отзывов. Его положение показывает оценку `Positive`, цвет —
правильный класс: красный для отрицательных, зелёный для положительных.
Вертикальная черта стоит на 0,5. Красные точки справа от неё —
ложноположительные ответы при выборе самой вероятной метки.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/zhao21c-figure5.png|Оценки Positive для 25 отзывов. Несколько отрицательных отзывов лежат правее порога 0,5 и ошибочно классифицируются как положительные.]]

*Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein и Sameer Singh,
[Calibrate Before Use, ICML 2021, PMLR 139](https://proceedings.mlr.press/v139/zhao21c.html),
[рис. 5, с. 5 PDF](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=5),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
График из статьи, без изменения подписей.*

Для этого запроса авторы получили 67% точности при пороге 0,5
и 94% при пороге 0,68, подобранном с использованием правильных меток
на валидации. Без этих меток нужен другой способ выбрать порог.

Чтобы оценить склонность к одной метке, оставим те же примеры,
их порядок и шаблон запроса, а вместо тестового отзыва подставим
`N/A`. Например, в запросе выше изменится только тестовая строка:
`Review: i liked it` станет `Review: N/A`, а `Sentiment:` останется пустым.
В отдельном двухпримерном запросе из [§5](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=5)
GPT-3 отдала `Positive` 61,8% вероятности после нормировки по двум
меткам. Это измерение на `N/A`, а не порог, подобранный к графику.
Если изменить примеры или их порядок, фоновую оценку нужно вычислить заново.

Пусть $p_y$ — доля вероятности метки $y$ среди допустимых меток на
реальном отзыве, а $b_y>0$ — та же доля на `N/A`. Контекстная калибровка
для классификации сравнивает отношения $p_y/b_y$. На самом `N/A`
они все равны единице. Для реального отзыва с условными оценками
0,60 и 0,40 исходный ответ был бы `Positive`. После поправки на фон
0,618 и 0,382 получаем $0.60/0.618\approx0.971$ и
$0.40/0.382\approx1.047$: выбираем `Negative`. При двух классах
это то же самое, что поднять порог с 0,5 до 0,618:
$p_+/b_+>(1-p_+)/(1-b_+)$ равносильно $p_+>b_+$.

В экспериментах авторы усредняли фон по трём входам: `N/A`,
`[MASK]` и пустой строке. При четырёх примерах средняя точность SST-2
у GPT-2 XL изменилась с 64,9% до 73,8%, у GPT-3 2.7B — с 59,1% до
79,9%. У GPT-3 175B с восемью примерами результат слегка снизился:
95,6% → 95,3%. Во всех этих сравнениях веса, шаблон и наборы
примеров для двух вариантов оставались прежними
([табл. 1](https://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf#page=7),
[табл. 4 приложения, с. 12](https://arxiv.org/pdf/2102.09690v2#page=12)).
Поправка меняет правило выбора метки. Она не устанавливает,
как часто прогнозы с оценкой 0,8 действительно верны.

<a id="gpt-decoder-route-end"></a>

[[#gpt-decoder-route|К четырём главам]] · [[#полный-маршрут|Основной курс]]
