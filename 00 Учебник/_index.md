---
title: Учебник по NLP и LLM
type: textbook-chapter
status: active
last_updated: 2026-07-20
---

# Учебник по NLP и LLM

Последовательный визуальный курс: от производной и представления текста до
современных языковых моделей, обучения рассуждению, поиска и агентных систем.

> [!info] Как устроена база
> Учебник задаёт порядок чтения. [[02 Areas/ML & DL/01 Справочник/_index|Справочник]]
> хранит точные определения, [[02 Areas/ML & DL/02 Атлас моделей/_index|Атлас]]
> сравнивает модели, а [[02 Areas/ML & DL/03 Исследовательские линии/_index|обзоры направлений]]
> отслеживают свежие papers.

## Полный маршрут

Перед началом: [[02 Areas/ML & DL/00 Учебник/00 Как пользоваться учебником]].

### I. Математические и ML-основания

1. [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры]]
2. [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/02 Производная и градиент]]
3. [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/03 Вероятность, правдоподобие и логарифм]]
4. [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/04 Функции потерь]]
5. [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/05 Train, validation и test]]
6. [[02 Areas/ML & DL/00 Учебник/00 Математические и ML-основания/06 Переобучение, регуляризация и оценивание]]

### II. Классическое машинное обучение

Этот блок устраняет скачок от общих функций потерь сразу к нейронным сетям.
Он основан на полном оригинальном курсе Inria/scikit-learn и содержит не только
определения, но исходные лекции, notebooks, упражнения, решения и 212 рисунков.

- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/00 Карта модуля|Карта модуля и источники]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/01 Задача обучения и predictive pipeline]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/02 Линейная регрессия и классификация]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/03 Деревья решений]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/04 Bagging, random forest и gradient boosting]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/05 Оценивание, кросс-валидация и выбор порога]]
- [[02 Areas/ML & DL/00 Учебник/01 Классическое машинное обучение/06 Кластеризация и её ограничения]]

### III. Нейронные сети

7. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP]]
8. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Функции активации]]
9. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation|Вычислительный граф и backpropagation]]
10. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/02 Оптимизация и стабильность обучения|SGD, Momentum, Adam и расписания шага]]
11. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections]]
12. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация]]
13. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet]]
14. [[02 Areas/ML & DL/00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE]]

### IV. Текст до Transformer

15. [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 Представление текста числами]]
16. [[02 Areas/ML & DL/00 Учебник/03 Языковое моделирование/00 N-граммная языковая модель]]
17. [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/01 От слов к embeddings|Word2Vec, GloVe и распределительные представления]]
18. [[02 Areas/ML & DL/00 Учебник/02 Представление текста и токенизация/02 BPE, WordPiece и Unigram]]
19. [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT]]
20. [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/02 LSTM и GRU]]
21. [[02 Areas/ML & DL/00 Учебник/04 RNN, LSTM и Seq2Seq/03 Seq2Seq и bottleneck фиксированного вектора]]
22. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/01 От Seq2Seq к Transformer|Attention для перевода]]

### V. Transformer, BERT и GPT

23. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V]]
24. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/Masking, multi-head и формы тензоров|Маскирование, головы и формы тензоров]]
25. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/04 Позиционная информация]]
26. [[02 Areas/ML & DL/00 Учебник/05 Attention и Transformer/03 Полный Transformer|Полный encoder-decoder Transformer]]

Перед BERT и GPT: [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/01 Три архитектурных паттерна|как из общего Transformer получаются encoder-only, decoder-only и encoder–decoder модели]].
27. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/03 BERT, RoBERTa и DeBERTa]]
28. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/04 GPT-1 — генеративное предобучение]]
29. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык]]
30. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning]]
31. [[02 Areas/ML & DL/00 Учебник/06 Encoder, Decoder и Encoder-Decoder/07 T5 — text-to-text Transformer]]

### VI. Анатомия современной LLM

32. [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/02 Современный decoder block]]
33. [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual]]
34. [[02 Areas/ML & DL/00 Учебник/07 Анатомия современной LLM/04 RoPE]]
35. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/01 MHA, MQA и GQA]]
36. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/02 MLA и сжатие KV-cache]]
37. [[02 Areas/ML & DL/00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание]]
38. [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating]]
39. [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/02 Mixture of Experts — routing, capacity и serving]]
40. [[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры]]

После механизмов удобно пройти [[02 Areas/ML & DL/02 Атлас моделей/_index|атлас семейств]],
чтобы увидеть, как они комбинируются в Llama, Qwen, DeepSeek, GLM, Kimi и
других моделях.

### VII. Вычислительные основы ML-систем

Этот блок переводит архитектуру модели в измеримые системные требования. Главы
нужно читать по порядку: системная постановка задаёт ограничения, GPU и roofline
объясняют пределы исполнения, арифметика и численные форматы дают бюджет шага,
а input pipeline и profiling учат находить фактическое узкое место.

1. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/01 Модель как часть системы]]
2. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти]]
3. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/03 Измерение производительности и roofline]]
4. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE]]
5. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision]]
6. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/06 Data pipeline, padding и packing]]
7. [[02 Areas/ML & DL/00 Учебник/10 ML Systems/07 Profiling ML-нагрузки]]

### VIII. Обучение LLM

41. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных]]
42. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/42 Next-token prediction]]
43. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/43 Scaling laws]]
44. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/44 Distributed training и mixed precision]]
45. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/01 SFT и instruction data]]
46. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/02 Preference data]]
47. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/03 Reward modeling]]
48. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/04 Policy gradient и PPO для LLM]]
49. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/05 DPO]]
50. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers]]
51. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/07 GRPO и DeepSeek-R1]]
52. [[02 Areas/ML & DL/00 Учебник/12 Post-training и Alignment/08 Reasoning distillation]]

После обучения reasoning-моделей находится отдельная
сквозная глава: [[02 Areas/ML & DL/00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|как распределять дополнительные вычисления во время ответа]].
Она связывает RLVR, verifiers, best-of-N, последовательное исправление и поиск,
не вводя ещё один номер в основной программе.

53. [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы]]

### IX. Эксплуатация и системы вокруг модели

54. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/54 Декодирование и выбор следующего токена]]
55. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55a Физика LLM inference — prefill, decode и roofline|Физика LLM inference: prefill, decode и roofline]]
    - 55.1. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55 KV-cache, пакетирование и PagedAttention|KV-cache, пакетирование и PagedAttention]]
    - 55.2. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55b Scheduling — continuous batching, chunked prefill и prefix caching|Scheduling: continuous batching, chunked prefill и prefix caching]]
    - 55.3. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer|Serving engines: vLLM, SGLang, TensorRT-LLM и FlashInfer]]
56. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/56 FlashAttention]]
57. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей]]
58. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58 Спекулятивное декодирование]]
    - 58.1. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a Распределённый inference и disaggregated serving|Параллелизм и коллективные операции]]
    - 58.2. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode|Раздельное обслуживание prefill и decode]]
    - 58.3. [[02 Areas/ML & DL/00 Учебник/14 Inference и оптимизация/58b Benchmarking, SLO и эксплуатация inference|Benchmarking, SLO и эксплуатация inference]]
59. [[02 Areas/ML & DL/00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация]]
60. [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning]]
61. [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid]]
62. [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction]]
63. [[02 Areas/ML & DL/00 Учебник/15 Embeddings, Retrieval и RAG/63 RAG — полный конвейер]]
64. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64 Мультимодальные модели|От ViT и CLIP к мультимодальной языковой модели]]
    - 64.1. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64a Connectors и fusion|Connectors и fusion: projector, Q-Former, resampler и cross-attention]]
    - 64.2. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64b Разрешение, tiling и пространственные позиции|Разрешение, tiling и пространственные позиции]]
    - 64.3. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64c Обучение VLM — alignment, instruction tuning и данные|Обучение VLM: alignment, instruction tuning и данные]]
    - 64.4. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64d Документы, OCR и visual grounding|Документы, OCR и visual grounding]]
    - 64.5. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели|Видео, аудио и omni-модели]]
    - 64.6. [[02 Areas/ML & DL/00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM|Оценивание, отказы и serving VLM]]
65. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/65 Tool use — от вызова функции к действию]]
66. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/66 Agent harness и context engineering]]
67. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов]]
68. [[02 Areas/ML & DL/00 Учебник/17 Tools и Agents/68 Оценивание агентных систем]]

## Другие режимы

- [[02 Areas/ML & DL/01 Справочник/_index|Справочник]]
- [[02 Areas/ML & DL/02 Атлас моделей/_index|Атлас моделей]]
- [[02 Areas/ML & DL/03 Исследовательские линии/_index|Обзоры направлений]]
- [[02 Areas/ML & DL/04 Вопросы/_index|Вопросы]]
- [[02 Areas/ML & DL/05 Источники/_index|Источники]]
- [[02 Areas/ML & DL/06 Практика/_index|Практика]]
