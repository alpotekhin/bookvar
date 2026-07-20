---
title: Редакционная матрица Bookvar
type: textbook-chapter
status: editorial
last_updated: 2026-07-20
---

# Редакционная матрица Bookvar

Этот документ связывает утверждённую программу с фактическими страницами.
Он не публикуется в читательской навигации. Строка считается закрытой только
тогда, когда существует полноценная глава, проверены её зависимости, источники,
иллюстрации и маршрут на сайте.

## Сравнительные карты источников

Матрица отвечает на вопрос «какая глава закрывает тему», а три source-map —
«из каких готовых объяснений и иллюстраций она собирается»:

- [[02 Areas/ML & DL/05 Источники/Source maps/Темы 01–18|темы 1–18]];
- [[02 Areas/ML & DL/05 Источники/Source maps/Темы 19–40|темы 19–40]];
- [[02 Areas/ML & DL/05 Источники/Source maps/Темы 41–68|темы 41–68]].

Статус главы нельзя повысить до `canonical` только на основании объёма текста.
Для каждого смыслового фрагмента сначала выбираются основной курс или книга,
первичный источник для проверки и готовая figure/table/slide. Если подходящего
визуального материала пока нет, это незакрытый пробел, а не повод автоматически
создавать собственную схему.

## Обозначения

- `принято` — глава существует по нормативному маршруту, сопоставлена с
  source-map и прошла содержательный, визуальный и технический аудит;
- редакционные ограничения конкретных готовых материалов фиксируются в
  source-map и реестре происхождения, а не скрываются заменяющей их авторской
  схемой.

## I. Математические и ML-основания

| № | Глава | Текущее покрытие | Действие | Обязательный визуал |
|---:|---|---|---|---|
| 1 | Векторы, матрицы и тензоры | `00 Математические…/01…` | принято | геометрия вектора; формы тензоров; broadcasting |
| 2 | Производная и градиент | `00 Математические…/02…` | принято | касательная; поверхность и градиент; вычислительный граф |
| 3 | Вероятность, правдоподобие и логарифм | `00 Математические…/03…` | принято | вероятность/плотность; likelihood; log-sum |
| 4 | Функции потерь | `00 Математические…/04…` | принято | MSE и cross-entropy; decision boundary; градиенты |
| 5 | Train, validation и test | `00 Математические…/05…` | принято | data split; leakage; model-selection loop |
| 6 | Переобучение, регуляризация и оценивание | `00 Математические…/06…` | принято | train/val curves; bias–variance; regularizers |

## II. Нейронные сети

| № | Глава | Текущее покрытие | Действие | Обязательный визуал |
|---:|---|---|---|---|
| 7 | Нейрон и MLP | `01 Основы…/01 Нейрон и MLP` | принято | neuron; layer shapes; decision regions |
| 8 | Функции активации | `01 Основы…/02 Функции активации` | принято | sigmoid/tanh/ReLU/ELU/GELU; gradient comparison |
| 9 | Вычислительный граф и backpropagation | `01 Основы…/01 Нейрон, градиент…` | принято | local derivatives; reverse pass; shared parameter |
| 10 | SGD, Momentum, Adam и schedules | `01 Основы…/02 Оптимизация…` | принято | optimizer trajectories; momentum; LR schedules |
| 11 | Инициализация, нормализация и residual connections | `01 Основы…/05…` | принято | activation variance; norm axes; residual stream |
| 12 | Dropout и регуляризация | `01 Основы…/06…` | принято | dropout train/inference; weight decay; augmentation |
| 13 | CNN: LeNet → ResNet | `01 Основы…/07…` | принято | convolution; receptive field; residual block; timeline |
| 14 | Autoencoder и VAE | `01 Основы…/08…` | принято | bottleneck; latent space; reparameterization |

## III. Текст до Transformer

| № | Глава | Текущее покрытие | Действие | Обязательный визуал |
|---:|---|---|---|---|
| 15 | Представление текста числами | `02…/01 Представление текста числами` | принято | one-hot → dense; embedding table; context |
| 16 | N-граммная языковая модель | `03…/00 N-граммная языковая модель` | принято | count table; Markov window; smoothing |
| 17 | Word2Vec, GloVe и распределительные представления | `02…/01 От слов к embeddings` | принято | skip-gram/CBOW; vector arithmetic; co-occurrence |
| 18 | BPE, WordPiece и Unigram | `02…/02…` | принято | merge process; tokenizer comparison; multilingual example |
| 19 | RNN и BPTT | `04…/01 RNN и BPTT` | принято | recurrent cell; unrolling; gradient path |
| 20 | LSTM и GRU | `04…/02 LSTM и GRU` | принято | cell-state highway; gates step by step; GRU comparison |
| 21 | Seq2Seq и fixed-vector bottleneck | `04…/03 Seq2Seq…` | принято | encoder/decoder states; bottleneck; teacher forcing |
| 22 | Attention для перевода | `05…/01…` | принято | alignment matrix; scoring; context per output token |

## IV. Transformer, BERT и GPT

| № | Глава | Текущее покрытие | Действие | Обязательный визуал |
|---:|---|---|---|---|
| 23 | Self-attention: Q, K, V | `05…/02…` | принято | existing three-stage set + numerical matrix |
| 24 | Masking, heads и tensor shapes | часть той же главы | принято | causal mask; head split/merge; complete shapes |
| 25 | Позиционная информация | `05…/04 Позиционная информация` | принято | sinusoid; relative bias; rotation preview |
| 26 | Полный encoder-decoder Transformer | `05…/03 Полный Transformer` | принято | complete stack; encoder block; decoder block; data flow |
| 27 | BERT, RoBERTa, DeBERTa | `06…/03 BERT, RoBERTa и DeBERTa` | принято | MLM; bidirectional context; architecture diffs |
| 28 | GPT-1 | `06…/04 GPT-1` | принято | pretrain/fine-tune pipeline; decoder block; objectives |
| 29 | GPT-2 | `06…/05 GPT-2` | принято | autoregression; zero-shot task formatting; scale diff |
| 30 | GPT-3 | `06…/06 GPT-3` | принято | in-context prompt; scaling table; few-shot evaluation |
| 31 | T5 и text-to-text | `06…/07 T5` | принято | unified task format; span corruption; encoder-decoder |

Страница `06…/01 Три архитектурных паттерна` — обязательное продолжение темы 26
и сравнительный вход в темы 27–31. Она не повторяет полный Transformer, а
сопоставляет режимы видимости и задачи encoder-only, decoder-only и
encoder–decoder моделей. Её читательский маршрут —
`textbook/transformer/encoder-decoder-patterns`. Короткая прежняя страница
`06…/02 Encoder-Decoder
Transformer` исключена из читательского маршрута как дубликат.

## V. Анатомия современной LLM

| № | Глава | Текущее покрытие | Действие | Обязательный визуал |
|---:|---|---|---|---|
| 32 | Современный decoder block | `07…/02 Современный decoder block` | принято | annotated block; residual stream; tensor shapes |
| 33 | Pre-norm, RMSNorm, SwiGLU и residual stream | `07…/03 Pre-norm…` | принято | pre/post norm; RMSNorm; gated FFN |
| 34 | RoPE и варианты | `07…/04 RoPE` | принято | 2D rotation; relative phase; frequency bands |
| 35 | MHA, MQA и GQA | `08…/01…` | принято | KV sharing; memory table; quality/throughput trade-off |
| 36 | MLA и KV compression | `08…/02…` | принято | latent projection; cache contents; comparison with GQA |
| 37 | Длинный контекст | `08…/03 Длинный контекст…` | принято | context extension methods; train/inference mismatch |
| 38 | Dense FFN | `09…/01 Dense FFN…` | принято | token-wise FFN; expansion; activation/gating |
| 39 | MoE | `09…/02 Mixture of Experts…` | принято | routing; capacity; load balance; expert parallelism; serving |
| 40 | Mamba, RWKV, RetNet и гибриды | `09…/03 Mamba, RWKV…` | принято | recurrence/convolution/attention comparison; scan |

Страница `07…/01 LLaMA как базовая архитектура` служит сборочным примером для
тем 32–34: на одной опубликованной конфигурации она соединяет decoder block,
RMSNorm, SwiGLU и RoPE перед раздельным разбором механизмов. Её читательский
маршрут — `textbook/modern-llm/llama-architecture`.

## VI. Атлас семейств

Атлас не повторяет главы 28–40. Для каждого семейства обязательны: timeline,
карточки релизов, архитектурный diff, pre-training, post-training, inference,
уровень доказательности и первичные источники.

| Семейства | Текущее покрытие | Действие |
|---|---|---|
| GPT, BERT, T5 | канонические страницы семейств | принято |
| Llama, Qwen, DeepSeek, GLM, Kimi | релизы и архитектурные изменения проверены | принято |
| Mistral/Mixtral, Gemma, Phi, Falcon, Yi | страницы и временные линии проверены | принято |
| InternLM, Baichuan, Jamba, Nemotron | страницы и временные линии проверены | принято |
| мультимодальные семейства | каноническая сравнительная карта | принято |

## VII. Обучение LLM

| № | Глава | Текущее покрытие | Действие | Обязательный визуал |
|---:|---|---|---|---|
| 41 | Сбор, очистка и смешивание данных | `11…/41 Сбор, очистка…` | принято | data pipeline; filtering; mixture weights |
| 42 | Next-token pre-training | `11…/42 Next-token prediction` | принято | shifted targets; packed sequences; loss surface |
| 43 | Scaling laws | `11…/43 Scaling laws` | принято | Kaplan/Chinchilla curves; compute allocation |
| 44 | Distributed training и precision | `11…/44 Distributed training…` | принято | data/tensor/pipeline parallel; memory; collectives |
| 45 | Instruction data и SFT | `12…/01…` | принято | chat template; token masking; curriculum |
| 46 | Preference data | `12…/02…` | принято | pair collection; annotator agreement; data pathologies |
| 47 | Reward modeling | `12…/03…` | принято | pairwise loss; RM scoring; overoptimization |
| 48 | Policy gradient и PPO | `12…/04…` | принято | rollout; advantages; clipped objective; KL loop |
| 49 | DPO | `12…/05…` | принято | preference likelihood ratios; DPO vs PPO |
| 50 | RLVR и verifiers | `12…/06…` | принято | verifier pipeline; outcome/process reward; failure modes |
| 51 | GRPO и DeepSeek-R1 | `12…/07…` | принято | group advantages; R1 stages; ablations |
| 52 | Reasoning distillation | `12…/08…` | принято | teacher traces; filtering; student training |
| 53 | Synthetic data и curricula | `11…/53 Синтетические данные…` | принято | generation/filtering loop; self-improvement risks |

Сквозная глава `13 Reasoning и Test-time Compute/01 Test-time compute.md`
обслуживает темы 50–52 и соединяет обучение reasoning-моделей с эксплуатацией:
она рассматривает best-of-N, verifiers, последовательные revisions и поиск при
фиксированном бюджете. Это необходимое продолжение, а не отдельная 69-я тема.
Её читательский маршрут — `textbook/reasoning/test-time-compute`; в sidebar она
следует сразу после темы 52 и до перехода к синтетическим данным.

## VIII. Эксплуатация и системы вокруг модели

| № | Глава | Текущее покрытие | Действие | Обязательный визуал |
|---:|---|---|---|---|
| 54 | Декодирование и sampling | `14…/54 Декодирование…` | принято | distribution transforms; beam/tree; degeneration |
| 55 | KV-cache, batching и PagedAttention | `14…/55 KV-cache…` | принято | prefill/decode; cache growth; paging; scheduler |
| 56 | FlashAttention | `14…/56 FlashAttention` | принято | HBM/SRAM traffic; tiling; exactness |
| 57 | Quantization | `14…/57 Квантизация…` | принято | ranges; per-channel/group; weight/activation/KV |
| 58 | Speculative decoding | `14…/58 Спекулятивное декодирование` | принято | draft/verify/accept loop; expected speedup |
| 59 | Evaluation и contamination | `18…/59 Оценивание моделей…` | принято | evaluation stack; contamination; confidence intervals |
| 60 | Embeddings и metric learning | `15…/60 Embeddings…` | принято | bi-encoder; contrastive geometry; hard negatives |
| 61 | Retrieval | `15…/61 Retrieval…` | принято | indexing/query path; sparse/dense; ANN |
| 62 | Reranking | `15…/62 Reranking…` | принято | retrieve-then-rerank; cross encoder; late interaction |
| 63 | RAG | `15…/63 RAG — полный конвейер` | принято | ingestion; retrieval; generation; evaluation; operations |
| 64 | Multimodal models | `16…/64 Мультимодальные модели` | принято | projector/cross-attention; resolution; audio/video |
| 65 | Tool use | `17…/65 Tool use…` | принято | schema → call → result; training data; errors |
| 66 | Agent harness и context engineering | `17…/66 Agent harness…` | принято | harness layers; context assembly; execution loop |
| 67 | Memory, planning и orchestration | `17…/67 Память…` | принято | memory types; planner/executor; multi-agent boundaries |
| 68 | Evaluation агентов | `17…/68 Оценивание…` | принято | trajectory grading; environment; reproducibility |

## Реестр основных читательских маршрутов

Это нормативное соответствие строк программы и reader routes. Продолжения и
служебные страницы перечислены отдельно ниже и не заменяют основной маршрут.

| № | Основной reader route |
|---:|---|
| 1 | `textbook/ml-foundations/tensors` |
| 2 | `textbook/ml-foundations/derivatives-gradients` |
| 3 | `textbook/ml-foundations/probability-likelihood-logarithms` |
| 4 | `textbook/ml-foundations/loss-functions` |
| 5 | `textbook/ml-foundations/train-validation-test` |
| 6 | `textbook/ml-foundations/generalization-regularization` |
| 7 | `textbook/neural-networks/mlp` |
| 8 | `textbook/neural-networks/activation-functions` |
| 9 | `textbook/neural-networks/backpropagation` |
| 10 | `textbook/neural-networks/optimizers-schedules` |
| 11 | `textbook/neural-networks/initialization-normalization-residuals` |
| 12 | `textbook/neural-networks/dropout-regularization` |
| 13 | `textbook/neural-networks/cnn-lenet-resnet` |
| 14 | `textbook/neural-networks/autoencoder-vae` |
| 15 | `textbook/text-representation/numerical-representation` |
| 16 | `textbook/language-modeling/ngram` |
| 17 | `textbook/text-representation/word2vec-glove` |
| 18 | `textbook/text-representation/tokenization` |
| 19 | `textbook/recurrent-networks/rnn-bptt` |
| 20 | `textbook/recurrent-networks/lstm-gru` |
| 21 | `textbook/recurrent-networks/seq2seq-bottleneck` |
| 22 | `textbook/transformer/from-seq2seq-to-transformer` |
| 23 | `textbook/transformer/self-attention` |
| 24 | `textbook/transformer/masking-heads-shapes` |
| 25 | `textbook/transformer/positional-information` |
| 26 | `textbook/transformer/full-transformer` |
| 27 | `textbook/models/bert-roberta-deberta` |
| 28 | `textbook/models/gpt-1` |
| 29 | `textbook/models/gpt-2` |
| 30 | `textbook/models/gpt-3` |
| 31 | `textbook/models/t5` |
| 32 | `textbook/modern-llm/decoder-block` |
| 33 | `textbook/modern-llm/norm-gating-residual` |
| 34 | `textbook/modern-llm/rope` |
| 35 | `textbook/modern-llm/mha-mqa-gqa` |
| 36 | `textbook/modern-llm/mla-kv-compression` |
| 37 | `textbook/modern-llm/long-context` |
| 38 | `textbook/modern-llm/dense-ffn` |
| 39 | `textbook/moe/mixture-of-experts` |
| 40 | `textbook/modern-llm/sequence-model-alternatives` |
| 41 | `textbook/pretraining/data-pipeline` |
| 42 | `textbook/pretraining/next-token-objective` |
| 43 | `textbook/pretraining/scaling-laws` |
| 44 | `textbook/pretraining/distributed-systems` |
| 45 | `textbook/post-training/sft-instruction-data` |
| 46 | `textbook/post-training/preference-data` |
| 47 | `textbook/post-training/reward-modeling` |
| 48 | `textbook/post-training/ppo` |
| 49 | `textbook/post-training/dpo` |
| 50 | `textbook/post-training/rlvr-verifiers` |
| 51 | `textbook/post-training/grpo-deepseek-r1` |
| 52 | `textbook/post-training/reasoning-distillation` |
| 53 | `textbook/pretraining/synthetic-data-curricula` |
| 54 | `textbook/inference/decoding` |
| 55 | `textbook/inference/kv-cache-batching-pagedattention` |
| 56 | `textbook/inference/flashattention` |
| 57 | `textbook/inference/quantization` |
| 58 | `textbook/inference/speculative-decoding` |
| 59 | `textbook/evaluation/models-contamination` |
| 60 | `textbook/retrieval/embeddings-metric-learning` |
| 61 | `textbook/retrieval/sparse-dense-hybrid` |
| 62 | `textbook/retrieval/reranking` |
| 63 | `textbook/rag/full-pipeline` |
| 64 | `textbook/multimodal/models` |
| 65 | `textbook/agents/tool-use` |
| 66 | `textbook/agents/harness-context-engineering` |
| 67 | `textbook/agents/memory-planning-orchestration` |
| 68 | `textbook/agents/evaluation` |

## Страницы, которые не должны находиться в маршруте читателя

Две служебные страницы верхнего уровня являются оболочкой reader route, а не
строками программы: `textbook/index` открывает учебник, а
`textbook/how-to-use` объясняет навигацию. Вместе с тремя явно описанными выше
продолжениями это исчерпывает пять опубликованных страниц сверх 68 основных
глав.

- `Карта крупных учебных модулей.md`;
- `Покрытие программы.md`;
- `Редакционный аудит 2026-07-17.md`;
- `Аудит иллюстраций 2026-07-17.md`;
- `05 Attention и Transformer/00 Источники и визуальный стандарт.md`;
- все `00 Карта модуля и источники.md` после появления настоящих оглавлений;
- старые `01 SFT, RLHF и DPO.md` и `02 RLVR, reasoning и distillation.md`.

Они остаются редакционными документами или удаляются после переноса полезных
сведений, но не маскируются под учебные главы.

## Минимальный пакет источников для каждой строки

1. Одно основное учебное объяснение: курс, глава книги или качественный
   технический разбор.
2. Не менее одного первичного источника для архитектурного утверждения.
3. Официальная реализация или документация, когда технология имеет существенную
   инженерную сторону.
4. Готовая учебная визуализация из курса, книги, статьи или первоисточника,
   выбранная после сравнения нескольких кандидатов. Собственная схема допустима
   только тогда, когда документированный поиск не дал пригодного готового
   изображения; причина исключения и просмотренные источники фиксируются в
   редакционной заметке.
5. Запись о правах и модификациях для каждого локально опубликованного рисунка.

## Критерий двустороннего соответствия

- у каждой из 68 строк есть ровно один основной маршрут чтения;
- каждая опубликованная учебная страница обслуживает одну или несколько строк
  матрицы и не является редакционной заглушкой;
- справочник раскрывает механизмы, но не дублирует учебное повествование;
- атлас описывает релизы как изменения относительно базовых механизмов;
- новые papers сначала обновляют source note, research line, concept и model
  release, а учебник — только после стабилизации знания.
