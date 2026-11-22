# Покрытие NLP в основном курсе

Срез от 22 ноября 2026 года. В оглавлении 91 глава основного курса и
68 глав углублённых модулей. Исходная перестройка маршрута зафиксирована
в [журнале навигации](2026-11-21-linear-nlp-navigation.json). После неё
между токенизацией и RNN добавлена глава о морфологии, частях речи и
разметке последовательностей. Её пример связывает леммы и признаки с
контекстным выбором меток; отдельно разобраны оценивание и ошибки
сегментации. Следующая глава разбирает предложение с двумя возможными
присоединениями оборота: сначала строит дерево составляющих с помощью CKY,
затем восстанавливает дуги зависимостей стековым алгоритмом и считает ошибки.
После BERT добавлена глава о распознавании именованных
сущностей: BIO, выравнивание слов и субтокенов, восстановление
интервалов и оценка по полному совпадению границ и типа. На той же
ветке маршрута добавлена классификация документов: от TF-IDF,
логистической регрессии и наивного Байеса до fastText и BERT на
одном примере. Следующая глава ведёт от упоминаний в двух предложениях
к группам, внешним идентификаторам, отношениям и записи о событии.
Ошибки границ, группировки, поиска и извлечения оцениваются отдельно.
За T5 следует глава о переводе: как получить пары предложений, обучить
декодер, выбрать гипотезу и проверить, сохранился ли смысл.

Для проверки программы использовано оглавление [Jurafsky и Martin, Speech and
Language Processing, редакция от 19 августа 2026 года](https://web.stanford.edu/~jurafsky/slp3/).
Для обучения LLM, перевода и архитектур дополнительные ориентиры —
[CS336](https://cs336.stanford.edu/) и [CS224N](https://web.stanford.edu/class/cs224n/).
Материалы Jurafsky и Martin помечены авторами как *all rights reserved*;
ссылки на главы не являются разрешением на публикацию их рисунков.

| Что проверить | Что уже есть в Bookvar | Какого разбора пока не видно в оглавлении |
| --- | --- | --- |
| Суммаризация и ответы на вопросы | В T5 и GPT есть примеры генерации резюме и ответов; в RAG оценивают поиск источников и проверяют ответы. | Сопоставить извлекающие и генеративные методы на общих примерах. Для резюме разобрать покрытие и верность фактам отдельно от совпадения слов; для QA — поиск ответа, оценку диапазона и случаи без ответа. |
| Речь ([SLP, гл. 15–17](https://web.stanford.edu/~jurafsky/slp3/)) | В мультимодальном разделе уже есть спектрограммы, аудиоэнкодеры, токенизация звука и генерация речи. CER и WER разбираются в главе про OCR. | Сверить, достаточно ли этого для понимания ASR и TTS. Не считать упоминание WER объяснением обучения распознавателя. |

Границы разборов уточняются при полном чтении глав и источников. Для новых
разделов нужен разобранный пример, а для заимствованных рисунков —
разрешение и подпись с источником. Новые пункты войдут в оглавление только
после подготовки самих глав.

Редакторский проход зафиксирован для глав
[«Векторы, матрицы и тензоры»](2026-11-21-tensor-editorial-pass.json),
[«Производная и градиент»](2026-11-21-derivatives-editorial-pass.json),
[«Вероятность, правдоподобие и логарифм»](2026-11-21-probability-editorial-pass.json),
[«Функции потерь»](2026-11-21-loss-editorial-pass.json),
[«Train, validation и test»](2026-11-21-data-split-editorial-pass.json),
[«Переобучение, регуляризация и оценивание»](2026-11-21-generalization-editorial-pass.json),
[«Нейрон и многослойный перцептрон»](2026-11-21-mlp-editorial-pass.json),
[«Функции активации»](2026-11-21-activations-editorial-pass.json),
[«Вычислительный граф и обратное распространение»](2026-11-21-backprop-editorial-pass.json),
[«SGD, Momentum, Adam и расписания скорости обучения»](2026-11-21-optimizers-editorial-pass.json),
[«Инициализация, нормализация и остаточные связи»](2026-11-21-initialization-editorial-pass.json),
[«Dropout и регуляризация»](2026-11-21-dropout-editorial-pass.json),
[«Аугментация данных»](2026-11-21-augmentation-editorial-pass.json),
[«CNN — от свёртки до ResNet»](2026-11-21-cnn-editorial-pass.json),
[«Представление текста числами»](2026-11-21-text-representation-editorial-pass.json),
[«N-граммная языковая модель»](2026-11-21-ngram-editorial-pass.json),
[«Вероятность текста и предсказание следующего токена»](2026-11-21-next-token-editorial-pass.json),
[«Word2Vec, GloVe и распределительные представления»](2026-11-21-word2vec-glove-editorial-pass.json),
[«BPE, WordPiece и Unigram»](2026-11-21-tokenization-editorial-pass.json),
[«Морфология и разметка последовательностей»](2026-11-22-morphology-pos-editorial-pass.json),
[«Синтаксический разбор»](2026-11-22-syntactic-parsing-editorial-pass.json),
[«RNN и BPTT»](2026-11-21-rnn-bptt-editorial-pass.json),
[«LSTM и GRU»](2026-11-21-lstm-gru-editorial-pass.json),
[«Seq2Seq с фиксированным контекстом»](2026-11-21-seq2seq-editorial-pass.json),
[«Attention для перевода»](2026-11-21-attention-translation-editorial-pass.json),
[«Self-Attention — Q, K, V»](2026-11-21-self-attention-editorial-pass.json),
[«Маски, несколько голов и формы тензоров»](2026-11-21-masking-heads-editorial-pass.json),
[«Позиционная информация»](2026-11-21-positional-information-editorial-pass.json),
[«Полный Transformer: энкодер и декодер»](2026-11-21-full-transformer-editorial-pass.json),
[«Три архитектуры Transformer»](2026-11-21-architecture-patterns-editorial-pass.json),
[«BERT, RoBERTa и DeBERTa»](2026-11-21-bert-family-editorial-pass.json),
[«Распознавание именованных сущностей»](2026-11-22-ner-editorial-pass.json),
[«Классификация документов»](2026-11-22-document-classification-editorial-pass.json),
[«Извлечение информации из документов»](2026-11-22-information-extraction-editorial-pass.json),
[«T5 — text-to-text Transformer»](2026-11-21-t5-editorial-pass.json),
[«Машинный перевод»](2026-11-22-machine-translation-editorial-pass.json),
[«GPT-1 — генеративное предобучение»](2026-11-21-gpt1-editorial-pass.json),
[«GPT-2 — задачи как продолжение текста»](2026-11-21-gpt2-editorial-pass.json),
[«GPT-3 — масштабирование и in-context learning»](2026-11-21-gpt3-editorial-pass.json),
[«LLaMA как базовая архитектура»](2026-11-21-llama-editorial-pass.json),
[«Современный decoder block»](2026-11-21-decoder-block-editorial-pass.json),
[глава о нормализации и остаточных связях](2026-11-21-norm-gating-editorial-pass.json),
[«RoPE»](2026-11-21-rope-editorial-pass.json),
[«Dense FFN — расширение размерности и SwiGLU»](2026-11-21-dense-ffn-editorial-pass.json),
[«Transformer с нуля — формы, параметры и стоимость»](2026-11-21-transformer-shapes-editorial-pass.json),
[«MHA, MQA и GQA»](2026-11-21-gqa-editorial-pass.json),
[«MLA и сжатие KV-кэша»](2026-11-21-mla-editorial-pass.json),
[«Длинный контекст»](2026-11-21-long-context-editorial-pass.json),
[«Mixture of Experts»](2026-11-21-moe-editorial-pass.json),
[«SSM, Mamba, линейное внимание и гибриды»](2026-11-21-sequence-models-editorial-pass.json),
[«Данные и предобучение»](2026-11-22-data-pretraining-editorial-pass.json),
[«Сбор, очистка и смеси данных»](2026-11-22-data-collection-editorial-pass.json),
[«Дедупликация, PII и контроль качества корпуса»](2026-11-22-corpus-dedup-editorial-pass.json),
[«Целевая функция предобучения»](2026-11-22-ntp-objective-editorial-pass.json),
[«Законы масштабирования»](2026-11-22-scaling-laws-editorial-pass.json),
[«Диффузионные языковые модели: MDLM и LLaDA»](2026-11-22-masked-diffusion-editorial-pass.json),
[«Декодирование и выбор следующего токена»](2026-11-22-decoding-editorial-pass.json),
[«Постобучение языковых моделей»](2026-11-22-posttraining-intro-editorial-pass.json),
[«SFT: обучение на демонстрациях»](2026-11-22-sft-editorial-pass.json),
[«Fine-tuning и PEFT»](2026-11-22-peft-editorial-pass.json),
[«Данные предпочтений»](2026-11-22-preference-data-editorial-pass.json),
[«Модель награды»](2026-11-22-reward-model-editorial-pass.json),
[«Градиент стратегии и PPO»](2026-11-22-ppo-editorial-pass.json),
[«HHH, Constitutional AI и RLAIF»](2026-11-22-alignment-goals-editorial-pass.json),
[«DPO: обучение на парах ответов»](2026-11-22-dpo-editorial-pass.json),
[«RLVR: обучение по проверяемой награде»](2026-11-22-rlvr-editorial-pass.json),
[«GRPO и DeepSeek-R1»](2026-11-22-grpo-editorial-pass.json),
[«Дистилляция рассуждений»](2026-11-22-distillation-editorial-pass.json),
[«Вычисления во время ответа»](2026-11-22-test-time-compute-editorial-pass.json),
[«Оценивание моделей и контаминация»](2026-11-22-evaluation-editorial-pass.json),
[«Эмбеддинги и метрическое обучение»](2026-11-22-embeddings-editorial-pass.json),
[«Лексический, плотный и гибридный поиск»](2026-11-22-retrieval-editorial-pass.json),
[«Переранжирование документов»](2026-11-22-reranking-editorial-pass.json),
[«RAG: от источника до проверяемого ответа»](2026-11-22-rag-editorial-pass.json),
[«Визуальные токены: ViT, CLIP и SigLIP»](2026-11-22-vision-encoders-editorial-pass.json),
[«Как подключить визуальный энкодер к LLM»](2026-11-22-vision-connectors-editorial-pass.json),
[«Разрешение, плитки и позиции изображения»](2026-11-22-vision-resolution-editorial-pass.json),
[«Как обучают мультимодальные модели»](2026-11-22-vision-training-editorial-pass.json),
[«Чтение документов: OCR, поля и координаты»](2026-11-22-ocr-grounding-editorial-pass.json),
[«Как модели понимают видео и отвечают голосом»](2026-11-22-video-omni-editorial-pass.json),
[«Как оценивать качество и задержку VLM»](2026-11-22-vlm-evaluation-serving-editorial-pass.json),
[«Chameleon: текст и изображения в одной последовательности»](2026-11-22-chameleon-editorial-pass.json),
[«Как агент работает с инструментами»](2026-11-22-agent-intro-editorial-pass.json),
[«Как вызов функции становится действием»](2026-11-22-tool-use-editorial-pass.json),
[«Как агент управляет контекстом»](2026-11-22-harness-context-editorial-pass.json),
[«Память, планирование и оркестрация агентов»](2026-11-22-agent-memory-planning-editorial-pass.json),
[«Оценивание агентных систем»](2026-11-22-agent-evaluation-editorial-pass.json),
[«Программирующие агенты и работа с интерфейсами»](2026-11-22-computer-agents-editorial-pass.json),
[«Формальные доказательства и математические агенты»](2026-11-22-formal-math-editorial-pass.json),
[«Агенты научного поиска»](2026-11-22-scientific-discovery-editorial-pass.json) и
[«Безопасность агентных систем»](2026-11-22-agent-security-editorial-pass.json).

На 22 ноября редакторский проход по исходному основному русскому маршруту
завершён: 85 из 85. Морфология, синтаксис, NER, классификация документов,
извлечение информации и машинный перевод прошли отдельную приёмку;
теперь в маршруте 91 глава. В эту цепочку не входят практика и
углублённые модули.

Английское оглавление ведёт по той же последовательности, но только
35 глав открываются без сообщения об отсутствии отдельной английской
редакции. Остальные 56 показывают канонический русский текст с этим
предупреждением. Два пробела программы NLP в таблице выше остаются
открытыми.
