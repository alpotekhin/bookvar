# Покрытие NLP в основном курсе

Срез от 21 ноября 2026 года. В оглавлении 85 глав основного курса и
68 глав углублённых модулей. Это счётчик глав в оглавлении, не
результат новой редакторской приёмки. Навигационные правки и их исходные
ревизии зафиксированы в [отдельном журнале](2026-11-21-linear-nlp-navigation.json).

Для проверки программы использовано оглавление [Jurafsky и Martin, Speech and
Language Processing, редакция от 19 августа 2026 года](https://web.stanford.edu/~jurafsky/slp3/).
Для обучения LLM, перевода и архитектур дополнительные ориентиры —
[CS336](https://cs336.stanford.edu/) и [CS224N](https://web.stanford.edu/class/cs224n/).
Материалы Jurafsky и Martin помечены авторами как *all rights reserved*;
ссылки на главы не являются разрешением на публикацию их рисунков.

| Что проверить | Что уже есть в Bookvar | Какого разбора пока не видно в оглавлении |
| --- | --- | --- |
| Словоформы, леммы, морфология и правка текста ([SLP, гл. 2](https://web.stanford.edu/~jurafsky/slp3/2.pdf)) | В главе «Представление текста числами» разобраны NFC и NFKC, выбор леммы «сталь» или «стать» по контексту и таблица расстояния Левенштейна. Отдельная глава показывает BPE, WordPiece и Unigram. | Связать леммы и морфологические признаки с разметкой последовательности. Показать, как оценивают ошибки морфологического разбора. |
| Классификация документов ([SLP, гл. 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf)) | Bag of words и TF–IDF; общий модуль линейных моделей; примеры оценки тональности в главах о GPT. | Показать одну задачу от признаков и логистической регрессии до BERT. Не дублировать главу о линейных моделях. |
| Части речи и именованные сущности ([SLP, гл. 18](https://web.stanford.edu/~jurafsky/slp3/18.pdf)) | Витерби встречается в токенизации; BERT описан отдельно. | Разметка последовательности, HMM/CRF, переход к набору меток на выходе энкодера, выравнивание слов с субтокенами, точность границ сущности. |
| Синтаксис ([SLP, гл. 19](https://web.stanford.edu/~jurafsky/slp3/19.pdf) и [гл. 20](https://web.stanford.edu/~jurafsky/slp3/20.pdf)) | В основном курсе нет главы о синтаксическом разборе. | Разобрать одно предложение как дерево составляющих и как дерево зависимостей, затем показать работу парсера и измерение его ошибок. |
| Извлечение информации и связь упоминаний ([SLP, гл. 21](https://web.stanford.edu/~jurafsky/slp3/21.pdf) и [гл. 24](https://web.stanford.edu/~jurafsky/slp3/24.pdf)) | Структурированные ответы встречаются в главах о RAG, агентах и документах. | На одном документе провести путь от упоминаний к отношению и записи об объекте; отдельно различить кореференцию и связывание с внешней базой. |
| Машинный перевод ([SLP, гл. 13](https://web.stanford.edu/~jurafsky/slp3/13.pdf)), суммаризация и ответы на вопросы | Перевод используется в Seq2Seq, Transformer и GPT-2; BLEU, chrF, COMET и BLEURT рассматриваются в главе об оценивании. В GPT-3 и RAG есть примеры ответов на вопросы. | Собрать перевод в один законченный пример со ссылкой на существующий разбор метрик. Проверить, где объясняются извлекающий и генеративный ответы и как проверяют факты в резюме. |
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
[«T5 — text-to-text Transformer»](2026-11-21-t5-editorial-pass.json),
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

На 22 ноября редакторский проход по существующим главам основного
русского маршрута завершён: 85 из 85. На собранных страницах проверены
336 переходов между соседними главами в двух локалях; в эту цепочку
не входят практика и углублённые модули.

Английское оглавление ведёт по той же последовательности, но только
32 главы открываются без сообщения об отсутствии отдельной английской
редакции. Остальные 53 показывают канонический русский текст с этим
предупреждением. Это не 85 английских глав. Пробелы программы NLP
в таблице выше остаются открытыми.
