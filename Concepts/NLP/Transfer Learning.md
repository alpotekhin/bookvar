---
title: "Transfer Learning"
aliases: [transfer learning in NLP, pre-training, fine-tuning paradigm, embeddings evolution, Word2Vec, GloVe, ELMo]
type: concept
status: legacy
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/BERT|BERT]]"
  - "[[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]]"
  - "[[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]]"
  - "[[02 Areas/ML & DL/Papers/T5|T5]]"
  - "[[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing|Pre-train, Prompt, and Predict]]"
  - "[[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning|Revisiting Few-sample BERT Fine-tuning]]"
courses:
  - "[[02 Areas/ML & DL/Courses/Stanford CS224N/07 — Attention|CS224N Lecture 7]]"
sources:
  - "[Jay Alammar — The Illustrated BERT, ELMo](https://jalammar.github.io/illustrated-bert/)"
  - "[Lena Voita — Transfer Learning in NLP](https://lena-voita.github.io/nlp_course/transfer_learning.html)"
  - "[Sebastian Ruder — NLP's ImageNet Moment](https://ruder.io/nlp-imagenet/)"
  - "[d2l.ai — BERT: Pre-training](https://d2l.ai/chapter_natural-language-processing-pretraining/)"
---

# Transfer Learning / Embeddings

## Зачем это нужно: почему не учить с нуля

До 2018 года каждую NLP-задачу решали отдельной моделью, обученной на task-specific данных. Хочешь sentiment analysis? Собери 50K размеченных отзывов. Хочешь NER? Ещё 30K размеченных предложений. Для каждой задачи — отдельная разметка, отдельная архитектура, отдельное обучение.

**Transfer learning** перевернул эту парадигму: **один раз** обучить модель на огромном unlabeled корпусе (Wikipedia, Common Crawl), а затем **адаптировать** к конкретной задаче с минимумом данных. Аналогия с computer vision: ImageNet pre-training → fine-tuning на медицинских снимках. Sebastian Ruder в 2018 назвал это **"NLP's ImageNet moment"**.

## Эволюция: от Word2Vec до GPT-4

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png]]
*Эволюция представлений: от static word embeddings (Word2Vec) к контекстуализированным представлениям (ELMo, BERT) (источник: Jay Alammar)*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-transfer-learning.png]]
*Сравнение подходов: ELMo (feature-based), GPT (fine-tuning, left-to-right), BERT (fine-tuning, bidirectional) (источник: Jay Alammar)*

История transfer learning в NLP — это пять эпох, каждая из которых решала ключевое ограничение предыдущей.

### Эпоха 1: Static Word Embeddings (2013-2017)

#### Word2Vec (Mikolov et al., 2013)

Революционная идея: слова как **плотные векторы** в непрерывном пространстве. Близкие по смыслу слова → близкие векторы.

Две архитектуры:
- **CBOW** (Continuous Bag of Words): предсказывает целевое слово по контексту
- **Skip-gram**: предсказывает контекст по целевому слову

Знаменитые аналогии: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$.

**Ограничение**: каждое слово = **один вектор**, независимо от контекста. «Bank» в «river bank» и «bank account» имеют одинаковый вектор.

#### GloVe (Pennington et al., 2014)

GloVe (Global Vectors) — альтернативный подход: вместо нейросети используется **матричная факторизация** матрицы ко-встречаемости слов.

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2$$

где $X_{ij}$ — количество совместных появлений слов $i$ и $j$ в окне.

**Преимущество перед Word2Vec**: использует глобальную статистику корпуса, а не только локальные окна. На практике — сопоставимое качество.

**Ограничение**: те же статические эмбеддинги — нет контекста.

#### FastText (Bojanowski et al., 2017)

Расширение Word2Vec: разбивает слова на character n-grams и суммирует их эмбеддинги. Слово `"where"` → `["<wh", "whe", "her", "ere", "re>"]`.

**Преимущество**: работает с OOV-словами (если хоть часть n-grams знакома) и лучше для морфологически богатых языков (русский, турецкий).

### Эпоха 2: Contextualized Representations (2017-2018)

#### CoVe (McCann et al., 2017)

Первая попытка контекстных эмбеддингов: энкодер от NMT-модели (machine translation) используется для создания «words in context» — контекстуализированных представлений.

#### ELMo (Peters et al., 2018)

**Embeddings from Language Models** — прорыв: двунаправленный LSTM обучается как языковая модель (forward + backward), и **все скрытые слои** используются для создания эмбеддинга.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png]]
*Контекстуализированные эмбеддинги: одно слово получает разные векторы в зависимости от контекста (источник: Jay Alammar)*

**Ключевая инновация**: разные downstream tasks нуждаются в **разной** информации, поэтому ELMo использует **task-specific взвешенное усреднение** трёх слоёв:

$$\text{ELMo}_k = \gamma^{\text{task}} \sum_{j=0}^{L} s_j^{\text{task}} \cdot h_{k,j}$$

- Нижние слои → синтаксис (POS, constituency)
- Верхние слои → семантика (word sense, coreference)

**Результат**: +1-5% на шести NLP-бенчмарках, просто заменив Word2Vec на ELMo.

**Использование**: feature-based — ELMo замораживается, его выходы подаются в task-specific модель как фичи.

### Эпоха 3: Pre-train → Fine-tune (2018-2020)

Два ключевых сдвига по Lena Voita:
1. **От слов к словам-в-контексте** (Word2Vec → ELMo)
2. **От замены эмбеддингов к замене целых моделей** (ELMo → GPT/BERT)

#### ULMFiT (Howard & Ruder, 2018)

Первая формализация fine-tuning для NLP. Три этапа: LM pre-training → LM fine-tuning на task domain → classifier fine-tuning. Введены slanted triangular learning rates и discriminative fine-tuning (разные LR для разных слоёв).

#### GPT (Radford et al., 2018)

OpenAI GPT — первый **Transformer-based** pre-trained model для NLP. Обучение: causal language modeling (предсказание следующего токена) на BookCorpus.

Fine-tuning: добавляем task-specific linear head, обновляем **все** параметры.

#### BERT (Devlin et al., 2018)

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-transfer-learning.png]]
*Transfer learning в BERT: pre-training на unlabeled data → fine-tuning на task-specific data (источник: Jay Alammar)*

Два нововведения:
- **Masked Language Modeling (MLM)**: случайно маскируем 15% токенов, модель предсказывает их по **двустороннему** контексту. Три стратегии маскирования: `[MASK]` (80%), случайный токен (10%), без изменений (10%).
- **Next Sentence Prediction (NSP)**: бинарная классификация — следует ли предложение B за предложением A.

**Результат**: +7% average GLUE vs GPT-1, благодаря bidirectional контексту. BERT стал стандартом для NLU на 2-3 года.

### Эпоха 4: Pre-train → Prompt → Predict (2020+)

#### GPT-3 (Brown et al., 2020)

175B параметров, обученных на 300B+ токенов. Радикальный shift: **без fine-tuning**. Модель решает задачи через prompting — описание задачи в natural language + примеры:

```
Translate English to French:
sea otter => loutre de mer
cheese => fromage
house =>
```

Три режима:
- **Zero-shot**: только инструкция, без примеров
- **One-shot**: один пример
- **Few-shot**: 2-64 примера в контексте

**Результат**: few-shot GPT-3 конкурирует с fine-tuned BERT на многих задачах, **без обновления весов**.

#### T5 (Raffel et al., 2020)

"Text-to-Text Transfer Transformer" — все задачи приводятся к единому формату input text → output text:

```
translate English to German: That is good  →  Das ist gut
summarize: <long text>                      →  <summary>
sentiment: This movie was great             →  positive
```

Систематическое исследование: какой pre-training objective, corpus, model size, и fine-tuning strategy работают лучше всего.

### Эпоха 5: Foundation Models (2022+)

InstructGPT, ChatGPT, GPT-4 — RLHF (Reinforcement Learning from Human Feedback) добавляет alignment после pre-training. Модели становятся «assistant»-ами, следующими инструкциям.

## Два подхода к использованию pre-trained моделей

### Feature-based (ELMo style)

```
frozen_model → extract embeddings → task-specific classifier
```

Pre-trained модель замораживается, её выходы используются как **фичи** для отдельного classifier. Преимущества: быстро, модель не портится, можно комбинировать фичи из разных слоёв.

### Fine-tuning (BERT/GPT style)

```
pre-trained_model + task_head → update ALL parameters on task data
```

Вся модель дообучается на task-specific данных. Преимущества: лучшее качество, модель полностью адаптируется. Типичные параметры fine-tuning (BERT): 2-4 epochs, LR = 2-5e-5, batch = 16-32.

### Probing: что именно модель выучила

Probing studies (Tenney et al., 2019) показали, что разные слои BERT кодируют разную лингвистическую информацию:

| Слои | Что кодируют |
|------|-------------|
| Нижние (1-4) | POS tagging, морфология |
| Средние (5-8) | Синтаксис (constituency, dependencies) |
| Верхние (9-12) | Семантика (coreference, word sense) |
| Последний слой | MLM objective (плохая инициализация для downstream) |

Это объясняет, почему **Re-init верхних слоёв** при few-shot fine-tuning может улучшить результаты (из [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning|Revisiting Few-sample BERT Fine-tuning]]).

## Четыре парадигмы NLP

Из [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing|Pre-train, Prompt, and Predict]]:

| Парадигма | Что инженерим | Примеры | Эра |
|-----------|--------------|---------|-----|
| Non-neural supervised | Features | SVM + BoW, CRF | до 2013 |
| Neural supervised | Architecture | BiLSTM, CNN-text | 2013-2018 |
| **Pre-train → Fine-tune** | Objective | BERT + task head | 2018-2021 |
| **Pre-train → Prompt** | Prompt | GPT-3 few-shot, ChatGPT | 2020+ |

Каждая парадигма перенесла сложность из одной области в другую: от ручного feature engineering к architecture engineering, от него к objective engineering, и наконец к prompt engineering.

## Почему pre-trained representations изменили NLP

### 1. Экономия данных

BERT pre-trained на ~16GB текста (Wikipedia + BookCorpus) → fine-tune на 1K-100K размеченных примеров. Разметка для pre-training **не нужна** — модель учится на сыром тексте.

### 2. Знания как побочный эффект

Petroni et al. (2019) — *"Language Models as Knowledge Bases?"* — показали, что BERT хранит фактические знания. Prompt `"Dante was born in ___"` → модель предсказывает `"Florence"` без обучения на структурированных фактах.

### 3. Универсализация

Одна модель (GPT-3, T5) решает **сотни** задач: translation, summarization, QA, classification, code generation. Не нужно проектировать отдельную архитектуру для каждой задачи.

### 4. Доступность

Организации без ресурсов на pre-training (дни GPU-кластеров, миллионы долларов) могут **скачать** pre-trained модель и fine-tune за часы на одном GPU. Hugging Face Hub → 500K+ моделей.

## Хронология

| Год | Milestone | Тип transfer | Значение |
|-----|-----------|-------------|----------|
| 2013 | Word2Vec | Static embeddings | Слова как векторы |
| 2014 | GloVe | Static embeddings | Глобальная статистика |
| 2017 | CoVe | Feature-based | Первые контекстные эмбеддинги |
| 2017 | FastText | Static embeddings | Character n-grams, OOV |
| 2018 | ELMo | Feature-based | Контекстные LSTM-эмбеддинги |
| 2018 | ULMFiT | Fine-tuning | Формализация fine-tuning для NLP |
| 2018 | GPT-1 | Fine-tuning | Первый Transformer pre-training |
| **2018** | **BERT** | **Fine-tuning** | **+7% GLUE, NLP's ImageNet moment** |
| 2019 | GPT-2 | Zero-shot | "Unsupervised Multitask Learners" |
| 2020 | GPT-3 | In-context learning | 175B, few-shot без fine-tuning |
| 2020 | T5 | Fine-tuning | Text-to-text universal transfer |
| 2021 | LoRA | Parameter-efficient FT | Fine-tuning без обновления всех весов |
| 2022 | InstructGPT | RLHF alignment | Transfer + human feedback |
| 2023 | GPT-4, LLaMA | Foundation models | Масштаб + доступность |

## Key papers

- [[02 Areas/ML & DL/Papers/BERT|BERT]] — pre-train → fine-tune, +7% GLUE, bidirectional transfer
- [[02 Areas/ML & DL/Papers/GPT 2.0|GPT-2]] — zero-shot prompting
- [[02 Areas/ML & DL/Papers/GPT 3.0|GPT-3]] — few-shot in-context learning, 175B
- [[02 Areas/ML & DL/Papers/T5|T5]] — систематическое исследование transfer, text-to-text
- [[02 Areas/ML & DL/Papers/Pre-train, Prompt, and Predict- A Systematic Survey of Prompting Methods in Natural Language Processing|Pre-train, Prompt, and Predict]] — таксономия четырёх парадигм
- [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning|Revisiting Few-sample BERT Fine-tuning]] — probing layers, re-init
- Peters et al. 2018 — *Deep contextualized word representations* (ELMo)
- Howard & Ruder 2018 — *Universal Language Model Fine-tuning for Text Classification* (ULMFiT)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] — первый этап transfer learning
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — второй этап
- [[02 Areas/ML & DL/Concepts/NLP/Prompt-based Learning|Prompt-based Learning]] — альтернатива fine-tuning
- [[02 Areas/ML & DL/Concepts/NLP/Tokenization|Tokenization]] — словарь токенизатора как часть pre-trained модели
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — paradigm shift
- [[02 Areas/ML & DL/Concepts/Architectures/GPT-3|GPT-3]] — in-context learning
- [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] — parameter-efficient fine-tuning

## Дополнительные ресурсы

- [Jay Alammar — The Illustrated BERT, ELMo](https://jalammar.github.io/illustrated-bert/) — лучшие визуализации эволюции от ELMo к BERT
- [Lena Voita — Transfer Learning in NLP](https://lena-voita.github.io/nlp_course/transfer_learning.html) — глубокий обзор с математикой
- [Sebastian Ruder — NLP's ImageNet Moment](https://ruder.io/nlp-imagenet/) — историческая перспектива
- [d2l.ai — BERT Pre-training](https://d2l.ai/chapter_natural-language-processing-pretraining/) — учебник с кодом
