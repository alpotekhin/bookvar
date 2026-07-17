---
title: "Text Classification"
aliases: [sentiment analysis, document classification, text categorization, классификация текста]
type: concept
category: NLP
papers:
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/Adversarial Training Methods for Semi-Supervised Text Classification]]"
  - "[[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
courses: []
sources:
  - "[Sebastian Raschka — Understanding LLMs for Text Classification](https://magazine.sebastianraschka.com/p/understanding-large-language-models)"
  - "[Hugging Face — Text Classification](https://huggingface.co/tasks/text-classification)"
---

# Text Classification

## Что это такое

Text Classification --- одна из фундаментальных задач [[02 Areas/ML & DL/Concepts/NLP/NLP|NLP]]: присвоить входному тексту одну или несколько предопределённых меток. Это может быть определение тональности отзыва (positive / negative), тематики новости, языка, токсичности, жанра, намерения пользователя --- всё, что можно свести к выбору из конечного набора классов.

Исторически text classification --- первая задача, на которой каждая новая парадигма NLP доказывала свою состоятельность: от Bag-of-Words + Naive Bayes до BERT fine-tuning и LLM prompting.

## Эволюция подходов

### Эра 1: статистические методы (до 2013)

**Bag-of-Words (BoW) + классический ML:**
- Текст представляется как вектор частот слов (или TF-IDF весов)
- Классификатор: Naive Bayes, Logistic Regression, SVM
- Теряется порядок слов, но работает удивительно хорошо на многих задачах
- Naive Bayes на IMDB: ~85% accuracy --- сильный baseline

**N-gram модели:** биграммы и триграммы частично сохраняют порядок, улучшая BoW на 1--3%.

### Эра 2: нейросетевые эмбеддинги (2013--2018)

**Word2Vec / GloVe + RNN/CNN:**
- Word2Vec (Mikolov, 2013): плотные 300-мерные эмбеддинги вместо разреженных BoW
- TextCNN (Kim, 2014): свёрточные фильтры по эмбеддингам --- быстро и эффективно
- BiLSTM + Attention: рекуррентные сети с механизмом внимания для длинных документов
- IMDB: ~90--92% accuracy

**Adversarial Training (Miyato et al., 2017):**
Из [[02 Areas/ML & DL/Papers/Adversarial Training Methods for Semi-Supervised Text Classification]]: добавление adversarial perturbation к эмбеддингам при обучении. Virtual Adversarial Training снизил ошибку на IMDB с 7.39% до **5.91%** --- почти SOTA на тот момент (5.94%). Метод работает и в semi-supervised режиме (когда размечена только часть данных).

### Эра 3: Pre-train + Fine-tune (2018--2022)

**BERT и [CLS] токен:**
Из [[02 Areas/ML & DL/Papers/BERT]] (Section 4.1) --- стандартная схема fine-tuning:

```
Input:  [CLS] text [SEP]
C = T_[CLS]                    # 768-мерный вектор [CLS] токена
P(y) = softmax(C * W^T)        # W in R^(num_classes x 768)
Loss = CrossEntropy(P, y_true)
```

Fine-tuning обновляет **все** параметры BERT + классификационную голову $W$. Для задач с парами предложений: `[CLS] sent_A [SEP] sent_B [SEP]`.

Результаты на [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE]]:

| Модель | SST-2 | MRPC (F1) | CoLA (MCC) | GLUE avg |
|--------|:-----:|:---------:|:----------:|:--------:|
| OpenAI GPT | 91.3 | 82.3 | 45.4 | 72.8 |
| BERT-Large | 94.9 | 89.3 | 60.5 | 80.5 |
| [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] | 96.4 | 90.9 | 68.0 | 88.5 |
| [[02 Areas/ML & DL/Concepts/Architectures/DeBERTa|DeBERTa]] v3 | 97.0+ | 92+ | 72+ | 92+ |

### Эра 4: LLM prompting (2022+)

**Zero/few-shot classification через LLM:**
Вместо fine-tuning --- промпт: *"Classify the sentiment of this review as positive or negative: ..."*

Из [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] (Section 4.1):

| Задача | Fine-tuned BERT | LLM (ChatGPT) | Вывод |
|--------|:--------------:|:-------------:|-------|
| Sentiment (IMDB, SST) | ~95% | ~93% | Comparable |
| Toxicity (CivilComments) | BERT wins | Хуже | Fine-tuned лучше |
| NLI (RTE, SNLI) | BERT wins | --- | Fine-tuned лучше |
| Adversarial NLI (ANLI R3) | Плохо | Лучше | LLM на OOD данных |
| Topic classification | --- | Хорошо | LLM для diverse domains |

**Практическое правило:** для стандартных задач с labeled data --- fine-tuned encoder (BERT/RoBERTa); для OOD / diverse / zero-shot --- LLM.

## Типы задач классификации

| Тип | Описание | Примеры | Метрика |
|-----|----------|---------|---------|
| **Binary** | Два класса | IMDB (pos/neg), SST-2 | Accuracy, F1 |
| **Multiclass** | Один из $K$ классов | DBpedia (14 тем), AG News (4 темы) | Accuracy |
| **Multi-label** | Несколько меток одновременно | Reuters (темы статьи), arXiv (категории) | Micro/Macro F1 |
| **Ordinal** | Упорядоченные классы | Рейтинги 1--5, степень токсичности | MAE, Quadratic Kappa |
| **Entailment / NLI** | Отношение между двумя предложениями | SNLI, MNLI, RTE | Accuracy |
| **Aspect-based** | Тональность по аспектам | "Еда отличная, но обслуживание ужасное" | F1 per aspect |

## Sentiment Analysis: подробный пример

Sentiment analysis --- самый популярный подтип text classification. Определение эмоциональной окраски текста.

**Уровни анализа:**
- **Document-level:** весь отзыв -> positive/negative
- **Sentence-level:** каждое предложение -> pos/neg/neutral
- **Aspect-level:** "Камера отличная, но батарея слабая" -> camera: pos, battery: neg

**Основные датасеты:**

| Датасет | Размер | Классы | Средняя длина |
|---------|:------:|:------:|:------------:|
| IMDB | 50K | 2 | 239 слов |
| SST-2 | 67K | 2 | 19 слов |
| SST-5 | 11K | 5 | 19 слов |
| Yelp Full | 650K | 5 | ~150 слов |
| Amazon Reviews | 3.6M | 5 | ~80 слов |

## Практические рекомендации

**Выбор подхода по ситуации:**

| Ситуация | Рекомендация |
|----------|-------------|
| < 100 примеров, нет GPU | LLM zero/few-shot prompting |
| 100--1000 примеров | SetFit или few-shot fine-tuning |
| 1000--100K примеров, есть GPU | Fine-tune BERT/RoBERTa |
| > 100K примеров | Fine-tune, рассмотреть distillation |
| Нужна интерпретируемость | Logistic Regression + TF-IDF |
| Стриминг / real-time | Лёгкая модель (DistilBERT, TinyBERT) |

**Типичные ошибки:**
1. **Несбалансированные классы** --- accuracy обманчива; использовать F1, PR-AUC
2. **Data leakage** --- train/test split по документам, не по предложениям
3. **Длинные документы** --- BERT обрезает до 512 токенов; стратегии: truncation, chunking + pooling, Longformer
4. **Нестабильность fine-tuning на малых данных** --- RTE (2.5K train): 48% degenerate runs с некорректным Adam (из [[02 Areas/ML & DL/Papers/Revisiting Few-sample BERT Fine-tuning]])

## Key papers

- [[02 Areas/ML & DL/Papers/BERT]] --- стандарт fine-tuning через [CLS] токен
- [[02 Areas/ML & DL/Papers/RoBERTa]] --- оптимизированный pre-training, GLUE 88.5
- [[02 Areas/ML & DL/Papers/Adversarial Training Methods for Semi-Supervised Text Classification]] --- adversarial perturbations для text classification
- [[02 Areas/ML & DL/Papers/Harnessing the Power of LLMs in Practice- A Survey on ChatGPT and Beyond]] --- LLM vs fine-tuned comparison

## Related concepts

- [[02 Areas/ML & DL/Concepts/NLP/GLUE|GLUE / SuperGLUE]] --- стандартный benchmark включающий classification задачи
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] --- основной метод адаптации pre-trained моделей
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] --- архитектура, определившая современный подход
- [[02 Areas/ML & DL/Concepts/NLP/Transfer Learning|Transfer Learning]] --- парадигма pre-train + fine-tune
- [[02 Areas/ML & DL/Concepts/NLP/Prompt-based Learning|Prompt-based Learning]] --- альтернатива fine-tuning
- [[02 Areas/ML & DL/Concepts/Training/Adversarial Training|Adversarial Training]] --- регуляризация через perturbations
