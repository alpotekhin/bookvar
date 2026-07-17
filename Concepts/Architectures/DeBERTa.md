---
title: "DeBERTa"
aliases: [Decoding-enhanced BERT with Disentangled Attention]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/DeBERTa]]"
  - "[[02 Areas/ML & DL/Papers/BERT]]"
  - "[[02 Areas/ML & DL/Papers/RoBERTa]]"
courses: []
sources:
  - "[He et al. — DeBERTa (ICLR 2021)](https://arxiv.org/abs/2006.03654)"
  - "[Microsoft DeBERTa GitHub](https://github.com/microsoft/DeBERTa)"
  - "[HuggingFace DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-large)"
---

# DeBERTa — Decoding-enhanced BERT with Disentangled Attention

## Зачем эта модель появилась

К 2020 году [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] установила планку для encoder-only моделей: правильный training recipe превращает BERT в мощный NLU backbone. Но архитектура оставалась без изменений — тот же Transformer encoder с той же реализацией attention.

He et al. (Microsoft, 2020) задали вопрос: **что если attention в BERT неоптимален?** Конкретнее: BERT складывает content embedding и position embedding в один вектор, теряя возможность моделировать их взаимодействие раздельно. DeBERTa предлагает **disentangled attention** — раздельное моделирование содержания и позиции.

Результат: DeBERTa XXL (1.5B) стала **первой моделью, превзошедшей человеческий baseline на SuperGLUE** (89.9 vs 89.8, январь 2021).

## Ключевая инновация 1: Disentangled Attention

### Проблема с BERT

В BERT каждый токен представлен **одним вектором** — суммой content и position embeddings:

$$e_i = W_e[\text{token}_i] + W_p[\text{position}_i]$$

Затем attention считается через стандартный dot-product: $A_{i,j} = q_i \cdot k_j^T$. Проблема: **information mixing**. Когда мы считаем attention между словами, мы не можем разделить, насколько важно **содержание** слова vs его **позиция** — всё смешано в одном векторе.

### Решение DeBERTa: два вектора

Каждый токен представлен **двумя отдельными векторами**:
- $H_i$ — content embedding (что это за слово)
- $P_{i|j}$ — relative position embedding (где слово $i$ находится относительно слова $j$)

Attention score раскладывается на **четыре компоненты** (декомпозиция взаимодействий):

$$A_{i,j} = \underbrace{H_i H_j^T}_{\text{content↔content}} + \underbrace{H_i P_{j|i}^T}_{\text{content↔position}} + \underbrace{P_{i|j} H_j^T}_{\text{position↔content}} + \underbrace{P_{i|j} P_{j|i}^T}_{\text{position↔position}}$$

Четвёртый член (position↔position) **удалён** в реализации — он не даёт значимого вклада, потому что relative position embedding не зависит от конкретных токенов.

**Интуиция через пример:** рассмотрим «deep learning» в двух контекстах:
- *"I study **deep learning** at university"* — «deep» и «learning» стоят рядом → strong attention
- *"I study learning processes in **deep** water"* — те же слова, но далеко друг от друга → weak attention

Content-to-content: одинаковый в обоих случаях (те же слова). Но **content-to-position** и **position-to-content** различаются: близость усиливает attention, удалённость ослабляет. Disentangled attention может это моделировать, стандартный BERT — нет.

### Масштабирование

Все три оставшихся компоненты нормализуются на $\sqrt{3d}$ (вместо стандартного $\sqrt{d}$), потому что итоговый score — сумма трёх членов, каждый с дисперсией ~$d$:

$$A_{i,j} = \frac{H_i H_j^T + H_i P_{j|i}^T + P_{i|j} H_j^T}{\sqrt{3d}}$$

### Relative Position Encoding

DeBERTa использует **relative** position embeddings (до $k=512$ позиций), а не absolute. Relative PE:
- Лучше обобщаются на последовательности разной длины
- Кодируют **расстояние между токенами**, а не абсолютное положение в последовательности
- Матрица $\delta(i,j)$ отсекает расстояния больше $k$: если $|i-j| \geq k$, используется максимальное значение

## Ключевая инновация 2: Enhanced Mask Decoder (EMD)

### Проблема

Disentangled attention использует только **relative** позиции. Но для MLM задачи нужны **absolute** позиции:

*"a new **[MASK]** opened beside the new **[MASK]**"*

Оба маскированных токена имеют похожий контекст (оба окружены «new»), но это разные слова: «store» (подлежащее, начало предложения) и «mall» (дополнение, конец предложения). **Абсолютная позиция** критична для различения их синтаксических ролей.

### Решение

**Enhanced Mask Decoder** — дополнительный слой перед MLM prediction head, который вводит **absolute position embeddings** только для маскированных токенов:

```
Disentangled Transformer layers (relative position only)
    ↓
Enhanced Mask Decoder (adds absolute position)
    ↓
MLM Prediction Head
```

Так DeBERTa получает лучшее из двух миров:
- **Relative positions** в основных слоях → лучшее моделирование зависимостей
- **Absolute positions** только в decoder → точное предсказание маскированных токенов

## Scale-Invariant Fine-Tuning (SiFT)

DeBERTa также предлагает **virtual adversarial training** для fine-tuning: вместо adversarial perturbations в input embedding space, SiFT нормализует embedding вектора перед perturbation:

$$\tilde{e}_i = e_i + \Delta_i, \quad \|\Delta_i\| \leq \epsilon$$

Это улучшает generalization на downstream задачах, особенно на малых датасетах.

## Архитектура в целом

| Компонент | DeBERTa | BERT/RoBERTa |
|-----------|---------|--------------|
| Attention | Disentangled (3 компоненты) | Standard dot-product |
| Position encoding | Relative (в attention) + Absolute (в EMD) | Absolute (в embeddings) |
| Pre-training | MLM (без NSP, как RoBERTa) | MLM + NSP (BERT) / MLM (RoBERTa) |
| Fine-tuning | SiFT (virtual adversarial) | Standard |
| Data efficiency | **Обучена на половине данных RoBERTa** | — |

## Ключевые результаты

### DeBERTa-Large vs RoBERTa-Large (половина training data!)

| Задача | RoBERTa-Large | DeBERTa-Large | Дельта |
|--------|---------------|---------------|--------|
| MNLI | 90.2% | **91.1%** | +0.9% |
| SQuAD v2.0 F1 | 88.4% | **90.7%** | +2.3% |
| RACE | 83.2% | **86.8%** | +3.6% |
| WikiText-103 PPL | 21.6 | **19.5** | -2.1 |

DeBERTa обучена на **половине данных** RoBERTa и при этом стабильно лучше — свидетельство того, что архитектурные улучшения дают реальный выигрыш.

### DeBERTa XXL (1.5B) на SuperGLUE

| Модель | Params | SuperGLUE (macro-avg) |
|--------|--------|-----------------------|
| T5-11B | 11B | 89.3 |
| Human baseline | — | 89.8 |
| **DeBERTa XXL** | **1.5B** | **89.9** (single) / **90.3** (ensemble) |

**Первая модель, превзошедшая человека на SuperGLUE.** При этом DeBERTa XXL в **7 раз меньше** T5-11B — демонстрация того, что правильная архитектура может компенсировать масштаб.

## Почему DeBERTa важна

### 1. Архитектурная инновация в эпоху «just scale»

DeBERTa показала, что **архитектурные улучшения работают** — в период, когда основным трендом было просто увеличивать модель. Disentangled attention — одна из немногих модификаций attention, дающих стабильный прирост.

### 2. Эталон SuperGLUE

Превышение human baseline на SuperGLUE стало **milestone для NLU** — аналогичным тому, как AlphaGo победила чемпиона мира в Go. Это показало: encoder-only модели при правильном масштабировании достигают human-level NLU.

### 3. Практическая рабочая лошадка

DeBERTa-v3-large — один из самых популярных backbone для production NLU:
- **NER**: лучшая точность среди моделей сопоставимого размера
- **Classification**: GLUE/SuperGLUE fine-tuned модели
- **QA span extraction**: SQuAD, конкурентные QA системы
- **Sentence embeddings**: через mean pooling или [CLS] token

Где fine-tuned encoder-only модели всё ещё лучше LLM — при наличии аннотированных данных, для задач NLU.

## DeBERTa-v2 и v3

- **DeBERTa-v2**: увеличенный vocabulary (128K), улучшенный tokenizer
- **DeBERTa-v3**: заменяет MLM на **ELECTRA-style Replaced Token Detection (RTD)** — более sample-efficient pre-training. DeBERTa-v3-large — текущий рекомендуемый вариант для production.

## Key papers

- [[02 Areas/ML & DL/Papers/DeBERTa]] — оригинал (He et al., ICLR 2021)
- [[02 Areas/ML & DL/Papers/RoBERTa]] — предшественник (оптимизация training recipe)
- [[02 Areas/ML & DL/Papers/BERT]] — архитектурная основа

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] — архитектурный прародитель
- [[02 Areas/ML & DL/Concepts/Architectures/RoBERTa|RoBERTa]] — предшественник по training recipe
- [[02 Areas/ML & DL/Concepts/Architectures/Encoder-only|Encoder-only]] — архитектурный класс
- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling (MLM)]] — pre-training objective
- [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|Self-Attention]] — базовый механизм
- [[02 Areas/ML & DL/Concepts/NLP/Positional Encoding|Positional Encoding]] — relative vs absolute PE

## Дополнительные ресурсы

- [Microsoft DeBERTa GitHub](https://github.com/microsoft/DeBERTa) — официальная реализация
- [HuggingFace DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large) — рекомендуемый чекпоинт для production
- [SuperGLUE Leaderboard (archive)](https://super.gluebenchmark.com/leaderboard) — исторический leaderboard
