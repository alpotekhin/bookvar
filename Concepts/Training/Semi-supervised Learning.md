---
title: "Semi-supervised Learning"
aliases: [semi-supervised, SSL, полу-контролируемое обучение]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Adversarial Training Methods for Semi-Supervised Text Classification]]"
courses: []
sources:
  - "[Miyato et al. -- Adversarial Training Methods for Semi-Supervised Text Classification (ICLR 2017)](https://arxiv.org/abs/1605.07725)"
  - "[Xie et al. -- Unsupervised Data Augmentation for Consistency Training (UDA, 2020)](https://arxiv.org/abs/1904.12848)"
  - "[Chapelle et al. -- Semi-Supervised Learning (book, 2006)](https://www.molgen.mpg.de/3659531/MITPress--SemisupervisedLearning.pdf)"
  - "[Lee -- Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method (2013)](https://citeseerx.ist.psu.edu/document?doi=62bc43e4-fb92-3467-2dfa-f75a7fe8bd49)"
---

# Semi-supervised Learning

## Зачем это нужно: разметка -- это дорого

В типичной NLP задаче:
- **Неразмеченных** текстов -- миллиарды (Common Crawl, Wikipedia)
- **Размеченных** примеров -- тысячи или меньше (аннотация стоит денег и времени)

Semi-supervised Learning (SSL) -- парадигма, которая использует **оба источника**: учится на маленьком labeled dataset + большом unlabeled dataset. Идея: структура неразмеченных данных содержит полезную информацию о decision boundary, даже без labels.

```
Supervised:        [labeled data] --> model
Semi-supervised:   [labeled data] + [unlabeled data] --> model
                    (тысячи)        (миллионы)
Unsupervised:      [unlabeled data] --> representations
```

## Три основных подхода

### 1. Self-training (pseudo-labeling)

Самый простой метод. Идея: обучи модель на labeled данных, примени к unlabeled, **добавь высокоуверенные предсказания** как новые labels.

```
Алгоритм:
1. Обучить модель M на labeled data D_l
2. Применить M к unlabeled data D_u
3. Для каждого x in D_u:
   - p = M(x)
   - Если max(p) > threshold (напр. 0.95):
     - Добавить (x, argmax(p)) в D_l как pseudo-label
4. Переобучить M на расширённом D_l
5. Повторить шаги 2-4 до сходимости
```

**Плюсы**: простота, работает с любой моделью.
**Минусы**: **confirmation bias** -- модель уверенно ошибается, эти ошибки добавляются как "правильные" labels, и ошибка накапливается. Threshold критически важен.

### 2. Consistency Regularization

Идея: если слегка **возмутить** вход, предсказание **не должно меняться**. Это предположение не требует labels.

$$\mathcal{L}_{consistency} = \mathbb{E}_{x \sim D_u}\left[\text{dist}\left(f(x; \theta), f(\text{augment}(x); \theta)\right)\right]$$

Типы возмущений:
- **Data augmentation**: back-translation, synonym replacement, dropout
- **Adversarial perturbation**: VAT (worst-case возмущение в embedding space)
- **Stochastic**: разные dropout masks (один и тот же вход, два прохода)

#### Virtual Adversarial Training (VAT)

Из Miyato et al. (2017) -- самый теоретически обоснованный вариант consistency regularization:

$$\mathcal{L}_{vat} = KL\left[p(\cdot \mid s; \hat{\theta}) \;\|\; p(\cdot \mid s + r_{v\text{-}adv}; \theta)\right]$$

где $r_{v\text{-}adv}$ -- наихудшее возмущение в embedding space. VAT находит **самое чувствительное направление** и сглаживает модель именно там.

Из экспериментов Miyato et al.:
- Random perturbation с unlabeled: IMDB 7.20% --> 6.78%
- VAT с unlabeled: **5.91%** (vs 6.21% supervised adversarial)
- Unlabeled data добавляют ценный обучающий сигнал

#### UDA (Unsupervised Data Augmentation)

Xie et al. (2020) показали, что **quality of augmentation** критически важна:

```
Weak augmentation (random noise):     мало помогает
Strong augmentation (back-translation, RandAugment):  значительно помогает
```

UDA + back-translation на IMDb: **4.20% error** (supervised BERT baseline: 4.51%).

### 3. Language Model Pre-training

Dai & Le (2015) предложили: обучить language model на unlabeled data, затем использовать как **инициализацию** для supervised classifier.

```
Этап 1 (unsupervised): LM pretraining на unlabeled corpus
   LSTM: p(w_t | w_1, ..., w_{t-1})

Этап 2 (supervised): Fine-tune на labeled data
   Classifier: p(y | x) с инициализацией из Этапа 1
```

Это **прото-BERT**: идея pre-training на unlabeled данных с последующим fine-tuning. BERT (2018) масштабировал этот подход с LSTM до Transformer.

## Сравнение подходов

| Метод | Нужны labels для unlabeled? | Тип augmentation | Риски |
|-------|---------------------------|------------------|-------|
| Self-training | Нет (pseudo-labels) | Не нужна | Confirmation bias |
| Consistency Reg. | Нет | Data aug / adversarial | Нужна хорошая augmentation |
| VAT | Нет | Adversarial (learned) | Один гиперпараметр (eps) |
| LM Pre-training | Нет | N/A | Indirect signal |

## Assumptions semi-supervised learning

SSL работает **не всегда**. Он основан на assumptions о данных:

### 1. Smoothness Assumption
Если $x_1$ и $x_2$ близки в input space, их labels $y_1$ и $y_2$ должны совпадать. **Consistency regularization** напрямую эксплуатирует это.

### 2. Cluster Assumption
Данные образуют кластеры, и точки в одном кластере имеют одинаковый label. Decision boundary проходит через **low-density regions**. Self-training работает лучше, когда кластеры хорошо разделены.

### 3. Manifold Assumption
Данные лежат на low-dimensional manifold в high-dimensional space. Это позволяет использовать unlabeled данные для "выучивания" structure manifold.

**Если assumptions нарушены** (например, два класса перемешаны без кластерной структуры), SSL может **ухудшить** результат по сравнению с чистым supervised learning.

## SSL в эпоху BERT и LLM

После 2018 года semi-supervised learning в NLP **трансформировался**:

### Pre-BERT era (2013-2018)

SSL был **основным** способом использовать unlabeled данные:
- Word2Vec/GloVe: unsupervised word embeddings
- LM pre-training (Dai & Le, 2015): LSTM language model как инициализация
- VAT (Miyato et al., 2017): consistency regularization

### Post-BERT era (2018+)

BERT-style pre-training **поглотил** большинство SSL use-cases:
- Massive pre-training на unlabeled данных (MLM/CLM)
- Fine-tuning на labeled данных
- Это по сути **SSL в масштабе**: pre-training = unsupervised learning на manifold

**Но SSL не умер.** Современные применения:

1. **Узкие домены**: медицина, юриспруденция -- мало labeled, мало и unlabeled данных (не хватает для pre-training с нуля). Domain-adaptive pre-training + SSL fine-tuning.

2. **Few-shot с pseudo-labels**: обучи base BERT, получи pseudo-labels на unlabeled domain data, переобучи. Работает при <100 labeled примерах.

3. **Noisy Student** (Xie et al., 2020): обучи teacher, получи pseudo-labels с noise, обучи larger student. Улучшает ImageNet SOTA, работает и в NLP.

4. **MixText** (Chen et al., 2020): interpolation labeled + unlabeled в hidden space. Эффективен при <100 labeled примерах.

## Практический рецепт для SSL в NLP (2024+)

```
Есть labeled (< 10K) + unlabeled данные?
|
|-- Unlabeled в том же домене?
|   |-- Да, > 100K --> Domain-adaptive MLM pre-training
|   |                  затем fine-tune на labeled
|   |-- Да, < 100K --> VAT / UDA при fine-tuning
|
|-- Есть pre-trained модель для домена?
|   |-- Да --> Fine-tune + pseudo-labeling (iterative)
|   |-- Нет --> Pre-train на unlabeled + fine-tune
|
|-- Labeled < 100?
    |-- Self-training с высоким threshold
    |-- Или in-context learning (GPT-4) без SSL вообще
```

## Хронология

| Год | Milestone | Работа |
|-----|-----------|--------|
| 2006 | Semi-supervised Learning book | Chapelle et al. |
| 2013 | Pseudo-Label | Lee |
| 2015 | LM pre-training для SSL | Dai & Le |
| 2016 | VAT (images) | Miyato et al. |
| **2017** | **VAT для текста** | **Miyato, Dai, Goodfellow** |
| 2018 | BERT (pre-training поглощает SSL) | Devlin et al. |
| 2020 | UDA, MixText, FixMatch | Xie et al., Chen et al. |
| 2020 | Noisy Student | Xie et al. |

## Key papers

- [[02 Areas/ML & DL/Papers/Adversarial Training Methods for Semi-Supervised Text Classification]] -- VAT для текста (Miyato et al., ICLR 2017)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Adversarial Training|Adversarial Training]] -- VAT как метод consistency regularization
- [[02 Areas/ML & DL/Concepts/Training/Pre-training|Pre-training]] -- modern large-scale SSL
- [[02 Areas/ML & DL/Concepts/Training/Few-shot Fine-tuning|Few-shot Fine-tuning]] -- low-data fine-tuning, часто дополняется SSL
- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling]] -- pre-training = SSL в масштабе

## Дополнительные ресурсы

- [Miyato et al. -- AT/VAT для текста (ICLR 2017)](https://arxiv.org/abs/1605.07725) -- оригинал
- [Xie et al. -- UDA (NeurIPS 2020)](https://arxiv.org/abs/1904.12848) -- data augmentation + consistency
- [Chapelle et al. -- Semi-Supervised Learning (book)](https://www.molgen.mpg.de/3659531/MITPress--SemisupervisedLearning.pdf) -- фундаментальный учебник
