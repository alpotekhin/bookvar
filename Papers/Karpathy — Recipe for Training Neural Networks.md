---
title: "A Recipe for Training Neural Networks"
url: https://karpathy.github.io/2019/04/25/recipe/
authors: [Andrej Karpathy]
year: 2019
date_reviewed: 2026-04-07
type: paper-review
category: paper
tags:
  - training
  - debugging
  - best-practices
  - Karpathy
concepts:
  - "[[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]"
  - "[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]"
raw: "[[02 Areas/ML & DL/raw/articles/karpathy/recipe-for-training-nns.html]]"
---

# A Recipe for Training Neural Networks

**Author:** Andrej Karpathy
**Published:** April 2019 (blog post)
**URL:** https://karpathy.github.io/2019/04/25/recipe/

## TL;DR

Практическое руководство по отладке и обучению нейросетей. Главный тезис: **"fast and furious" подход не работает** -- нужна системная, пошаговая методология. Пост описывает 6-этапный рецепт: от изучения данных до финального squeeze. Один из самых цитируемых практических постов в deep learning.

## Два фундаментальных наблюдения

### 1. Neural net training -- leaky abstraction

30-строчные примеры из библиотек создают **ложное** впечатление plug-and-play. В реальности:
- Backprop + SGD не "магически" заставит сеть работать
- BatchNorm не "магически" ускорит сходимость
- RNN не позволяет просто "подключить" текст
- Нужно **понимать**, что происходит под капотом

### 2. Neural net training fails silently

- Синтаксических ошибок нет -- код запускается
- Но сеть может обучаться **неправильно** и всё равно давать какой-то результат
- Примеры тихих багов: перевернутые labels при augmentation, off-by-one в autoregressive model, clip loss вместо clip gradients, неправильная инициализация из pretrained checkpoint

## Рецепт (6 этапов)

### 1. Become one with the data

- Потратить **часы** на визуальный осмотр данных
- Искать: дубликаты, corrupted примеры, label noise, data imbalance
- Понять: нужны ли локальные или глобальные features? Сколько вариации? Какая шумная?
- Написать код для search/filter/sort, визуализировать outliers

### 2. Set up end-to-end skeleton + dumb baselines

Начать с **простейшей** модели (linear classifier / tiny ConvNet). Чеклист:

| Tip | Зачем |
|-----|-------|
| **Fix random seed** | Воспроизводимость |
| **Disable augmentation** | Убрать источник багов |
| **Verify loss @ init** | Должен быть `-log(1/n_classes)` для softmax |
| **Init well** | Bias = mean of targets, учёт class imbalance |
| **Overfit one batch** | Убедиться что loss -> 0 на 2-5 примерах |
| **Visualize before the net** | Проверить что именно подаётся в модель |
| **Backprop to chart dependencies** | Убедиться что пример i зависит только от входа i |
| **Generalize a special case** | Сначала loop, потом vectorize |

### 3. Overfit

- Два этапа: (a) достичь низкого training loss, (b) затем регуляризовать
- **Don't be a hero**: copy-paste архитектуру из лучшей paper (напр. ResNet-50)
- **Adam is safe**: начинать с Adam LR=3e-4 (прощает плохие hyperparams)
- **Complexify one at a time**: добавлять сигналы/features по одному
- **НЕ** доверять default learning rate decay (он может быть для ImageNet epochs)

### 4. Regularize

Отдать часть training accuracy ради validation accuracy:

1. **Get more data** -- единственный гарантированный способ улучшить результат
2. **Data augment** -- полу-фейковые данные
3. **Creative augmentation** -- domain randomization, simulation, GANs
4. **Pretrain** -- почти никогда не вредит
5. **Stick with supervised learning** -- unsupervised pretraining (2008-эра) не работает в CV (но NLP/BERT -- другая история)
6. **Smaller input** -- убрать spurious features
7. **Smaller model** -- если можно (напр. avg pooling вместо FC)
8. **Decrease batch size** -- BatchNorm с маленьким batch = stronger regularization
9. **Dropout** -- осторожно с BatchNorm (они конфликтуют)
10. **Weight decay** -- увеличить
11. **Early stopping** -- остановиться по validation loss
12. **Try larger model** -- с early stopping может быть лучше маленькой

### 5. Tune

- **Random search > grid search** -- сеть чувствительнее к одним параметрам, чем к другим
- Bayesian optimization toolboxes существуют, но "state of the art -- an intern" (шутка)

### 6. Squeeze out the juice

- **Ensembles** -- +2% accuracy гарантированно, можно distill через dark knowledge
- **Leave it training** -- сети обучаются неинтуитивно долго. "I accidentally left a model training during winter break and when I got back in January it was SOTA"

## Ключевые цитаты

> "A 'fast and furious' approach to training neural networks does not work and only leads to suffering."

> "The qualities that in my experience correlate most strongly to success in deep learning are **patience and attention to detail**."

## Мой чеклист (выжимка)

- [ ] Провести EDA данных (часы, не минуты)
- [ ] Начать с простейшей модели, verify loss @ init
- [ ] Overfit на 1 batch (loss -> 0)
- [ ] Visualize data непосредственно перед моделью
- [ ] Backprop check: градиент только от своего примера
- [ ] Overfit full dataset (focus on training loss)
- [ ] Regularize: data > augment > pretrain > dropout/wd
- [ ] Random search гиперпараметров
- [ ] Ensemble (если нужно)

## Связанные заметки

- [[02 Areas/ML & DL/Papers/Karpathy — Unreasonable Effectiveness of RNNs|Karpathy — Unreasonable Effectiveness of RNNs]]
- [[02 Areas/ML & DL/Concepts/Training/nanoGPT|nanoGPT]]
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]]
- [[02 Areas/ML & DL/Concepts/Training/RLHF|RLHF]]
