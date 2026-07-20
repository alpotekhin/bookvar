---
title: Dropout и регуляризация нейросетей
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://d2l.ai/chapter_multilayer-perceptrons/dropout.html
  - https://cs231n.github.io/neural-networks-2/
---

# Dropout и регуляризация нейросетей

Регуляризация не является одной операцией. Это любое ограничение, которое
помогает предпочесть решение с лучшим поведением на новых данных: штраф на веса,
случайное удаление активаций, augmentation, early stopping или изменение
обучающего распределения. Dropout особенно полезен как пример, потому что его
forward pass различается между training и inference.

## Inverted dropout

Для активации $h$ и вероятности удаления $p$ генерируется Bernoulli mask $m$:

$$
h'=\frac{m}{1-p}h,
\qquad
m\sim\operatorname{Bernoulli}(1-p).
$$

Деление на $1-p$ сохраняет математическое ожидание:

$$
\mathbb{E}[h']=h.
$$

На inference mask не используется и дополнительного масштабирования не нужно.
Именно поэтому забытый `model.eval()` меняет предсказания: dropout продолжает
случайно выключать элементы, а BatchNorm может продолжить обновлять статистики.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/d2l/dropout.svg]]

*Один и тот же MLP до и после применения dropout к скрытым активациям. Источник:
Zhang et al., [D2L: Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html),
CC BY-SA 4.0.*

Dropout не удаляет веса из модели. На каждом шаге он выбирает другую подсеть
активаций, а все подсети делят параметры. Следующий слой не может полагаться на
неизменное присутствие одного признака.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-7-14/cs231n-dropout-train.jpeg]]

*Слева полная сеть, справа — одна случайно выбранная обучающая подсеть. На
следующем шаге маска будет другой, но веса оставшихся связей останутся общими.
Источник: Stanford CS231n,
[Regularization](https://cs231n.github.io/neural-networks-2/#reg), прямая
[ссылка](https://cs231n.github.io/assets/nn2/dropout.jpeg).*

```python
def dropout(x, p, training):
    if not training or p == 0:
        return x
    keep = (torch.rand_like(x) >= p)
    return keep * x / (1 - p)
```

Полезно вручную проверить две крайние точки. При $p=0$ операция тождественна.
При $p\to1$ множитель $1/(1-p)$ растёт, поэтому дисперсия обучающих активаций
становится большой, хотя среднее сохраняется. Равенство математических ожиданий
не означает одинаковых распределений на обучении и инференсе: именно этот шум
и создаёт регуляризующий эффект.

## Weight decay

L2 regularization добавляет штраф

$$
J(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2.
$$

В SGD это приводит к multiplicative shrinkage. В адаптивных оптимизаторах
decoupled weight decay (AdamW) отделяет уменьшение веса от масштабирования
градиента. Bias и параметры normalization часто исключают из weight decay:
штрафовать scale/shift тем же способом не всегда осмысленно.

## Data augmentation

Augmentation задаёт преобразования, при которых label должен сохраниться. Для
изображений это crop, flip, color jitter; для аудио — шум или временной сдвиг.
В NLP безопасных преобразований меньше: синоним может изменить регистр, факт или
намерение, а перестановка слов — синтаксис. Поэтому текстовые augmentation нужно
валидировать на конкретной задаче.

Для LLM основной аналог — разнообразие и качество обучающих данных, смешивание
доменных источников, synthetic generation с фильтрацией и curricula. Сам факт
увеличения корпуса не гарантирует регуляризации, если добавляются дубликаты.

## Early stopping

Checkpoint выбирают по validation metric, а обучение прекращают, когда улучшение
не продолжается. Нужно сохранять именно лучшую версию, а не последнюю. Patience
защищает от остановки на случайном шуме, но многократный ручной подбор patience
по одному validation set тоже адаптирует процедуру к нему.

## Регуляризация должна соответствовать режиму

| Режим | Часто полезно | Что проверить |
|---|---|---|
| маленький supervised dataset | augmentation, transfer learning, weight decay, dropout | leakage и class balance |
| pre-training большой LLM | data quality/dedup, weight decay, schedule | memorization, contamination |
| SFT | небольшой LR, mixture с general data, early stopping | capability regression |
| LoRA/PEFT | rank, dropout адаптеров, target modules | under/overfitting adapters |
| RL/post-training | KL/reference constraints, reward validation | reward overoptimization |

Dropout не обязателен во всех современных LLM. При огромных данных и других
ограничениях многие foundation models обучаются с нулевым dropout. Это не делает
метод устаревшим; меняется баланс источников overfitting и stochasticity.

## Что нужно унести из главы

- Inverted dropout сохраняет expectation и активен только при training.
- Weight decay ограничивает параметры, dropout — зависимость от конкретных
  активаций, augmentation — чувствительность к допустимым преобразованиям.
- `train()` и `eval()` меняют семантику dropout и BatchNorm.
- Regularizer выбирают по данным и режиму, а не по универсальному рецепту.
- Контрольная метрика остаётся обязательной: сильная регуляризация может вызвать
  underfitting.

## Источники

- D2L, [Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html).
- Srivastava et al., [Dropout](https://jmlr.org/papers/v15/srivastava14a.html).
- Stanford CS231n, [Regularization](https://cs231n.github.io/neural-networks-2/#reg).
- Loshchilov, Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).

**Дальше:** convolution вводит структурное ограничение другого типа — локальность
и общие веса для всех пространственных позиций.
