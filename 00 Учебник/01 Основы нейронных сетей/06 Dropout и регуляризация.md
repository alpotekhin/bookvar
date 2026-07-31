---
title: Dropout и регуляризация нейросетей
type: textbook-chapter
status: canonical
last_updated: 2026-07-31
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

### Один forward pass в числах

Пусть скрытый слой вернул $h=(2,-1,3,4)$, а $p=0.5$. Для маски
$m=(1,0,1,0)$ на обучении получим

$$
h'=\frac{m\odot h}{1-p}=(4,0,6,0).
$$

В этом проходе сумма активаций изменилась с $8$ до $10$: inverted dropout не
обещает сохранить конкретный вектор. Но каждая координата остаётся с
вероятностью $0.5$ и при этом удваивается, поэтому по множеству масок
$\mathbb E[h'_j]=h_j$. Следующий слой обучается и на варианте выше, и на
вариантах вроде $(0,-2,0,8)$. Он не может закрепить решение за одной постоянно
доступной координатой.

При inference используется исходный $h$, без случайной маски. Это приближает
усреднение предсказаний большого числа подсетей одной детерминированной сетью.
Приближение не является точным равенством для глубокой нелинейной модели, но
оно объясняет выбранное масштабирование.

## Weight decay

L2 regularization добавляет штраф

$$
J(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2.
$$

В SGD это приводит к multiplicative shrinkage. В адаптивных оптимизаторах
decoupled weight decay (AdamW) отделяет уменьшение веса от масштабирования
градиента. Bias и параметры normalization часто исключают из weight decay:
штрафовать scale/shift тем же способом не всегда осмысленно.

Для обычного SGD градиент L2-штрафа равен $\lambda\theta$, поэтому обновление
можно переписать как

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\nabla L(\theta_t).
$$

Здесь L2 regularization и weight decay совпадают. В Adam координатное
масштабирование градиента меняет и добавку $\lambda\theta$: параметры с разной
историей градиентов уменьшаются по-разному. AdamW сначала делает адаптивный шаг
по loss, а shrinkage применяет отдельно:

$$
\theta_{t+1}=(1-\eta\lambda)\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Поэтому фразы «L2 penalty» и «decoupled weight decay» нельзя считать
синонимами, не указав optimizer.

## Data augmentation

Augmentation задаёт преобразования, при которых label должен сохраниться. Для
изображений это crop, flip, color jitter; для аудио — шум или временной сдвиг.
В NLP безопасных преобразований меньше: синоним может изменить регистр, факт или
намерение, а перестановка слов — синтаксис. Поэтому текстовые augmentation нужно
валидировать на конкретной задаче.

Для LLM основной аналог — разнообразие и качество обучающих данных, смешивание
доменных источников, synthetic generation с фильтрацией и curricula. Сам факт
увеличения корпуса не гарантирует регуляризации, если добавляются дубликаты.

![[Assets/Sources/Harvard ML Systems/vol2/robust_ai/data_augmentation.png|Примеры допустимых преобразований изображения]]

*Исходный объект и варианты со сдвигом и поворотом сохраняют класс цифры.
Рисунок полезен именно как проверка инвариантности: преобразование считается
augmentation только пока метка остаётся корректной. Источник: Harvard CS249r,
[Robust AI](https://github.com/harvard-edge/cs249r_book/tree/main/book/quarto/contents/vol2/robust_ai),
CC BY-NC-SA 4.0.*

## Early stopping

Checkpoint выбирают по validation metric, а обучение прекращают, когда улучшение
не продолжается. Нужно сохранять именно лучшую версию, а не последнюю. Patience
защищает от остановки на случайном шуме, но многократный ручной подбор patience
по одному validation set тоже адаптирует процедуру к нему.

Early stopping ограничивает не норму параметров напрямую, а число шагов,
за которое модель успевает подстроиться под выборку. Типичная траектория выглядит
так: training loss продолжает падать, validation loss достигает минимума и затем
растёт. Останавливать нужно не в момент первого ухудшения, а восстанавливать
checkpoint с лучшей заранее выбранной validation metric. Test set при выборе
момента остановки не используется.

## Как выбрать ограничение по наблюдаемой ошибке

Сначала нужно назвать failure mode. Большой разрыв между training и validation
при хорошем training score указывает на variance: помогают дополнительные
данные, augmentation, weight decay, dropout или более ранняя остановка. Если
плохи обе метрики, усиление регуляризации обычно только увеличит underfitting;
сначала нужны более выразительная модель, признаки, срок обучения или исправление
оптимизации. Если validation хорош, а production ухудшается, проблема может быть
в distribution shift — dropout её сам по себе не решает.

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
