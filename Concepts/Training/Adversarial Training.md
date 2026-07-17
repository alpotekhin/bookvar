---
title: "Adversarial Training"
aliases: [adversarial training for text, virtual adversarial training, VAT, FGM, состязательное обучение]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/Adversarial Training Methods for Semi-Supervised Text Classification]]"
courses: []
sources:
  - "[Miyato et al. -- Adversarial Training Methods for Semi-Supervised Text Classification (ICLR 2017)](https://arxiv.org/abs/1605.07725)"
  - "[Goodfellow et al. -- Explaining and Harnessing Adversarial Examples (FGSM, 2015)](https://arxiv.org/abs/1412.6572)"
  - "[Zhu et al. -- FreeLB (2020)](https://arxiv.org/abs/1909.11764)"
  - "[Jiang et al. -- SMART (2020)](https://arxiv.org/abs/1911.03437)"
---

# Adversarial Training

## Зачем это нужно: регуляризация через worst-case

Нейросети удивительно хрупки: **минимальные изменения** входа могут кардинально изменить предсказание. В CV это adversarial images (добавь невидимый шум -- и панда становится гиббоном). В NLP проблема аналогична: модель может быть уверена в ответе, но малое возмущение в embedding space разрушает предсказание.

**Adversarial Training** -- метод регуляризации: при обучении находим **наихудшее малое возмущение** входа (adversarial perturbation) и учим модель быть к нему устойчивой. Это делает decision boundary модели **более гладкой** и улучшает generalization.

Miyato, Dai & Goodfellow (ICLR 2017) адаптировали adversarial training для текста, решив ключевую проблему: текст **дискретен**, а adversarial perturbations требуют **непрерывных** входов.

## Ключевая идея: perturbation в embedding space

### Проблема дискретности текста

В CV adversarial perturbation $r$ добавляется к пикселям: $x' = x + r$. В NLP вход -- это **one-hot вектора** (дискретные), к которым нельзя добавить непрерывное возмущение. Нельзя "немного изменить" слово "cat" -- оно либо "cat", либо другое слово.

### Решение: возмущение word embeddings

Вместо дискретного входа, perturbation применяется к **word embeddings** -- непрерывным векторам, в которые преобразуются слова:

```
Стандартный проход:
  words --> [embedding lookup] --> v1, v2, v3 --> LSTM --> prediction

Adversarial проход:
  words --> [embedding lookup] --> v1+r1, v2+r2, v3+r3 --> LSTM --> prediction
                                   ↑ adversarial perturbation
```

Возмущённый embedding **не соответствует никакому реальному слову** -- это точка в embedding space между словами. Поэтому adversarial training для текста -- не защита от реального adversary (который может менять только слова), а **регуляризация** через smoothing classification function.

## Формализация

### Стандартный Adversarial Training (AT)

Для классификатора $p(y|s; \theta)$ с concatenated embeddings $s = [\bar{v}^{(1)}, \bar{v}^{(2)}, \ldots, \bar{v}^{(T)}]$:

Adversarial perturbation -- linearized worst-case:

$$r_{adv} = -\epsilon \frac{g}{\|g\|_2}, \quad g = \nabla_s \log p(y \mid s; \hat{\theta})$$

где $\hat{\theta}$ -- текущие параметры (заморожены при вычислении $r_{adv}$), $\epsilon$ -- единственный гиперпараметр (норма возмущения).

Adversarial loss:

$$\mathcal{L}_{adv}(\theta) = -\frac{1}{N} \sum_{n=1}^{N} \log p(y_n \mid s_n + r_{adv,n}; \theta)$$

**Полный loss**: $\mathcal{L} = \mathcal{L}_{ce} + \mathcal{L}_{adv}$ -- стандартный cross-entropy + adversarial term.

**Важно**: $\hat{\theta}$ -- это константа. Градиенты **не проходят** через процесс создания adversarial примеров. Обратное распространение идёт только через $\theta$ в $\mathcal{L}_{adv}$.

### Virtual Adversarial Training (VAT)

Ключевое расширение: VAT **не требует label** $y$. Вместо этого VAT минимизирует **KL-дивергенцию** между предсказаниями на чистом и возмущённом входе:

$$\mathcal{L}_{vat} = KL\left[p(\cdot \mid s; \hat{\theta}) \;\|\; p(\cdot \mid s + r_{v\text{-}adv}; \theta)\right]$$

$$r_{v\text{-}adv} = \arg\max_{\|r\| \leq \epsilon} KL\left[p(\cdot \mid s; \hat{\theta}) \;\|\; p(\cdot \mid s + r; \hat{\theta})\right]$$

**Интуиция**: VAT находит направление, в котором output distribution модели **наиболее чувствительна**, и учит модель быть **smooth** (одинаковый output) в этом направлении.

**Главное преимущество**: VAT работает на **unlabeled данных** -- открывает semi-supervised learning.

### Нормализация embeddings

Без ограничений модель может тривиально "обнулить" adversarial perturbations, обучив embeddings с **очень большой нормой**: если $\|v\| = 10^6$, то $\epsilon$-возмущение незаметно. Решение -- **нормализация** по статистике корпуса:

$$\bar{v}_k = \frac{v_k - E(v)}{\sqrt{Var(v)}}, \quad E(v) = \sum_j f_j v_j$$

где $f_j$ -- частота $j$-го слова.

## Алгоритм одного шага обучения

```
1. Forward pass: вычислить стандартный loss L_ce
2. Backward pass: получить градиенты по embeddings g = grad(L_ce, embeddings)
3. Вычислить adversarial perturbation: r_adv = -eps * g / ||g||_2
4. Forward pass с возмущёнными embeddings: L_adv
5. [Если VAT]: Forward pass на unlabeled данных:
   - Найти r_v-adv через power iteration (~1 шаг)
   - Вычислить L_vat = KL[p(.|s) || p(.|s + r_v-adv)]
6. Total loss = L_ce + L_adv + alpha * L_vat
7. Backward pass и update параметров
```

## Ключевые результаты

### IMDB Sentiment (Table 2 из статьи)

| Метод | Test Error Rate |
|-------|----------------|
| Baseline LSTM | 7.39% |
| + Adversarial Training | 6.21% |
| + **Virtual Adversarial Training** | **5.91%** |
| + Adv + VAT | 6.09% |
| SOTA (One-hot bi-LSTM) | 5.94% |

VAT (5.91%) **лучше SOTA** (5.94%) -- при том, что использует более простую архитектуру.

### Elec и RCV1 (semi-supervised)

| Датасет | Baseline | + Adv + VAT |
|---------|----------|-------------|
| Elec | 6.24% | **5.40%** (-13%) |
| RCV1 | 7.40% | **6.71%** (-9%) |

### Качество word embeddings (Table 3)

Adversarial training **качественно улучшает** обученные embeddings:

```
Ближайшие соседи к "good":
  Baseline:     great, bad, well, decent, nice
  Adversarial:  decent, great, nice, entertaining, solid

"bad" как 3-й ближайший к "good" (baseline) --> исчезает (adversarial)
```

Adversarially trained embeddings лучше разделяют семантически противоположные слова.

## Эволюция: от LSTM к Transformer

Оригинальная работа Miyato et al. использовала LSTM (pre-Transformer era). Современные методы адаптировали те же принципы для Transformer/BERT:

| Метод | Год | Модель | Ключевая идея |
|-------|-----|--------|---------------|
| AT/VAT (Miyato) | 2017 | LSTM | Оригинал: perturbation в embedding space |
| **FGM** (Fast Gradient Method) | 2017 | Любая | Один шаг gradient-based perturbation |
| **PGD** (Projected Gradient Descent) | 2018 | Любая | Multi-step adversarial, сильнее FGM |
| **FreeLB** (Zhu et al.) | 2020 | BERT | PGD с accumulated gradients, экономит compute |
| **SMART** (Jiang et al.) | 2020 | BERT | Smoothness + Bregman proximal regularization |
| **R-Drop** | 2021 | Любая | KL между двумя dropout masks (implicit adversarial) |

### FreeLB: adversarial training без дополнительного cost

FreeLB делает $K$ шагов PGD, но **аккумулирует градиенты** модели на каждом шаге. Результат: adversarial training практически **бесплатно** (в отличие от AT, которое удваивает compute):

```
Стандартный AT:
  Forward (clean) --> grad --> Forward (adv) --> grad --> Update
  Cost: 2x forward + 2x backward

FreeLB (K=3):
  Forward (adv_0) --> grad (accumulate)
  Forward (adv_1) --> grad (accumulate)
  Forward (adv_2) --> grad (accumulate)
  Update (averaged gradients)
  Cost: ~K forward + K backward, но NO clean forward
```

FreeLB на BERT: +0.8% на GLUE, +1.1% на ARC, +1.3% на CommonsenseQA.

## Практические рекомендации

### Когда использовать

1. **Fine-tuning на малых данных** (<10K примеров) -- adversarial training работает как сильный regularizer
2. **Robustness-critical задачи** -- модерация, токсичность, медицина
3. **Semi-supervised** -- VAT на unlabeled данных
4. **Соревнования** -- +0.5-1.5% на GLUE/SuperGLUE

### Гиперпараметры

```
epsilon: 1.0 (для normalized embeddings)
         0.01-0.1 (для un-normalized)
Метод: FGM (быстрый, 1 шаг) или FreeLB (K=3, лучше но дороже)
alpha_vat: 1.0 (вес VAT loss)
```

### Когда НЕ использовать

- Большие датасеты (>100K) -- прирост минимален, compute не оправдан
- Generative tasks (CLM) -- adversarial training плохо работает для генерации
- Когда compute budget ограничен и нужен максимум experiments

## Key papers

- [[02 Areas/ML & DL/Papers/Adversarial Training Methods for Semi-Supervised Text Classification]] -- оригинал: AT/VAT для текста (Miyato, Dai, Goodfellow, ICLR 2017)

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Semi-supervised Learning|Semi-supervised Learning]] -- VAT -- один из основных semi-supervised методов
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] -- adversarial training как регуляризация при fine-tuning
- [[02 Areas/ML & DL/Concepts/Training/Few-shot Fine-tuning|Few-shot Fine-tuning]] -- adversarial training особенно полезен при малых данных
- [[02 Areas/ML & DL/Concepts/Architectures/BERT|BERT]] -- основная модель для modern adversarial training (FreeLB, SMART)

## Дополнительные ресурсы

- [Miyato et al. -- оригинальная статья (ICLR 2017)](https://arxiv.org/abs/1605.07725)
- [Zhu et al. -- FreeLB (ICLR 2020)](https://arxiv.org/abs/1909.11764) -- современный adversarial training для BERT
- [Goodfellow et al. -- Explaining and Harnessing Adversarial Examples (2015)](https://arxiv.org/abs/1412.6572) -- фундаментальная работа по adversarial examples
