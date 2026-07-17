---
title: "REALM"
aliases: [REALM, Retrieval-Augmented Language Model Pre-Training, Retrieval-Augmented Language Model]
type: concept
category: Retrieval
papers: ["[[02 Areas/ML & DL/Papers/REALM|REALM]]"]
courses: []
sources:
  - "[Guu et al. — REALM: Retrieval-Augmented Language Model Pre-Training (ICML 2020)](https://arxiv.org/abs/2002.08909)"
---

# REALM

## Зачем это нужно: знания в параметрах vs знания в документах

Языковые модели (BERT, GPT) хранят знания **имплицитно** — в весах нейронной сети. Чтобы «помнить» больше фактов, нужна модель побольше. T5-11B запоминает много, но:

- Непонятно, **какие** факты модель запомнила, а какие нет
- Знания **устаревают** с момента обучения (knowledge cutoff)
- Чтобы обновить один факт, нужно **переобучить** всю модель
- Хранение в параметрах **неэффективно**: 11B параметров для знаний, которые помещаются в текстовый файл

Guu et al. (Google Research, 2020) предложили **REALM** — фреймворк, в котором языковая модель **учится извлекать** документы из Wikipedia на этапе **pre-training**, а не только на fine-tuning. Retriever обучается end-to-end через MLM loss.

Ключевой результат: **REALM 330M бьёт T5-11B** на Open-domain QA. Модель в **30 раз меньше**, но с доступом к external knowledge.

## Архитектура: retrieve-then-predict

### Generative process

REALM моделирует $p(y|x)$ как двухшаговый процесс с latent variable $z$ (документ):

$$p(y|x) = \sum_{z \in \mathcal{Z}} p(y|z, x) \cdot p(z|x)$$

где $\mathcal{Z}$ — весь корпус Wikipedia (~13M пассажей), $x$ — input (замаскированное предложение или вопрос), $y$ — target (замаскированные токены или ответ).

Интуиция: чтобы предсказать замаскированное слово, модель **извлекает** документ, который может помочь, и **использует** информацию из него для предсказания.

### Knowledge Retriever $p(z|x)$

Dense inner product модель (bi-encoder):

$$f(x, z) = \text{Embed}_{\text{input}}(x)^T \cdot \text{Embed}_{\text{doc}}(z)$$

$$p(z|x) = \text{softmax}(\text{relevance scores})$$

где:
$$\text{Embed}_{\text{input}}(x) = W_{\text{input}} \cdot \text{BERT}_{\text{CLS}}(x)$$
$$\text{Embed}_{\text{doc}}(z) = W_{\text{doc}} \cdot \text{BERT}_{\text{CLS}}(z_{\text{title}}, z_{\text{body}})$$

Два BERT-based энкодера (query и document) с линейной проекцией. Relevance score = dot product в embedding space. Top-k retrieval через **MIPS** (Maximum Inner Product Search).

### Knowledge-Augmented Encoder $p(y|z,x)$

Отдельный Transformer на конкатенации $x$ и $z_{\text{body}}$:

**Pre-training** (MLM):
$$p(y_j | z, x) \propto \exp(w_j^T \cdot \text{BERT}_{\text{MASK}(j)}(\text{join}(x, z_{\text{body}})))$$

**Fine-tuning** (Open-QA):
$$p(y | z, x) \propto \sum_{s \in S(z,y)} \exp(\text{MLP}(h_{\text{START}(s)}; h_{\text{END}(s)}))$$

где $S(z, y)$ — множество спанов в $z$, совпадающих с ответом $y$.

**Rich cross-attention**: в отличие от bi-encoder retriever'а, augmented encoder выполняет полное cross-attention между input и документом — это позволяет глубоко интегрировать информацию.

## Training: end-to-end через MLM

### Marginal likelihood

Обучение максимизирует $\log p(y|x)$:

$$\nabla \log p(y|x) = \sum_{z \in \mathcal{Z}} r(z) \nabla f(x, z)$$

где:
$$r(z) = \left(\frac{p(y|z,x)}{p(y|x)} - 1\right) p(z|x)$$

**Что retriever учит**: документ $z$ получает **положительный** update, если $p(y|z,x) > p(y|x)$ — то есть если этот документ помогает предсказать ответ **лучше среднего**. Документ, который хуже среднего, получает отрицательный update.

Это **unsupervised**: не нужны пары «запрос-документ» с разметкой. MLM signal сам учит retriever находить полезные документы.

### Approximation: top-k вместо суммы по всему корпусу

Суммирование по **всем** 13M документам нереально. Аппроксимация: суммируем по top-$k$ документам ($k=8$ pre-training, $k=5$ fine-tuning). Это разумно, если большинство документов имеют near-zero probability.

Для нахождения top-$k$ используем MIPS по pre-computed document embeddings.

### Asynchronous MIPS Index Refresh

**Проблема**: document embeddings pre-computed, но параметры retriever'а обновляются каждый gradient step → index «устаревает».

**Решение**: два параллельных процесса:
1. **Primary trainer**: обучает модель, использует текущий MIPS index
2. **Secondary index builder**: асинхронно пересчитывает все document embeddings и перестраивает index

Index обновляется каждые **~500 training steps**. Между обновлениями index слегка устарел, но top-$k$ retrieval всё ещё приблизительно верный. Свежий $\theta$ используется для пересчёта $p(z|x)$ **после** retrieval — только для отобранных $k$ документов.

**Критичность refresh'а**: без асинхронного обновления EM падает на **-9.5 points** — retriever не учится.

## Inductive biases: трюки, без которых не работает

### Salient Span Masking

Стандартный random masking маскирует предлоги, артикли, стоп-слова — для них retrieval бесполезен. REALM маскирует **named entities** и **даты** (определяются BERT-based NER tagger + regex):

```
"The [MASK] is the currency of the United Kingdom"  ← salient span
"The currency [MASK] the United Kingdom"             ← local context enough
```

Эффект: **+5.9 EM** по сравнению с random masking.

### Null Document

Не все замаскированные токены требуют external knowledge. Добавляем пустой **null document** $\emptyset$ в top-$k$ — модель может «сказать», что retrieval не нужен, присвоив высокий вес пустому документу.

### Запрет trivial retrievals

Если pre-training корпус = knowledge корпус (оба = Wikipedia), модель может найти **ту же самую** страницу, из которой взят $x$, и тривиально восстановить замаскированный токен. Это учит retriever'а искать **exact string match**, а не семантическую релевантность. Решение: исключаем исходный документ из кандидатов.

### Warm-start через Inverse Cloze Task

**Cold-start проблема**: в начале обучения retriever извлекает случайные документы → encoder учится **игнорировать** retrieved documents → retriever не получает gradient → порочный круг.

Решение: warm-start retriever через **ICT** (Inverse Cloze Task): дано предложение, найди документ, из которого оно взято. Knowledge-augmented encoder warm-start'ится от uncased BERT-base.

## Результаты

| Benchmark | REALM (330M) | ORQA | T5-Large (770M) | T5-11B |
|-----------|-------------|------|-----------------|--------|
| **NaturalQuestions** | **40.4** | 33.3 | 29.8 | 36.6 |
| **WebQuestions** | **40.7** | 36.4 | 37.4 | 44.7* |
| **CuratedTrec** | **46.8** | 30.1 | — | — |

(*T5-11B результаты на WQ из concurrent work Roberts et al.)

REALM 330M **бьёт T5-11B** на NaturalQuestions — модель в 30x меньше, но с retrieval-augmented pre-training. На всех трёх бенчмарках REALM — **SOTA**, превосходя все предыдущие системы на **4-16% absolute accuracy**.

## Почему REALM важен

### Первый end-to-end learned retriever для pre-training

До REALM retriever'ы были либо heuristic (BM25), либо learned но только при fine-tuning (ORQA). REALM показал, что **MLM signal достаточен** для обучения качественного retriever'а — без supervised пар «запрос-документ».

### Аргумент за retrieval vs scaling

REALM 330M > T5-11B — **сильнейший** аргумент, что retrieval-augmented подход может быть эффективнее наращивания параметров. Вместо того чтобы запоминать все факты в весах, дай модели доступ к базе знаний.

### Прямой предшественник RAG, RETRO, Atlas

REALM заложил фундамент для:
- **RAG** (Lewis et al., 2020): retrieval-augmented generation (seq2seq вместо extractive)
- **RETRO** (Borgeaud et al., 2022): retrieval-augmented pre-training в масштабе Chinchilla
- **Atlas** (Izacard et al., 2022): few-shot learning with retrieval

### Interpretability и modularity

Retriever делает знания **явными**: можно посмотреть, какие документы модель использовала для каждого предсказания. Можно обновить knowledge corpus **без переобучения** модели (просто пересчитать index).

## Ограничения

- **Computational cost**: асинхронный MIPS refresh требует значительных ресурсов (пересчёт 13M embeddings)
- **Только extractive**: fine-tuning на Open-QA ищет span в документе — не может генерировать ответы (RAG решил эту проблему)
- **Knowledge corpus = Wikipedia**: привязан к одному источнику
- **Сложность имплементации**: два параллельных процесса, MIPS index management

## Key papers

- [[02 Areas/ML & DL/Papers/REALM|REALM]] — retrieval-augmented pre-training, end-to-end обучение retriever через MLM loss

## Related concepts

- [[02 Areas/ML & DL/Concepts/Retrieval/Retrieval-Augmented Generation|RAG]] — следующий шаг: generative вместо extractive
- [[02 Areas/ML & DL/Concepts/Retrieval/Dense Retrieval|Dense Retrieval]] — bi-encoder architecture, используемая в REALM
- [[02 Areas/ML & DL/Concepts/Retrieval/Self-RAG|Self-RAG]] — adaptive retrieval с reflection tokens
- [[02 Areas/ML & DL/Concepts/Training/Masked Language Modeling|Masked Language Modeling]] — training objective REALM
- [[02 Areas/ML & DL/Concepts/Retrieval/Open-domain QA|Open-domain QA]] — основной бенчмарк для REALM
- [[02 Areas/ML & DL/Concepts/Retrieval/Knowledge-Intensive NLP|Knowledge-Intensive NLP]] — класс задач, для которых REALM создан

## Дополнительные ресурсы

- [Guu et al. — REALM (ICML 2020)](https://arxiv.org/abs/2002.08909) — оригинальная статья
- [Google AI Blog — REALM](https://ai.googleblog.com/2020/08/realm-integrating-retrieval-into.html) — блог-пост от авторов
- [Lilian Weng — RAG (2024)](https://lilianweng.github.io/posts/2024-07-07-rest-of-rag/) — REALM в контексте эволюции RAG
