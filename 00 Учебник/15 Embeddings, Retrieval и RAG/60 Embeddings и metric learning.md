---
title: "Эмбеддинги и метрическое обучение"
type: textbook-chapter
status: canonical
last_updated: 2026-11-22
last_verified: 2026-11-22
source_language: mixed
previous: "[[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация]]"
next: "[[00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid]]"
primary_sources:
  - https://arxiv.org/abs/1908.10084
  - https://arxiv.org/abs/2004.04906
  - https://aclanthology.org/2021.repl4nlp-1.31/
  - https://arxiv.org/abs/2406.12356v3
  - https://www.sbert.net/docs/sentence_transformer/loss_overview.html
  - https://github.com/princeton-nlp/SimCSE
  - https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html
  - https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
reference_terms:
  - label: "Mean, CLS и last-token pooling"
    label_en: "Mean, CLS and last-token pooling"
    anchor: emb-pooling
  - label: "Косинус, скалярное произведение и норма"
    label_en: "Cosine, dot product and vector norms"
    anchor: emb-similarity
  - label: "Инструкции, токенизация и обрезка"
    label_en: "Instructions, tokenization and truncation"
    anchor: emb-input-contract
  - label: "SBERT, bi-encoder и cross-encoder"
    label_en: "SBERT, bi-encoders and cross-encoders"
    anchor: emb-encoders
  - label: "Контрастивное обучение эмбеддингов"
    label_en: "Contrastive embedding training"
    anchor: emb-contrastive
  - label: "Контрастивный пакет: численный пример"
    label_en: "Contrastive batch: worked example"
    anchor: emb-batch-example
  - label: "Температура, градиенты и несколько positives"
    label_en: "Temperature, gradients and multiple positives"
    anchor: emb-positive-sets
  - label: "Hard negatives и ложные отрицательные"
    label_en: "Hard negatives and false negatives"
    anchor: emb-negatives
  - label: "In-batch negatives и накопление градиентов"
    label_en: "In-batch negatives and gradient accumulation"
    anchor: emb-in-batch
  - label: "SimCSE и две dropout-маски"
    label_en: "SimCSE and two dropout views"
    anchor: emb-simcse
  - label: "Pair, contrastive и triplet loss"
    label_en: "Pair, contrastive and triplet losses"
    anchor: emb-metric-losses
  - label: "MarginMSE и дистилляция оценок"
    label_en: "MarginMSE and score distillation"
    anchor: emb-marginmse
  - label: "Анизотропия и whitening эмбеддингов"
    label_en: "Embedding anisotropy and whitening"
    anchor: emb-anisotropy
  - label: "Размерность и память векторного индекса"
    label_en: "Dimension and vector-index memory"
    anchor: emb-index-memory
  - label: "Matryoshka и вложенные размерности"
    label_en: "Matryoshka and nested dimensions"
    anchor: emb-matryoshka
  - label: "Совместимость энкодеров и пересчёт корпуса"
    label_en: "Encoder compatibility and corpus re-encoding"
    anchor: emb-compatibility
  - label: "Релевантность, qrels, Recall и Hit"
    label_en: "Relevance, qrels, Recall and Hit"
    anchor: emb-qrels
  - label: "MRR и nDCG для эмбеддингов"
    label_en: "MRR and nDCG for embeddings"
    anchor: emb-ranking-metrics
  - label: "Exact search: пять документов"
    label_en: "Exact search: five-document example"
    anchor: emb-exact-search
  - label: "Модель, ANN-индекс и неполная разметка"
    label_en: "Model, ANN-index and incomplete-label errors"
    anchor: emb-error-decomposition
  - label: "GradCache — общий loss и повторные проходы"
    label_en: "GradCache: joint loss and recomputation"
    anchor: emb-gradcache
---

# Эмбеддинги и метрическое обучение

Пользователь спрашивает: «Как изменить пароль, если я помню текущий?»
Поиск по базе поддержки должен показать инструкцию смены пароля.
Страница восстановления доступа тоже содержит слово «пароль», но
объясняет, как войти, когда текущий пароль забыт. Для другого запроса
порядок этих двух документов должен поменяться.

<a id="emb-password-case"></a>

Сохраним пять документов. Для этого вопроса разметчик поставил 3 полной
инструкции, 2 — инструкции только для одного интерфейса, 1 — странице,
которая указывает нужный раздел, но не объясняет шаги. Ноль означает,
что документ на вопрос не отвечает.

| Документ | Содержание | Балл |
|---|---|---:|
| $d_1$ | Полная инструкция смены пароля | 3 |
| $d_2$ | Восстановление доступа через резервную почту | 0 |
| $d_3$ | Смена пароля в мобильном приложении | 2 |
| $d_4$ | Настройка печати PDF | 0 |
| $d_5$ | Где находятся настройки безопасности, без инструкции | 1 |

Вектор, в который энкодер преобразует текст, называют эмбеддингом.
Векторы документов можно вычислить до первого запроса. Когда приходит
вопрос, система кодирует только его и сравнивает результат с уже
сохранёнными векторами. В обучении высокие оценки нужно давать парам,
которые связаны ответом на вопрос, а не просто темой учётной записи.
Вернёмся к этим пяти документам, когда будем разбирать
[[#emb-exact-search|порядок выдачи]].

<span id="1-от-токенных-представлений-к-представлению-последовательности"></span>

## Один вектор на текст

<a id="emb-pooling"></a>
<span id="что-усредняется"></span>

### Pooling по токенам

Transformer обычно возвращает матрицу состояний $H\in\mathbb R^{T\times d}$:
по одному вектору размерности $d$ для каждой из $T$ позиций.
Представление токена зависит от соседних токенов через attention.
Но для сравнения целых предложений ещё нужно выбрать способ получить
один вектор. Объединение состояний называют **pooling**.

Возьмём условные двумерные состояния трёх токенов:
$(1,0)$, $(0,2)$ и $(2,1)$. Их среднее — $(1,1)$.
В пакете предложение дополнено четвёртой позицией padding.
Пусть её выходное состояние равно $(100,0)$: нет основания считать,
что Transformer обязан вернуть в padding-позиции нули. Если включить
его в среднее, получится $(25.75,0.75)$ — почти другое направление.
Поэтому маска участвует не только в attention, но и в объединении выходов.

Пусть энкодер возвращает $h_1,\ldots,h_T$, а $m_t=1$ для включённой
позиции и 0 для исключённой. **Mean pooling** имеет вид

$$
e(x)=\frac{\sum_t m_t h_t}{\sum_t m_t}.
$$

Знаменатель — число включённых токенов, а не длина дополненной последовательности.
Для полностью пустой маски среднее не определено. Нулевая строка после
искусственного ограничения знаменателя тоже не становится пригодным
вектором для косинуса; такой вход нужно обработать отдельно.

Другие способы выбирают состояние `[CLS]`, последнее непустое состояние,
покомпонентный максимум или обучаемую комбинацию. В нашем примере
`[CLS]` в первой позиции дал бы $(1,0)$, последнее состояние — $(2,1)$,
а максимум — $(2,2)$. Эти правила не взаимозаменяемы после обучения.
Для causal decoder последнее состояние видит предшествующий текст,
тогда как ранние состояния не видят будущие позиции. Выбор pooling
связан с маской внимания и задачей обучения.

В оригинальном [Sentence-BERT, §3](https://arxiv.org/abs/1908.10084)
сравнивались CLS, mean и max; стандартной конфигурацией был mean.
В [DPR, §3.1](https://arxiv.org/abs/2004.04906) использовалось
состояние `[CLS]`. Это свойства конкретных обученных моделей,
а не доказательство универсального превосходства одного pooling.

<a id="emb-similarity"></a>
### Норма вектора и функция сравнения

Пусть запрос $q=(1,0)$, документы $a=(1,0)$ и $b=(2,2)$.
Скалярные произведения равны $q^\top a=1$ и $q^\top b=2$:
выше оказывается $b$. Косинус сравнивает направления и даёт
$\cos(q,a)=1$, $\cos(q,b)=1/\sqrt2\approx0.707$:
выше оказывается $a$. Норма $b$ повлияла на скалярное произведение.

Для поиска по косинусу ненулевой вектор нормируют:

$$
\begin{aligned}
\hat e(x)&=\frac{e(x)}{\|e(x)\|_2},\\
s(x,y)&=\hat e(x)^\top \hat e(y).
\end{aligned}
$$

После нормировки косинус, скалярное произведение и квадрат евклидова расстояния
задают один порядок, потому что
$\|\hat e_x-\hat e_y\|_2^2=2-2\hat e_x^\top\hat e_y$.
Чем больше произведение единичных векторов, тем меньше расстояние.
Без нормировки эквивалентности нет. Например, DPR обучался со скалярным
произведением ненормированных представлений; добавленная при эксплуатации
нормировка меняет его функцию ранжирования.

<details id="emb-code-mean-pooling">
<summary>Маска и нормировка векторов</summary>

```python
import numpy as np

def mean_pool(states, mask):
    states, mask = np.asarray(states, dtype=float), np.asarray(mask)
    if states.ndim != 2 or mask.shape != (states.shape[0],):
        raise ValueError("need a token matrix and a mask of equal length")
    if not np.isfinite(states).all() or not np.isin(mask, [0, 1]).all() or mask.sum() == 0:
        raise ValueError("need finite states and a nonempty binary mask")
    return (states * mask[:, None]).sum(axis=0) / mask.sum()

def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("need a finite vector matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cosine is undefined for a zero vector")
    return vectors / norms

h = [[1, 0], [0, 2], [2, 1], [100, 0]]
assert np.allclose(mean_pool(h, [1, 1, 1, 0]), [1, 1])
assert np.allclose(mean_pool(h, [1, 1, 1, 1]), [25.75, .75])
q, docs = np.array([[1., 0.]]), np.array([[1., 0.], [2., 2.]])
assert np.argmax((q @ docs.T)[0]) == 1
assert np.argmax((normalize_rows(q) @ normalize_rows(docs).T)[0]) == 0
```

</details>

Косинус лежит в $[-1,1]$, но не является вероятностью релевантности.
Порог 0.8 нельзя переносить между моделями только потому, что обе
используют косинус. Распределение оценок меняется вместе с обучением,
языком, корпусом и форматом запроса.

<a id="emb-input-contract"></a>
### Токенизация, инструкция и обрезка

Рабочая модель включает токенизатор, специальные токены, pooling,
ограничение длины и нормировку. Например, карточка
[multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base#usage)
показывает mean pooling по attention mask, L2-нормировку и префиксы
`query: ` и `passage: `, в том числе для неанглийского текста.
Фразы `query: Как изменить пароль?` и `passage: Как изменить пароль?`
имеют разные роли. Эти строки не следует переводить или удалять
без проверки инструкции конкретной модели.

Если документ длиннее входного окна, обрезка может удалить фрагмент
с ответом. Тогда энкодер не сможет сохранить сведения, которых не видел.
Разбиение корпуса на фрагменты и привязка результата к исходному
документу рассматриваются в следующей главе о retrieval.
Инструкцию иногда исключают из pooling, но оставляют во входе attention:
это тоже настройка обученной модели. В
[коде Pooling](https://github.com/huggingface/sentence-transformers/blob/8a880ec7f1524640619fe382940ee319144f74ca/sentence_transformers/sentence_transformer/modules/pooling.py)
этому соответствует `include_prompt`; его нельзя выбирать независимо
от принятой схемы обучения.

<a id="emb-encoders"></a>
<span id="2-почему-обычный-bert-не-является-готовой-моделью-предложений"></span>

## Bi-encoder и cross-encoder

Усреднить состояния предобученного BERT технически можно сразу.
Однако хорошее предсказание токенов ещё не означает, что косинус
этих средних правильно упорядочит целые предложения. Обучающая
задача должна связывать именно те представления и оценки,
которые затем используются в поиске. В
[SBERT, §4.1 и таблица 1](https://arxiv.org/abs/1908.10084)
средние состояния и `[CLS]` обычного BERT проверялись как отдельные
baseline; качество модели на других NLP-задачах не переносилось
автоматически на такую оценку сходства.

В **cross-encoder** два предложения проходят через BERT совместно, поэтому каждый
токен одного предложения может взаимодействовать с каждым токеном другого. Это
позволяет обучить оценку пары, но для поиска среди миллиона документов
нужно обработать миллион пар запрос–документ. Их можно объединять
в пакеты; представление документа всё равно нельзя полностью вычислить
без запроса. Bi-encoder кодирует стороны независимо; документы
можно обработать заранее, а во время запроса выполнять быстрый поиск ближайших
соседей. Разница показана в документации Sentence Transformers:

![[00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/bi-cross-encoder.png|SBERT: separate bi-encoder representations and joint cross-encoder scoring]]

*Bi-encoder и cross-encoder. Nils Reimers и Sentence Transformers contributors,
[Cross-Encoders](https://sbert.net/docs/cross_encoder/usage/usage.html),
[оригинальный PNG](https://github.com/huggingface/sentence-transformers/blob/8a880ec7f1524640619fe382940ee319144f74ca/docs/img/Bi_vs_Cross-Encoder.png).
Apache-2.0; без изменений. [Лицензия](/licenses/sbert-8a880ec7-Apache-2.0.txt).*

У левой схемы стрелки от двух текстов сходятся только в функции сравнения.
У правой оба текста входят в один Transformer до классификатора.
Pooling левой ветви не учитывает конкретный второй текст. Это позволяет
кэшировать векторы документов, но ограничивает взаимодействие признаков.

**Независимые вычисления** не означают разные веса. В Sentence-BERT
две ветви имеют общие параметры: один и тот же энкодер применяется к
обоим предложениям. В DPR запрос и документ кодируют две отдельные
BERT-модели. Оба варианта относятся к bi-encoder: решает место
взаимодействия текстов, а не факт разделения параметров.

Один из вариантов обучения исходного SBERT использует NLI-разметку:
следование (*entailment*), противоречие (*contradiction*) или
отсутствие обоих отношений (*neutral*). Например, из «Два человека
играют в шахматы» следует «Люди играют в игру», но не «Люди спят».
Два текста отдельно превращаются в векторы $u,v$; классификатор
получает конкатенацию $u$, $v$ и покомпонентного $|u-v|$ и
обучается cross-entropy по трём классам. Градиент проходит через
pooling в общий энкодер. После обучения для сравнения предложений
используются сами векторы и их косинус, без NLI-классификатора.
Трёхклассовая вероятность и косинус — разные выходы модели;
перенос на задачу сходства проверяется отдельно. Другие цели
SBERT — регрессия сходства и обучение на тройках — разобраны в разделе 5.

Если нужно сравнить все неупорядоченные пары из 10 000 предложений,
cross-encoder оценит $10000\cdot9999/2=49\,995\,000$ пар.
Bi-encoder выполнит 10 000 кодирований, но сравнения векторов
всё равно потребуются. Уменьшается число дорогих запусков Transformer,
а не исчезает задача поиска. Исторические измерения времени SBERT
относятся к оборудованию и моделям статьи 2019 года; это не прогноз
задержки современного сервиса.

Bi-encoder обязан сжать всё,
что понадобится для будущих сравнений, в один вектор до того, как увидит вторую
сторону. Поэтому его обучают непосредственно на парах или тройках, отражающих
целевое понятие близости. В документации Sentence Transformers
[функции потерь сопоставлены с форматами разметки](https://sbert.net/docs/sentence_transformer/loss_overview.html):
пары с вещественной оценкой, положительные пары без явных отрицательных,
тройки anchor–positive–negative и списки кандидатов задают разные наблюдения.
Нельзя заменить метку сходства вероятностью того, что ответ достаточен
для конкретного запроса, без изменения постановки.

Часто bi-encoder сначала извлекает несколько десятков кандидатов,
а cross-encoder переупорядочивает их. Если правильного документа
нет среди кандидатов, повторное ранжирование его не добавит.
Раздельно измеряют полноту первого этапа и качество порядка второго.

<span id="3-контрастивная-задача"></span>

## Контрастивное обучение

<a id="emb-contrastive"></a>
### Матрица оценок и целевой документ

Пусть пакет содержит три запроса и по одному правильному документу
для каждого. Каждый запрос сравнивается со всеми тремя документами.
Таким образом, шесть кодирований дают девять оценок. При $B$ парах матрица

$$
S_{ij}=\frac{s(e_q(q_i),e_d(d_j))}{\tau}
$$

содержит $B^2$ сравнений. $e_q,e_d$ — функции кодирования,
$s$ — косинус или скалярное произведение, $\tau>0$ — температура.
Диагональ состоит из размеченных положительных пар. Остальные документы
служат отрицательными **при предположении**, что они не отвечают другому
запросу в пакете. Потеря для направления
«запрос → документ»:

$$
\begin{aligned}
p_{ij}&=\frac{\exp S_{ij}}{\sum_k\exp S_{ik}},\\
\ell_i&=-\log p_{ii},\\
\mathcal L_{q\to d}&=\frac1B\sum_i\ell_i.
\end{aligned}
$$

Softmax нормирует одну строку: правильный документ должен выделяться
среди текущих кандидатов. Если пакет изменится, изменится и $p_{ii}$,
даже при неизменной оценке правильной пары. Поэтому $p_{ii}$ —
вероятность выбора в данной обучающей задаче, не готовая калиброванная
вероятность полезности документа во всём корпусе.

В DPR применён ненормированный dot product без отдельного параметра
температуры в формуле. Запись с $\tau$ обобщает эту схему;
она не означает, что DPR обучался с настройками современного MNRL.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/dpr-dual-encoder-and-loss.png|DPR dual encoders, dot-product scores and contrastive loss]]

*Danqi Chen, [ACL 2020 Open-Domain QA Tutorial](https://github.com/danqi/acl2020-openqa-tutorial/blob/3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0/slides/part5-dense-retriever-e2e-training.pdf#page=20),
слайд 20. Два энкодера, dot product и потеря выбора положительного passage.
На узком экране полный слайд прокручивается по горизонтали;
нажатие на изображение открывает оригинал.*

Фраза слайда «without any pre-training» относится к отсутствию
дополнительного дорогого предобучения retrieval-модуля, обсуждавшегося
в ORQA/REALM. Оба энкодера DPR инициализированы **предобученным BERT**.
Читать её как обучение всей системы с нуля по небольшой QA-выборке неверно.

<a id="emb-batch-example"></a>
<span id="один-пакет-в-числах"></span>

### Три кандидата в одном пакете

В пакет попали вопрос о смене пароля и два других запроса: о восстановлении
доступа и о печати PDF. Для первого запроса правильный документ $d_1$.
Документы $d_2$ и $d_4$, которые отвечают на два других вопроса, войдут
в ту же строку как отрицательные. Пусть после масштабирования оценки
$S_{1,:}=(2,1,0)$. Тогда

$$
\begin{aligned}
p_{11}&=\frac{e^2}{e^2+e+1}\approx0.665,\\
\ell_1&=-\log p_{11}\approx0.408.
\end{aligned}
$$

Если каждая строка имеет те же оценки, только 2 всегда стоит у правильного
документа, средняя потеря пакета тоже равна 0.408. Поднимем оценку страницы
восстановления до 1.9, оставив остальные прежними. Теперь доля правильной
инструкции в softmax равна 0.490, а потеря растёт до 0.713. Именно ошибка
различения смены и восстановления даёт более сильный сигнал, чем далёкая
страница о печати.

Если бы страница восстановления содержала и полноценную инструкцию смены,
её следовало бы пометить как ещё один положительный документ или исключить
из отрицательных. В первом варианте можно считать вероятность выбрать
любой подходящий ответ: числитель становится $e^2+e^{1.9}$, и потеря падает
примерно до 0.069. Обычный softmax с одной целевой позицией сам этого не сделает.

<a id="emb-positive-sets"></a>
<span id="почему-ближайшие-отрицательные-получают-больший-градиент"></span>

### Градиент по оценкам и векторам

Для одной строки производная по **масштабированной** оценке имеет вид
$\partial\ell_i/\partial S_{ij}=p_{ij}-\mathbf1[j=i]$.
У строки $(2,1,0)$ softmax примерно равен $(0.665,0.245,0.090)$.
Градиенты — $(-0.335,0.245,0.090)$. Если бы оценки были независимыми
параметрами, шаг спуска повысил бы первую и понизил две другие.
У страницы восстановления производная больше, чем у страницы о печати.

В модели оценки связаны общим вектором запроса. Пусть при $\tau=1$
мы получили строку $(2,1,0)$ скалярными произведениями запроса $(1,0)$
с документами $(2,0)$, $(1,1)$ и $(0,1)$. Если временно зафиксировать
документы и менять только запрос, его градиент — сумма документных
векторов с весами из предыдущего абзаца. Это примерно $(-0.425,0.335)$.
После шага 0.1 запрос становится $(1.0425,-0.0335)$, оценки — примерно
$(2.085,1.009,-0.0335)$. Вторая оценка даже выросла, но отрыв правильной
инструкции увеличился с 1 до 1.076. В реальном обучении этот градиент
передаётся дальше в веса энкодера; обновление весов может менять
представления сразу нескольких запросов и документов.

Температура меняет распределение этих усилий. Для исходных оценок
$(0.4,0.2,0)$ при $\tau=0.2$ получаются уже разобранные logits
$(2,1,0)$ и потеря 0.408. При $\tau=0.1$ это $(4,2,0)$
и потеря около 0.143: правильный документ уже первый, и распределение
сильнее сосредоточено на нём. По исходным оценкам градиент дополнительно
содержит множитель $1/\tau$. Следовательно, уменьшение температуры
не обязано увеличивать каждую потерю или каждый градиент.
Ошибочно размеченный близкий документ при этом остаётся целью отталкивания.

В
[MultipleNegativesRankingLoss](https://github.com/huggingface/sentence-transformers/blob/8a880ec7f1524640619fe382940ee319144f74ca/sentence_transformers/sentence_transformer/losses/multiple_negatives_ranking.py)
вместо $\tau$ задан `scale`: `scale = 1 / temperature`.
Для ненормированных dot-product моделей масштаб и нормы векторов
должны соответствовать обучению; копировать значение 20 по умолчанию
в любую модель оснований нет.

<details id="emb-code-contrastive-row">
<summary>Потеря и градиент одной строки</summary>

```python
import numpy as np

def contrastive_row(scores, positive_indices):
    scores = np.asarray(scores, dtype=float)
    positives = np.asarray(positive_indices)
    if scores.ndim != 1 or not np.isfinite(scores).all() or len(scores) == 0:
        raise ValueError("need a nonempty finite score vector")
    if positives.ndim != 1 or len(positives) == 0 or positives.dtype.kind not in "iu":
        raise ValueError("need a nonempty vector of integer indices")
    if len(set(positives)) != len(positives):
        raise ValueError("need unique positive indices")
    if np.any((positives < 0) | (positives >= len(scores))):
        raise ValueError("positive index out of range")
    log_z = np.logaddexp.reduce(scores)
    log_positive = np.logaddexp.reduce(scores[positives])
    probabilities = np.exp(scores - log_z)
    target = np.zeros_like(scores)
    target[positives] = np.exp(scores[positives] - log_positive)
    return float(log_z - log_positive), probabilities, probabilities - target

loss, probabilities, gradient = contrastive_row([2, 1, 0], [0])
assert np.isclose(loss, 0.4076059644443806)
assert np.isclose(gradient.sum(), 0)
multi_loss, _, _ = contrastive_row([2, 1.9, 0], [0, 1])
assert 0.068 < multi_loss < 0.069
```

</details>

При нескольких положительных здесь минимизируется отрицательный логарифм
**суммарной** вероятности множества. Можно выбрать другую цель — среднее
отдельных $-\log p_{ij}$ по всем положительным. Первая допускает сосредоточить
массу на одном из них; вторая требует поддержать каждый. Эти цели
не равны и выражают разные требования к полноте ответа.

Для двунаправленного обучения отдельно нормируют строки и столбцы,
затем усредняют две потери. Это распространено, например, в CLIP.
Для поиска вопрос → документ выбор обратного направления требует
обоснования: один документ может отвечать на несколько вопросов.

<span id="4-пары-определяют-геометрию"></span>

## Подбор обучающих пар

<a id="emb-negatives"></a>
### Откуда берутся положительные и отрицательные примеры

К инструкции о смене пароля случайная страница о печати служит лёгким
отрицательным примером. По ней модель может научиться отличать учётные
записи от печати, но всё ещё путать смену пароля с восстановлением.
Страница восстановления годится как **трудный отрицательный пример**
(*hard negative*), если поиск поднимает её высоко и проверка подтверждает,
что она не отвечает на запрос. Для вопроса о забытом пароле та же страница
становится положительным примером.

В DPR использовались следующие источники:

- положительный passage из набора чтения или найденный по строке ответа;
- случайные фрагменты корпуса для сравнения способов выбора отрицательных;
- положительные passages других вопросов в том же пакете как in-batch negatives;
- верхний результат BM25 без строки ответа как hard negative.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/dpr-positive-and-negative-examples.png|DPR question with positive and BM25 negative passages]]

*Danqi Chen,
[ACL 2020 Open-Domain QA Tutorial](https://github.com/danqi/acl2020-openqa-tutorial/blob/3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0/slides/part5-dense-retriever-e2e-training.pdf#page=21),
слайд 21. Источники положительных и отрицательных фрагментов в DPR.
Полный слайд без обрезки.*

Положительные примеры в работе тоже различались. Для NQ и SQuAD
сопоставляли исходный размеченный контекст с фрагментами корпуса.
Когда имелись только вопрос и строка ответа, брали найденный BM25
фрагмент с этой строкой. Например, наличие «Париж» в абзаце не
доказывает, что абзац отвечает на любой вопрос с ответом «Париж».
Такое автоматическое получение меток называют слабой или
дистанционной разметкой; её ограничения переходят в обучение.

Ложный отрицательный пример может содержать другой правильный ответ,
хотя разметка считает положительным только
один passage. При поиске трудных примеров такие случаи нужно проверять:
высокая оценка бывает и признаком ошибки модели, и признаком неполной разметки.
Поэтому отрицательные примеры проверяют по нескольким аннотациям, оценке модели-учителя
или правилам исключения и обновляют после изменения модели. Иначе обучение
заставляет раздвигать семантически правильные пары. В tutorial
[MS MARCO](https://sbert.net/examples/sentence_transformer/training/ms_marco/README.html)
для фильтрации применён cross-encoder: при оценке положительного
документа 9 и требуемом разрыве 3 берут отрицательные с оценкой ниже 6.
Оценка 6.5 не пройдёт этот фильтр. Разрыв — эвристика конкретной
процедуры, а не доказательство истинной нерелевантности; ошибки учителя
тоже возможны.

<a id="emb-in-batch"></a>
<span id="что-даёт-пакет"></span>

### Отрицательные документы из того же пакета

In-batch negatives экономят вычисления: $B$ кодирований каждой стороны дают
$B^2$ оценок. Но примеры внутри пакета должны быть достаточно разнообразны. Если
несколько запросов имеют общий правильный документ или батч построен из одной
узкой темы, предположение «всё вне диагонали отрицательно» может нарушаться.

![[00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/multiple-negatives.png|Multiple-negatives ranking loss: positives and shared batch negatives]]

*Один запрос $a_1$, его положительный $b_1$ и кандидаты $b_2,\ldots,b_5$.
Nils Reimers и Sentence Transformers contributors,
[MS MARCO: MultipleNegativesRankingLoss](https://sbert.net/examples/sentence_transformer/training/ms_marco/README.html),
[исходный PNG](https://github.com/huggingface/sentence-transformers/blob/8a880ec7f1524640619fe382940ee319144f74ca/docs/img/MultipleNegativeRankingLoss.png).
Apache-2.0; без изменений. Связь $a_1$–$b_1$ положительная,
остальные показанные пары отрицательные.*

Если каждый из $B$ элементов содержит ещё по одному явному отрицательному
документу, документных векторов будет $2B$, а матрица — $B\times2B$.
На запрос приходится один положительный и $2B-1$ отрицательных.
В оригинальном объяснении Sentence Transformers это сформулировано так:

> For a batch size of 64, we compare a query against 64+64=128 passages,
> from which only one passage should be close and the 127 others should be
> distant in vector space.

Sentence Transformers contributors, *MS MARCO*, Apache-2.0.
Условие «только один» относится к принятой разметке. Оно не гарантирует,
что другие 127 фрагментов действительно бесполезны. И $B^2$ оценок
не равны $B^2$ независимо размеченным человеческим примерам.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/dpr-in-batch-negatives.png|DPR historical ablation of in-batch and BM25 negatives]]

*Danqi Chen, тот же
[tutorial](https://github.com/danqi/acl2020-openqa-tutorial/blob/3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0/slides/part5-dense-retriever-e2e-training.pdf#page=22),
слайд 22. Сравнение схем обучения на development split Natural Questions;
это результаты DPR 2020 года, не измерения нашего примера.*

При семи gold negatives без in-batch объединения Top-20 равен 63.1%,
с объединением — 69.1%. Для 31 gold negative результат 70.8%;
с добавлением 32 BM25 negatives, по одному на каждый запрос
пакета, — 77.3%. Добавление уже двух на запрос дало 76.4%.
Поэтому таблица не обосновывает правило «чем больше отрицательных,
тем всегда лучше». На слайде Top-k обозначает наличие хотя бы
одного passage со строкой ответа, а не долю всех релевантных документов.
Эту разницу можно увидеть и на
[[#emb-qrels|примере с тремя релевантными документами]].

Накопление градиентов не заменяет совместный набор кандидатов.
Если в микропакете одна положительная пара без отрицательных,
softmax имеет один элемент, вероятность равна 1, потеря — 0.
Сложение градиентов от двух таких микропакетов оставит ноль.
Если объединить две пары до вычисления потери и оба документа
имеют одинаковую оценку для каждого запроса, потеря равна
$-\log(1/2)=\log2$ на запрос: появились отрицательные пары.
Поэтому «эффективный batch size» оптимизатора и размер множества
контрастивных кандидатов описывают разные величины.

При нескольких ускорителях можно обменяться эмбеддингами и собрать
общую матрицу. Нужно согласовать индексы положительных пар и
передачу градиента через обмен.

<a id="emb-gradcache"></a>
### GradCache: full-batch negatives with chunked encoder passes

Split four query–document pairs into two chunks, pairs 1–2 and 3–4.
In the figure, rows are queries and columns are documents. A query's
positive is on the diagonal; the other documents in its row are
negatives if the labels say they are irrelevant.

![[00 Учебник/Assets/Figures/curated/gradcache-comparison-2026-11-09/contaccum-figure1.png|Three panels: GradCache uses all 16 query–document scores; gradient accumulation uses only the two 2-by-2 diagonal blocks; ContAccum mixes current and detached past representations]]

*Jaehee Kim, Yukyung Lee, Pilsung Kang,
[A Gradient Accumulation Method for Dense Retriever under Memory Constraint, Figure 1](https://arxiv.org/html/2406.12356v3#S2.F1).
[Original PNG](https://arxiv.org/html/2406.12356v3/Styles/figures/comparison-camera_ready.png), unchanged,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).*

In **(a)** GradCache computes the row loss over all four columns.
Query 1 has three negatives, including documents 3 and 4 in the
other chunk. In **(b)** ordinary gradient accumulation computes
each loss separately, over just two columns. Query 1 has only
document 2 as a negative. The empty blocks in (b) are absent from
the softmax; they are not zero-valued logits. For the score row
$(1,0,0,0)$, the loss for query 1 is $\log(1+3/e)\approx0.7437$ in
(a) and $\log(1+1/e)\approx0.3133$ in (b). Accumulating the weight
gradients cannot put the two missing terms back into the denominator.

In **(c)** [[05 Источники/Papers/ContAccum|ContAccum]] appends four saved query
vectors and four saved document vectors to the two current pairs. A current
query now compares its positive with five negatives. A saved document
affects the gradient of the current query but receives no gradient
through its old encoder pass; the hatched cells show comparisons with
one trainable side. This changes the loss. GradCache instead
recomputes encoder activations for the same examples and preserves
the full-batch objective.

With two training pairs, let both the query and document inputs be the rows
of the identity matrix. A shared linear encoder starts with an identity weight
matrix, so the score matrix is also the identity. Each query has one positive
with score 1 and one negative with score 0. At temperature 1, the positive
probability is about 0.7311 and the mean loss is about 0.3133.
Splitting the two pairs into singleton losses removes both negatives:
each softmax then has one entry and both losses are zero.

GradCache keeps the two-by-two loss. It changes when the encoder's computation
graphs are built and discarded. First, encode both queries and both documents
without recording encoder activations. These four vectors are enough to
calculate the contrastive loss and its derivatives with respect to the vectors.
Then recompute each encoder chunk with autograd enabled and propagate its
already calculated representation gradient into the encoder weights.

The published paper describes the sequence as follows. The excerpt retains
the original English; the numeric example and executable check below are
Bookvar additions.

> **Step1: Graph-less Forward** Before gradient computation, we first run an extra
> encoder forward pass for each batch instance to get its representation.
> Importantly, this forward pass runs without constructing the computation
> graph. We collect and store all representations computed.
>
> **Step2: Representation Gradient Computation and Caching** We then compute the
> contrastive loss for the batch based on the representation from Step1 and
> have a corresponding computation graph constructed. Despite the mathematical
> derivation, automatic differentiation system is used in actual implementation,
> which automatically supports variations of contrastive loss. A backward pass
> is then run to populate gradients for each representation. Note that the
> encoder is not included in this gradient computation.
>
> **Step3: Sub-batch Gradient Accumulation** We run encoder forward one sub-batch
> at a time to compute representations and build the corresponding computation
> graph. We take the sub-batch’s representation gradients from the cache and
> run back propagation through the encoder. Gradients are accumulated for
> encoder parameters across all sub-batches.
>
> **Step4: Optimization** When all sub-batches are processed, we can step the
> optimizer to update model parameters as if the full batch is processed in a
> single forward-backward pass.

*Luyu Gao, Yunyi Zhang, Jiawei Han and Jamie Callan,
[Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup, §3.3](https://aclanthology.org/2021.repl4nlp-1.31/).
CC BY 4.0 under the [ACL Anthology policy](https://aclanthology.org/faq/copyright/).
Selected prose paragraphs reproduced verbatim; the intervening equations and
the sentence defining the cache symbols are omitted.*

For the two-pair example, the derivative with respect to the first query
vector is approximately $(-0.13447,0.13447)$. The first document gets the
same derivative; the second query and document get
$(0.13447,-0.13447)$. Each derivative includes the factor $1/2$ from the
mean over the two queries. A cached derivative therefore already includes
the full-batch reduction. Dividing it by the number of chunks again would
make the parameter gradient too small.

Let $h_i=f_\theta(x_i)$ be a representation and
$g_i=\partial L/\partial h_i$ its cached gradient. The chain rule gives

$$
\begin{aligned}
\nabla_\theta L&=\sum_i J_i^\top g_i,\\
J_i&=\frac{\partial f_\theta(x_i)}{\partial\theta}.
\end{aligned}
$$

Recomputing one chunk supplies its vector–Jacobian product without keeping
all other encoder graphs alive. In code, the scalar surrogate
$\sum_i h_i^\top\operatorname{stopgrad}(g_i)$ has the required parameter
gradient. Its numeric value is not the original loss, so it is not a replacement
for the loss in training logs. With shared encoder weights, query-side and
document-side contributions accumulate into the same parameter tensors.

The CPU check below uses a linear encoder, not a pretrained language model.
It compares every weight derivative with an ordinary full-batch backward pass.
No optimizer update happens between chunks.

```python
import torch
from torch import nn
from torch.nn import functional as F


def cached_encoder_grad(model, queries, documents, chunk_size):
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        q = torch.cat([model(x) for x in queries.split(chunk_size)])
        d = torch.cat([model(x) for x in documents.split(chunk_size)])
    q = q.detach().requires_grad_()
    d = d.detach().requires_grad_()
    labels = torch.arange(len(queries), device=queries.device)
    loss = F.cross_entropy(q @ d.T, labels)
    dq, dd = torch.autograd.grad(loss, (q, d))

    for inputs, gradients in [(queries, dq), (documents, dd)]:
        for x, g in zip(inputs.split(chunk_size), gradients.split(chunk_size)):
            # g already contains the mean reduction over all queries.
            (model(x) * g).sum().backward()
    return loss.detach(), [p.grad.detach().clone() for p in model.parameters()]


inputs = torch.eye(2, dtype=torch.float64)
encoder = nn.Linear(2, 2, bias=False, dtype=torch.float64)
with torch.no_grad():
    encoder.weight.copy_(torch.eye(2, dtype=torch.float64))

full_loss = F.cross_entropy(encoder(inputs) @ encoder(inputs).T, torch.arange(2))
full_loss.backward()
full_gradient = encoder.weight.grad.detach().clone()
cached_loss, cached_gradients = cached_encoder_grad(encoder, inputs, inputs, 1)

assert torch.allclose(cached_loss, full_loss.detach())
assert torch.allclose(cached_gradients[0], full_gradient, atol=1e-12)
print(round(float(cached_loss), 6))  # 0.313262
print(full_gradient)
# tensor([[-0.2689,  0.2689],
#         [ 0.2689, -0.2689]], dtype=torch.float64)

encoder.zero_grad(set_to_none=True)
for x in inputs.split(1):
    singleton_loss = F.cross_entropy(encoder(x) @ encoder(x).T, torch.zeros(1, dtype=torch.long))
    singleton_loss.backward()
assert torch.count_nonzero(encoder.weight.grad) == 0
```

This implementation assumes deterministic, example-independent encoder
computations. A training-mode Transformer usually has dropout. Its recomputed
chunk must use the same random choices as its first pass, including the separate
query and document passes when weights are shared. Otherwise the cached
$g_i$ belongs to one sampled representation but multiplies the Jacobian of
another. The [authors’ implementation](https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/grad_cache.py)
records a [RandContext](https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/context_managers.py)
for each chunk and restores CPU and relevant CUDA generator states for its
recomputation. The short example does not implement that mechanism and must
not be used unchanged with stochastic encoders.

Restoring RNG state does not restore arbitrary mutable model state.
Training-mode BatchNorm, for example, computes statistics from the chunk;
changing the chunk changes the function, and repeated forwards can update
running statistics twice. Equality to a full-batch gradient requires matching
the actual forward function, its parameters, stochastic choices and state.
Floating-point accumulation can also change the final low-order bits.

For a fixed encoder chunk size, retained encoder activations no longer grow
with the full contrastive batch. The representation and gradient caches still
grow as $O(Bd)$; the explicit score matrix in this example grows as $O(B^2)$.
Its loss computation can itself become the memory bottleneck. An extra encoder
forward pass also costs time. The paper's runtime measurements describe its
DPR experiment, not a fixed overhead for every model and GPU.


<a id="emb-simcse"></a>
### SimCSE: положительная пара без двух разных текстов

Для **unsupervised SimCSE** одно предложение проходит через энкодер
два раза с разными масками dropout. Получаются два стохастических
представления одного текста; они образуют положительную пару,
а другие предложения пакета служат отрицательными. Например,
«Two dogs are running.» в обоих проходах имеет тот же текст,
но немного разные промежуточные вычисления. Положительная пара
требует устойчивости к этой случайности, а отрицательные пары
не позволяют решить задачу одним одинаковым вектором для всех текстов.

![[00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/simcse.png|Unsupervised SimCSE dropout views and supervised NLI contrastive pairs]]

*Tianyu Gao, Xingcheng Yao, Danqi Chen,
[SimCSE](https://github.com/princeton-nlp/SimCSE/tree/49fa580a853752ede55b8c76d9debf748e214d3f),
[оригинальный PNG](https://github.com/princeton-nlp/SimCSE/blob/49fa580a853752ede55b8c76d9debf748e214d3f/figure/model.png).
MIT; без изменений. [Лицензия](/licenses/simcse-49fa580a-MIT.txt).
Сплошные и пунктирные стрелки различают положительные и отрицательные пары.
На узком экране две панели прокручиваются по горизонтали.*

В правой части **supervised SimCSE** использует NLI: entailment-пару
как положительную, contradiction-пару как трудную отрицательную.
Это отличается от исходного SBERT с трёхклассовой NLI-классификацией.
Такие метки отражают выбранное отношение между предложениями.
Например, при поиске опровержений утверждения противоречащий текст
как раз может быть релевантным документом. Цель retrieval нельзя
автоматически заменить NLI-контрастивной целью.

<span id="5-triplet-multiple-negatives-и-distillation"></span>

## Потери для пар, троек и оценок учителя

<a id="emb-metric-losses"></a>
### Расстояние, порядок и численная оценка пары

Формат разметки определяет, что можно требовать от модели.
Для пары с численной меткой сходства можно минимизировать квадрат
ошибки $\big(s(a,b)-y\big)^2$. При $y=0.8$ и прогнозе 0.6
получается 0.04, при прогнозе 0.7 — 0.01. Метка здесь должна
иметь смысл на выбранной шкале: идентификатор NLI-класса нельзя
подставить вместо численного сходства.

Для положительных/отрицательных пар **contrastive loss** с расстоянием
$d$ может иметь вид

$$
L=y d^2+(1-y)\max(0,m-d)^2.
$$

Здесь $y=1$ обозначает положительную пару. При $m=1$
положительная пара на расстоянии 0.3 даёт 0.09.
Отрицательная на расстоянии 0.6 даёт $(1-0.6)^2=0.16$,
на расстоянии 1.2 — ноль. Это определённая здесь форма цели;
коэффициент $1/2$ и соглашение о метках в реализации нужно проверять.

**Triplet loss** сравнивает положительный и отрицательный примеры
относительно одного anchor. Для функции сходства положительный
должен превосходить отрицательный на заданный запас $m$:

$$
\mathcal L=\max(0,m+s(a,n)-s(a,p)).
$$

При $m=0.2$, $s(a,p)=0.6$, $s(a,n)=0.5$ потеря равна 0.1.
После уменьшения отрицательного сходства до 0.3 она становится нулевой.
Тройка перестаёт давать обучающий сигнал, хотя другие отрицательные
кандидаты ещё могут ранжироваться неверно. Поэтому способ отбора
троек влияет на обучение. Batch-hard методы выбирают трудные
положительные и отрицательные внутри пакета; ошибочная отрицательная
метка в таком отборе может получить особенно большой вес.

Для расстояний записывают $\max(0,m+d(a,p)-d(a,n))$.
В [исходном SBERT, §3](https://arxiv.org/html/1908.10084v1#S3) для
троек использовали евклидово расстояние с запасом 1. На единичных
векторах поиск по наибольшему косинусу и наименьшему L2 вернёт
один порядок. Но зазор в 0.2 между двумя косинусами нельзя подставить
как зазор 0.2 в евклидово расстояние.

MNRL, разобранная выше, сравнивает с набором альтернатив через
softmax. Это другая цель, а не тот же hinge с большим числом
элементов. Сравнивать методы нужно при одинаковом корпусе, метках,
способе отбора кандидатов и бюджете, а не только по названиям loss.

<a id="emb-marginmse"></a>
### Дистилляция оценок учителя

Если сильная модель уже умеет оценивать пары, её ответы можно
использовать как разметку для ученика. Например, ансамбль дорогих
cross-encoders заранее оценивает пары запрос–документ, а компактный
оценщик учится воспроизводить эти значения:

![[00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/distillation.png|Sentence Transformers: cross-encoder teacher and cross-encoder student]]

*Дистилляция логитов ансамбля в компактный **cross-encoder** MiniLM.
Sentence Transformers contributors,
[Cross Encoder Knowledge Distillation](https://sbert.net/examples/cross_encoder/training/distillation/README.html),
[исходный PNG](https://github.com/huggingface/sentence-transformers/blob/8a880ec7f1524640619fe382940ee319144f74ca/docs/img/msmarco-training-ce-distillation.png).
Apache-2.0; без изменений. На этой схеме ученик оценивает пару совместно.*

При обучении **bi-encoder** по учителю можно сравнивать разности
оценок двух документов. **MarginMSE** задаёт цель

$$
\begin{aligned}
\Delta_T&=T(q,d_1)-T(q,d_2),\\
\Delta_S&=s(e_q(q),e_d(d_1))-s(e_q(q),e_d(d_2)),\\
L&=(\Delta_S-\Delta_T)^2.
\end{aligned}
$$

Учитель дал 5 и 3, ученик — 0.8 и 0.2. Разность учителя 2,
ученика 0.6, ошибка $(0.6-2)^2=1.96$. Если прибавить 100 к
обеим оценкам учителя, цель не изменится. Ученик воспроизводит
разность, а не абсолютное значение или вероятность.
В отличие от обычной одно-положительной MNRL, оба документа могут
быть релевантными. Например, учитель может поставить им 5 и 4.9:
целевой разрыв 0.1 не требует превращать второй ответ в нерелевантный.

Шкалы нужно согласовать. Разность двух косинусов ограничена
отрезком $[-2,2]$, поэтому не сможет точно воспроизвести разность
учителя 5. В
[Sentence Transformers MarginMSELoss](https://github.com/huggingface/sentence-transformers/blob/8a880ec7f1524640619fe382940ee319144f74ca/sentence_transformers/sentence_transformer/losses/margin_mse.py)
по умолчанию используется dot product. Выбор нормировки,
масштабирования учителя и функции потерь — часть постановки,
а не детали, которые можно поменять после обучения.

Учитель тоже может ошибаться. Если он не распознаёт отрицание
или опирается на неверную разметку, ученик получает такой же
обучающий сигнал. Независимый тест с человеческими суждениями
о релевантности нужен и после дистилляции.

Semantic textual similarity спрашивает, насколько
похожи два предложения в целом; retrieval спрашивает, удовлетворяет ли документ
информационную потребность запроса. Обучение только на NLI или STS даёт удобную
исходную геометрию, но не заменяет проверку на запросах и корпусе целевого
домена.

<span id="6-анизотропия-pooling-и-размерность"></span>

## Геометрия и размерность векторов

<a id="emb-anisotropy"></a>
### Общее направление и косинус

Если у многих векторов есть большая общая компонента, косинусы
могут мало различаться. Возьмём $x=(100,1)$ и $y=(100,-1)$.
Их косинус равен $9999/10001\approx0.9998$, хотя вторые координаты
противоположны. После вычитания общего вектора $(100,0)$ остаются
$(0,1)$ и $(0,-1)$ с косинусом $-1$. Пример показывает, как общий
компонент влияет на углы; он не устанавливает, какие тексты должны
считаться близкими.

**Анизотропией** называют неравномерность распределения по направлениям.
Узкий конус — один из возможных её видов. Centering вычитает средний
вектор; whitening дополнительно преобразует координаты с учётом
ковариации, уменьшая различия дисперсий. Эти операции могут убрать
общую компоненту, но могут удалить и полезный сигнал. Среднее и
ковариацию оценивают на обучающей части, а качество проверяют на
отложенных запросах. При недостатке данных ковариационная матрица
может быть вырожденной: её обращение требует ограничения ранга
или регуляризации.

Для простого примера whitening возьмём четыре точки
$(2,1),(2,-1),(-2,1),(-2,-1)$. Среднее равно нулю;
при делении на число точек дисперсии координат равны 4 и 1,
ковариация — нулю. Деление первой координаты на 2 даёт
$(1,1),(1,-1),(-1,1),(-1,-1)$: обе дисперсии теперь равны 1.
В общем случае сначала находят собственные направления ковариации,
затем масштабируют их обратно пропорционально корням собственных
значений. После этого нормы и углы могут измениться; L2-нормировка,
если она нужна для поиска по косинусу, выполняется отдельно.

В [SimCSE](https://github.com/princeton-nlp/SimCSE) распределение
представлений меняют самим контрастивным обучением. Это не повод
автоматически центрировать любую готовую модель. Сначала смотрят
на распределение оценок и ошибки ближайших соседей. Двумерная
UMAP-проекция не заменяет эту проверку: соседи и расстояния на
картинке могут отличаться от соседей в исходном пространстве.

<a id="emb-index-memory"></a>
### Сколько стоит размерность

Для $N$ векторов размерности $d$ с $b$ байтами на координату
непосредственное хранение требует $Ndb$ байт.
Миллион 768-мерных float32-векторов занимает
$10^6\cdot768\cdot4=3\,072\,000\,000$ байт:
3.072 GB, или примерно 2.861 GiB. Float16 сокращает этот объём
до 1.536 GB. При 128 координатах float32 нужно 0.512 GB — в шесть
раз меньше исходного варианта. Здесь не учтены структура индекса,
идентификаторы, метаданные и служебные буферы.

Уменьшение размерности меняет оценки, если модель не обучалась
сохранять сведения в коротком представлении. Для векторов
$(1,0,10)$ и $(0,1,10)$ полный косинус равен
$100/101\approx0.9901$, после сохранения только первых двух
координат — нулю. Нормировать нужно уже обрезанный вектор:
префикс единичного полного вектора в общем случае не имеет единичной нормы.

<a id="emb-matryoshka"></a>
### Matryoshka: обучение нескольких размеров

**Matryoshka Representation Learning** добавляет обучающий сигнал
не только ко всему вектору, но и к вложенным префиксам координат.
Например, энкодер выдаёт 768 чисел, а потери считают для первых
768, 512, 256, 128 и 64 координат. Для косинусной цели каждый
префикс отдельно нормируют:

$$
\begin{aligned}
z_m(x)&=\frac{e(x)_{1:m}}{\|e(x)_{1:m}\|_2},\\
\mathcal L_{\mathrm{MRL}}&=\sum_{m\in M}\lambda_m\mathcal L_m(z_m),
\qquad \lambda_m\ge0.
\end{aligned}
$$

Внутри $\mathcal L_m$ используется выбранная задача — например,
MNRL по запросам и документам. Один проход энкодера даёт полный
вектор, а все префиксы получают градиенты от своих потерь.
Координаты в начале должны поддерживать поиск даже без оставшихся.
Размеры $M$ и веса $\lambda_m$ задаются при обучении; произвольный
непроверенный размер не получает гарантии качества.

![[00 Учебник/Assets/Figures/curated/embeddings-review-2026-10/matryoshka-results.png|Matryoshka STSB Spearman correlation at nested embedding dimensions]]

*Tom Aarsen, Hugging Face / Sentence Transformers,
[Matryoshka Embeddings — Results](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html#results),
[оригинальный PNG](https://huggingface.co/datasets/huggingface/documentation-images/blob/1bade87e4c4e41d80d5f72ba2d7b38ab3194f044/blog/matryoshka/results.png).
Изображение из `huggingface/documentation-images`,
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
без изменений. Верхняя панель: корреляция Спирмена на STSBenchmark.
Нижняя: отношение к максимальному результату соответствующей модели.*

На графике сопоставлены два MPNet-base, обученных на AllNLI:
обычный MNRL и MNRL с Matryoshka. Это эксперимент из документации,
а не измерение нашей системы. При 64 из 768 координат — 8.3%
размерности — Matryoshka сохранила 98.37% максимальной корреляции
в этом эксперименте. Это **не 98.37% правильных ответов** и не
обещание такого же сохранения nDCG на другом корпусе.

В исходном объяснении отдельно оговорено:

> Despite the embeddings being smaller, training and inference of a Matryoshka
> model is not faster, not more memory-efficient, and not smaller. Only the
> processing and storage of the resulting embeddings will be faster and cheaper.

Sentence Transformers contributors, *Matryoshka Embeddings*, Apache-2.0.
Речь об обрезке выходных координат при неизменном энкодере:
Transformer по-прежнему вычисляет свои слои. Ускорение обработки
выходных векторов не равно ускорению их получения. Уменьшение числа
слоёв, которое документация обсуждает отдельно, — другое изменение модели.

<a id="emb-compatibility"></a>
### Когда пересчитывать корпус

Если поменять энкодер, pooling или текст на входе **документной** ветви,
прежний индекс больше не содержит нужных представлений. Обычно корпус
кодируют заново и строят новый индекс. Для перехода Matryoshka с 768 на 128
координат можно использовать сохранённые полные векторы: взять префиксы,
нормировать их для косинуса, затем перестроить индекс и подать в него
запросы той же размерности. Совпадение размерности само по себе
не позволяет смешивать векторы от разных энкодеров.

Даже преобразование, сохраняющее расстояния, требует согласованности.
Если оба вектора $(1,0)$ повернуть на $90^\circ$, их косинус останется 1.
Если повернуть только запрос, его косинус со старым документом станет 0.
Поэтому изменение только query-side обработки не требует пересчёта
неизменных документных векторов, но требует проверки качества:
новые запросные векторы могут не согласовываться со старым пространством.

<span id="7-как-проверять-эмбеддинг"></span>

## Качество поиска по эмбеддингам

<a id="emb-qrels"></a>
### Что считать правильным результатом

Оценочный набор для поиска состоит из запросов, корпуса и суждений
о релевантности документов — *qrels*. Одних пар похожих предложений
недостаточно: вопрос может требовать конкретного условия, даты или
доказательства, а не просто текста на ту же тему.

Для запроса $q$ обозначим множество релевантных документов $R_q$, а первые
$k$ результатов — $C_k(q)$. Тогда

$$
\begin{gathered}
\operatorname{Recall@k}(q)=\frac{|R_q\cap C_k(q)|}{|R_q|},\\
\operatorname{Hit@k}(q)=\mathbf1[|R_q\cap C_k(q)|>0].
\end{gathered}
$$

При трёх релевантных документах и одном найденном Recall@k равен $1/3$,
а Hit@k — 1. Precision@k делит число найденных релевантных на число
выданных результатов; при трёх выданных документах из примера это $1/3$.
Для Recall@k требуется оценка полного множества релевантных документов;
при неполной разметке это полнота относительно известных меток.
Значения усредняют по запросам, указывая правила усреднения.
Если $|R_q|=0$, полнота не определена. Нужно выяснить, есть ли
в корпусе ответ, которого нет в разметке. Если отсутствие ответа
установлено, такой запрос подходит для проверки отказа от выдачи,
а не для расчёта обычного Recall@k. Правило исключения либо условного
значения фиксируют вместе с метрикой.
В некоторых QA-работах hit называют recall; такую конвенцию указывают явно.
Различие соответствует
[определению полноты в Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html).

<a id="emb-ranking-metrics"></a>
### Порядок внутри первых результатов

Recall и hit не меняются при перестановке уже выбранных документов.
**Reciprocal rank** учитывает позицию первого релевантного: если
первый стоит на месте $r$, значение равно $1/r$; если релевантного
нет в оценённой выдаче — 0. MRR усредняет эти значения по запросам.
MRR@k ограничивает просмотр первыми $k$ позициями. Мера не учитывает,
сколько других релевантных документов расположено дальше.

Для градуированной релевантности **nDCG** сопоставляет качество
порядка с идеальной выдачей. Используем экспоненциальный gain:

$$
\begin{aligned}
\mathrm{DCG@k}&=\sum_{i=1}^k\frac{2^{r_i}-1}{\log_2(i+1)},\\
\mathrm{nDCG@k}&=\frac{\mathrm{DCG@k}}{\mathrm{IDCG@k}}.
\end{aligned}
$$

Здесь $r_i\ge0$ — степень релевантности на позиции $i$;
IDCG — DCG первых $k$ документов при идеальной сортировке всех
размеченных документов по релевантности. Gain $2^r-1$ сильнее
различает высокие степени; встречается и линейный gain $r$,
который задаёт другую метрику. Соглашение должно быть записано.
Если нет ни одного положительного gain, IDCG равен нулю;
такие запросы требуют отдельного правила учёта. Эта форма nDCG
приведена в
[Introduction to Information Retrieval, §8.4](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).

<a id="emb-exact-search"></a>
<span id="полный-пример-пять-документов"></span>

### Выдача по запросу о смене пароля

Вернёмся к [[#emb-password-case|пяти документам из базы поддержки]].
Теперь посмотрим на готовый индекс. Возьмём учебные единичные
векторы из кода ниже и вектор запроса $(1,0)$. Это отдельный набор
координат, не результат шага обучения из примера с градиентом.
Разметка прежняя: $d_1:3$, $d_3:2$, $d_5:1$; у остальных ноль.

Скалярные произведения равны $(1,0.8,0,-1,0.6)$, поэтому порядок
выдачи: $d_1,d_2,d_5,d_3,d_4$. В первых трёх найдено два из трёх
релевантных: Recall@3 $=2/3$, Hit@3 $=1$, reciprocal rank $=1$.
Страница восстановления заняла место мобильной инструкции, хотя поиск
точно отсортировал векторы по оценке.
DCG@3 равен $7+0+1/\log_2 4=7.5$.
Идеальные степени $(3,2,1)$ дают
$7+3/\log_2 3+1/2\approx9.393$, поэтому nDCG@3 $\approx0.799$.

Другая выдача $d_2,d_3,d_1$ имеет ту же полноту $2/3$, но первый
релевантный появился на втором месте: reciprocal rank $=1/2$.
DCG@3 теперь равен $0+3/\log_2 3+7/2\approx5.393$,
nDCG@3 $\approx0.574$. Одного Recall@3 недостаточно, чтобы увидеть разницу.

<details id="emb-code-ranked-metrics">
<summary>Расчёт метрик выдачи</summary>

```python
import numpy as np

ids = np.array(["d1", "d2", "d3", "d4", "d5"])
docs = np.array([[1, 0], [.8, .6], [0, 1], [-1, 0], [.6, .8]])
query = np.array([1., 0.])
qrels = {"d1": 3, "d3": 2, "d5": 1}

def metrics(ranked, qrels, k):
    # One query; complete nonnegative qrels; unique ranked document IDs.
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(set(ranked)) != len(ranked) or len(ranked) < k:
        raise ValueError("need at least k distinct ranked documents")
    if not all(np.isfinite(r) and 0 <= r <= 3 for r in qrels.values()):
        raise ValueError("this example uses relevance grades from 0 to 3")
    relevant = {doc for doc, grade in qrels.items() if grade > 0}
    if not relevant:
        raise ValueError("handle unanswerable queries separately")
    top = list(ranked[:k])
    grades = np.array([qrels.get(doc, 0) for doc in top], dtype=float)
    hits = grades > 0
    first = np.flatnonzero(hits)
    discount = np.log2(np.arange(2, k + 2))
    dcg = ((2.**grades - 1) / discount).sum()
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2.**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return {
        "recall": len(relevant.intersection(top)) / len(relevant),
        "hit": float(hits.any()),
        "rr": 0. if len(first) == 0 else 1. / (first[0] + 1),
        "ndcg": dcg / idcg,
    }

scores = docs @ query
ranked = ids[np.argsort(-scores, kind="stable")]
assert ranked.tolist() == ["d1", "d2", "d5", "d3", "d4"]
a = metrics(ranked, qrels, 3)
b = metrics(["d2", "d3", "d1", "d5", "d4"], qrels, 3)
assert np.isclose(a["recall"], 2 / 3) and a["hit"] == a["rr"] == 1
assert .798 < a["ndcg"] < .800
assert a["recall"] == b["recall"] and b["rr"] == .5
assert .574 < b["ndcg"] < .575
```

</details>

<a id="emb-error-decomposition"></a>
<span id="модель-индекс-и-разметка-дают-разные-ошибки"></span>

### Точный и приближённый поиск

Сначала найдём точный top-3 по сохранённым векторам: $d_1,d_2,d_5$.
Он служит эталоном для проверки приближённого индекса
(*approximate nearest neighbors*, ANN). **ANN recall@k** — доля
документов из точного top-k, которую нашёл приближённый поиск.
В этом примере возможны такие результаты:

| Выдача индекса | ANN recall@3 | Recall@3 по разметке |
|---|---:|---:|
| $d_1,d_2,d_5$ | $3/3$ | $2/3$ |
| $d_1,d_3,d_5$ | $2/3$ | $3/3$ |

Во второй строке индекс пропустил $d_2$. С точки зрения точного поиска
это ошибка. Но $d_3$, попавший вместо него, отвечает на вопрос.
Приближение может случайно улучшить отдельную выдачу, поэтому оценку
индекса по точным соседям и оценку релевантности по разметке считают
отдельно.

При разбиении данных связанные фрагменты одного документа, дубликаты
и близкие перефразировки не должны случайно попадать одновременно
в обучение и тест. Для задачи с обновляющейся базой отдельно проверяют
временное разбиение: модель и правила подготовки не должны знать
будущие документы. Отрицательные примеры для обучения добывают из
разрешённого обучающего корпуса, не используя тестовые метки для
настройки. Qrels и состав корпуса фиксируют вместе с результатом:
в новой базе число известных релевантных документов может измениться.

Неразмеченный документ не обязательно нерелевантен. Если новая модель
извлекла хороший ответ за пределами исходного пула разметки, обычный
расчёт может засчитать его ошибкой. Такие случаи разбирают вручную
и дополняют разметку по заранее принятой процедуре для всех сравниваемых
систем. Для неотвечаемых запросов отдельно оценивают отказ или
сигнал отсутствия подходящих документов, а не только top-k.

Окончательная проверка включает срезы по языку, длине, редким сущностям,
отрицаниям, перефразировкам и смене домена; просмотр ошибок соседей;
память индекса, задержку при заданной нагрузке и стоимость пересчёта.
Утверждение «модель лучше» должно указывать набор, метрики и условия.
[MTEB](https://arxiv.org/abs/2210.07316) расширил оценку за пределы
одной STS-задачи, но его сводный балл не выбирает модель за нас для
внутренних инструкций или кода. Нужен отложенный набор запросов
из предполагаемого применения.

Для воспроизведения сохраняют версии модели и токенизатора,
префиксы и инструкции, правила разбиения и обрезки, pooling,
нормировку, размерность и формат векторов, снимок корпуса и qrels,
параметры индекса и метрик. Без этих данных смену геометрии легко
перепутать со сменой корпуса или потерями приближённого индекса.

## Источники и дальнейшее чтение

- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — исходное сравнение
  bi-encoder и cross-encoder для sentence embeddings.
- [DPR](https://arxiv.org/abs/2004.04906) и
  [ACL OpenQA Tutorial](https://github.com/danqi/acl2020-openqa-tutorial) —
  двухбашенное обучение, in-batch и BM25 negatives.
- [Sentence Transformers: Loss Overview](https://www.sbert.net/docs/sentence_transformer/loss_overview.html) —
  форматы выборок и применимые функции потерь.
- [MS MARCO training](https://sbert.net/examples/sentence_transformer/training/ms_marco/README.html) —
  in-batch и явные negatives, фильтрация учителем, MarginMSE;
  оригинальное подробное объяснение на английском.
- [[05 Источники/Papers/GradCache|GradCache]] — исходные этапы алгоритма,
  связь с градиентом полной партии и авторская реализация повторных проходов.
- [[05 Источники/Papers/ContAccum|ContAccum]] — рис. 1, локальные и полные наборы
  кандидатов и роль сохранённых представлений.
- [SimCSE](https://github.com/princeton-nlp/SimCSE) — схема двух dropout-проходов
  и отличие supervised NLI-варианта; ссылки на статью и код.
- [Matryoshka Embeddings](https://sbert.net/examples/sentence_transformer/training/matryoshka/README.html) —
  вложенные размерности, исходный эксперимент и ограничения обрезки.
- [Introduction to Information Retrieval, глава 8](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html) —
  тестовые коллекции, релевантность и оценка ранжированных результатов.
- [MTEB](https://arxiv.org/abs/2210.07316) — широкий benchmark эмбеддингов;
  использовать как внешний ориентир, не как замену доменной оценке.

← [[00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация|Оценивание моделей и контаминация]]

Дальше — [[00 Учебник/15 Embeddings, Retrieval и RAG/61 Retrieval — от BM25 до dense и hybrid|лексический, плотный и гибридный поиск]].
В примере мы сравнили запрос со всеми пятью векторами. Следующая глава
показывает, как найти кандидатов в большом корпусе и зачем к оценкам
эмбеддингов добавляют совпадения по словам.
