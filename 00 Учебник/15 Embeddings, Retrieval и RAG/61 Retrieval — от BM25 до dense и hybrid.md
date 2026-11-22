---
title: "Лексический, плотный и гибридный поиск"
type: textbook-chapter
status: canonical
last_updated: 2026-11-22
last_verified: 2026-11-22
source_language: mixed
aliases:
  - "Retrieval: от BM25 до плотного и гибридного поиска"
previous: "[[00 Учебник/15 Embeddings, Retrieval и RAG/60 Embeddings и metric learning]]"
next: "[[00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction]]"
primary_sources:
  - https://web.stanford.edu/~jurafsky/slp3/11.pdf
  - https://github.com/danqi/acl2020-openqa-tutorial
  - https://arxiv.org/abs/2401.08281
  - https://milvus.io/docs/ivf-flat.md
  - https://milvus.io/docs/hnsw.md
  - https://milvus.io/docs/ivf-pq.md
  - https://aclanthology.org/2023.acl-long.99/
  - https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
  - https://github.com/castorini/anserini/blob/07d380635f1719736c6042c6e971fd7d08ee125e/docs/experiments-beir-fusion.md
  - https://github.com/nmslib/hnswlib/tree/3f3429661187e4c24a490a0f148fc6bc89042b3d
reference_terms:
  - label: "Полнота кандидатов и покрытие сведений"
    label_en: "Candidate recall and evidence coverage"
    anchor: ret-recall
  - label: "Retriever, reader и бюджеты кандидатов"
    label_en: "Retriever, reader and candidate budgets"
    anchor: ret-reader
  - label: "Инвертированный индекс и postings"
    label_en: "Inverted indexes and postings"
    anchor: ret-postings
  - label: "TF–IDF: частота и редкость"
    label_en: "TF–IDF: frequency and rarity"
    anchor: ret-tfidf
  - label: "BM25: насыщение и длина документа"
    label_en: "BM25: saturation and document length"
    anchor: ret-bm25
  - label: "BM25: численный пример"
    label_en: "BM25: worked example"
    anchor: ret-bm25-example
  - label: "Flat search, IP, L2 и косинус"
    label_en: "Flat search, IP, L2 and cosine"
    anchor: ret-flat
  - label: "ANN recall и качество поиска"
    label_en: "ANN recall and retrieval quality"
    anchor: ret-ann
  - label: "IVF: списки, центры и nprobe"
    label_en: "IVF: lists, centroids and nprobe"
    anchor: ret-ivf
  - label: "HNSW: уровни, M и ширина поиска"
    label_en: "HNSW: layers, M and search breadth"
    anchor: ret-hnsw
  - label: "PQ, ADC, остатки и OPQ"
    label_en: "PQ, ADC, residuals and OPQ"
    anchor: ret-pq
  - label: "Faiss: train, add и search"
    label_en: "Faiss: train, add and search"
    anchor: ret-faiss
  - label: "Chunking: границы и перекрытие"
    label_en: "Chunking: boundaries and overlap"
    anchor: ret-chunking
  - label: "Структурное и child–parent retrieval"
    label_en: "Structured and child–parent retrieval"
    anchor: ret-parent
  - label: "Происхождение фрагмента и смещения"
    label_en: "Chunk provenance and offsets"
    anchor: ret-provenance
  - label: "Декомпозиция, multi-query и ограничения запроса"
    label_en: "Decomposition, multi-query and query constraints"
    anchor: ret-query
  - label: "HyDE: гипотетический документ и реальные источники"
    label_en: "HyDE: hypothetical documents and real sources"
    anchor: ret-hyde
  - label: "Pre-/post-filtering и проверка доступа"
    label_en: "Pre-/post-filtering and authorization"
    anchor: ret-filtering
  - label: "Масштабы оценок при гибридном поиске"
    label_en: "Score scales in hybrid retrieval"
    anchor: ret-score-scale
  - label: "RRF: ранги, веса и глубина выдачи"
    label_en: "RRF: ranks, weights and retrieval depth"
    anchor: ret-rrf
  - label: "Union recall и потери после fusion"
    label_en: "Union recall and post-fusion losses"
    anchor: ret-fusion-limits
  - label: "Версионирование корпуса, модели и индекса"
    label_en: "Corpus, model and index versioning"
    anchor: ret-version
  - label: "Смена индекса, кэши и отзыв доступа"
    label_en: "Index cutover, caches and access revocation"
    anchor: ret-cutover
  - label: "Трасса потерянного ответа"
    label_en: "Tracing a lost answer"
    anchor: ret-trace
  - label: "Раздельная оценка каналов, ANN и fusion"
    label_en: "Separate evaluation of channels, ANN and fusion"
    anchor: ret-evaluation
  - label: "HNSW: очереди, efSearch и остановка"
    label_en: "HNSW: queues, efSearch and stopping"
    anchor: ret-hnsw-queue
  - label: "HNSW: выбор связей при построении"
    label_en: "HNSW: construction neighbor selection"
    anchor: ret-hnsw-links
---

# Лексический, плотный и гибридный поиск

В [[60 Embeddings и metric learning#emb-exact-search|предыдущей главе]]
мы сравнили запрос о смене пароля со всеми пятью векторами документов.
Теперь представим большую базу поддержки и вопрос из двух частей:
«Что означает E42 в Atlas v2 и как сменить пароль, если помню текущий?»
Искать придётся и точный код, и описание действия, которое в инструкции
может называться «изменение учётных данных».

<a id="ret-e42-case"></a>

В учебном примере среди прочих страниц найдутся четыре кандидата:

| ID | Что содержит документ |
| --- | --- |
| A | Расшифровка E42 и условия доступа в Atlas v2 |
| B | Общая справка о входе после обновления |
| C | Инструкция смены пароля при известном текущем |
| D | Восстановление забытого пароля |

Для полного ответа нужны A и C. Справка B относится к теме, а D
описывает другое действие. Поиск по словам и поиск по векторам могут
поставить документы в разном порядке. Сохранение нужной страницы хотя бы
в одном списке ещё не означает, что она попадёт в
[[#ret-rrf|объединённую выдачу]].

<a id="ret-recall"></a>
<span id="1-формальная-постановка"></span>

## Полнота кандидатов

Пусть $D$ — корпус, $q$ — запрос. Поиск возвращает упорядоченный список
документов; множество идентификаторов первых $k$ результатов обозначим
$C_k(q)$. Множество релевантных документов — $R(q)$. При $|R(q)|>0$

$$
\operatorname{Recall@k}(q)=
\frac{|C_k(q)\cap R(q)|}{|R(q)|}.
$$

Если для вопроса об Atlas поиск вернул A и B, полнота равна $1/2$.
Hit@2 при этом равен 1: найден хотя бы один из нужных документов.
Но из A можно узнать лишь причину ошибки. Шаги смены пароля записаны в C.
Когда для ответа нужны несколько частей, одного попадания в выдачу
недостаточно.

Та же проблема возникает при разбиении на фрагменты. Три окна из A
могут повторять одну фразу об E42. Число совпадений вырастет, но
инструкции C в выдаче всё равно нет. В таких задачах удобно отдельно
отмечать сведения, необходимые для ответа, и проверять их покрытие.

<a id="ret-reader"></a>
### Поиск кандидатов и чтение

Один из ранних примеров такого разделения — DrQA. Система сначала
находила статьи Википедии, затем нейронный reader извлекал ответ
из отобранного текста:

![[00 Учебник/Assets/Figures/curated/source-first-60-64/openqa-retriever-reader.png|DrQA: document retriever followed by extractive document reader]]

*Danqi Chen, [ACL 2020 Open-Domain QA Tutorial, part IV](https://github.com/danqi/acl2020-openqa-tutorial/blob/3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0/slides/part4-retriever-reader.pdf#page=4),
слайд 4; архитектура [DrQA](https://arxiv.org/abs/1704.00051).
Document Retriever выбирает статьи, Document Reader извлекает ответ.
Это система 2017 года, не схема современного генеративного LLM.
Нажатие на рисунок открывает просмотр с увеличением и ссылкой на файл.*

В такой схеме модуль чтения получает только выбранные источники.
Если поиск не вернул C, повторное ранжирование его не добавит:
оно может лишь изменить порядок имеющихся кандидатов. Если система
после чтения запрашивает ещё документы, это уже второй поиск,
и его результат нужно отличать от результата первого.

Число кандидатов и размер контекста reader — разные параметры.
Можно извлечь 100 фрагментов, повторно ранжировать их и передать
пять. Увеличение первого числа расширяет набор для отбора; увеличение
второго расходует контекст и может добавить повторы. Поэтому отдельно
проверяют полноту кандидатов, качество итогового порядка и достаточность
материала для ответа.

<span id="2-инвертированный-индекс-и-bm25"></span>

## Инвертированный индекс и BM25

<a id="ret-postings"></a>
### От термина к списку документов

На небольшом корпусе видно, что именно хранится в индексе.
Эти три строки нужны для расчёта BM25; они не описывают полные
документы из примера об Atlas.

| ID | Токены документа |
| --- | --- |
| A | ошибка e42 ошибка |
| B | ошибка e17 |
| C | порядок восстановления доступа |

**Инвертированный индекс** сопоставляет термину список документов,
в которых он встречается. Такой список называют *postings list*.
Для нашего корпуса:

| Термин | ID и частоты | Число |
| --- | --- | ---: |
| ошибка | A: 2, B: 1 | 2 |
| e42 | A: 1 | 1 |
| e17 | B: 1 | 1 |
| восстановления | C: 1 | 1 |

Для запроса «ошибка e42» не нужно просматривать текст $C$: достаточно
прочитать два списка. Объединение даёт кандидатов $A,B$; пересечение —
только $A$. Обычное ранжирование не обязано требовать все слова запроса,
поэтому решение «объединять или пересекать» зависит от типа запроса.
В запросе на точный код ошибки можно отдельно сделать код обязательным.

Индекс может хранить только ID и частоты либо дополнительно позиции.
Частоты подходят для BM25, но не позволяют проверить точную фразу:
«доступ разрешён» и «доступ не разрешён» содержат общие слова.
Для фразового поиска нужны позиции и соответствующее правило совпадения.

Токенизация запроса должна согласовываться с индексированием.
Если дефис, точка или знак подчёркивания входят в идентификатор,
их удаление может изменить запрос. Лемматизация помогает сопоставлять
словоформы, но преобразование кода ошибки в обычное слово не помогает.
Обычно текстовые поля и идентификаторы обрабатывают раздельно.
Исходную строку сохраняют для проверки точных совпадений.

<a id="ret-tfidf"></a>
### TF–IDF: частота и редкость

В разреженном представлении координаты соответствуют терминам словаря.
В документе занята лишь малая часть этих координат. Частота показывает,
насколько заметен термин внутри документа, а обратная документная
частота — насколько он отличает документ от остальных.

Например, при IDF $=\log_{10}(N/n_t)$ термин, встречающийся в одном
из 1000 документов, получает 3; встречающийся в 100 — 1; во всех — 0.
Здесь $n_t$ считает **документы**, а не общее число вхождений.
Сто повторений в одном документе увеличивают term frequency,
но не превращают один документ в сто.

Частоту часто ослабляют логарифмом, затем сравнивают нормированные
TF–IDF-векторы по косинусу. У DrQA использовались униграммы и биграммы:

![[00 Учебник/Assets/Figures/curated/source-first-60-64/openqa-tfidf-retriever.png|DrQA TF–IDF retrieval over Wikipedia articles]]

*Danqi Chen, [тот же tutorial](https://github.com/danqi/acl2020-openqa-tutorial/blob/3c82cf3bfa7f7f26aa4016cf4c48b83f1a4eaad0/slides/part4-retriever-reader.pdf#page=5),
слайд 5. На рисунке именно TF–IDF, **не BM25**. Документом в этой
системе была целая статья Википедии; нижняя строка слайда указывает
на отличие от поиска отдельных фрагментов.*

Словарь, правило частоты, IDF и нормировка определяют конкретный
вариант TF–IDF. Эти соглашения сохраняют при воспроизведении чисел
со слайда. Подробная последовательность
«векторы → веса → оценки → индекс» есть в
[Jurafsky & Martin, SLP3, §11.1](https://web.stanford.edu/~jurafsky/slp3/11.pdf).

<a id="ret-bm25"></a>
### BM25: насыщение частоты и длина

Слово, повторённое десять раз, не обязательно делает документ
в десять раз полезнее. Кроме того, в длинном документе совпадение
чаще возникает просто из-за объёма. BM25 учитывает эти две причины.
Зафиксируем вариант формулы для уникальных терминов запроса:

$$
\begin{aligned}
L_d&=1-b+b\frac{|d|}{\operatorname{avgdl}},\\
F(t,d)&=\frac{f(t,d)(k_1+1)}{f(t,d)+k_1L_d},\\
\operatorname{BM25}(q,d)
&=\sum_{t\in\operatorname{unique}(q)}
\operatorname{IDF}(t)F(t,d).
\end{aligned}
$$

$f(t,d)$ — число вхождений термина; $|d|$ — длина документа в токенах;
$\operatorname{avgdl}$ — средняя длина по корпусу; $k_1>0$ управляет
насыщением частотного множителя $F(t,d)$; $b\in[0,1]$ — влиянием длины.
Повторение слова в самом запросе здесь не добавляет новый вклад:
это явное соглашение данной
формулы, а не свойство всех вариантов BM25.

При $|d|=\operatorname{avgdl}$ и $k_1=1.2$ множитель частоты равен:

| Частота $f$ | Множитель $f(k_1+1)/(f+k_1)$ |
| ---: | ---: |
| 1 | 1 |
| 3 | 1.571 |
| 10 | 1.964 |
| $f\to\infty$ | 2.2 |

Частота выросла с 1 до 10, а множитель — примерно вдвое.
При $b=0$ длина не влияет на оценку. При положительном $b$
для длинного документа $L_d$ больше, поэтому при **той же частоте**
вклад меньше. Это сравнение не утверждает, что длинные документы
всегда проигрывают: они могут содержать больше совпадений и больше
нужных сведений.

Выберем положительную форму IDF:

$$
\operatorname{IDF}(t)=
\ln\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right).
$$

Именно такая IDF указана в
[Lucene BM25Similarity](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html).
В отличие от $\log(N/n_t)$, она остаётся положительной даже при $n_t=N$.
Встречаются другие варианты IDF и общие множители оценок; при переносе
порогов и сравнении реализаций формулу нужно фиксировать.

<a id="ret-bm25-example"></a>
<span id="один-термин-в-числах"></span>

### Как длина меняет оценку BM25

Пусть $N=1000$, $n_t=10$, $f=3$, $|d|=200$,
$\operatorname{avgdl}=100$, $k_1=1.2$, $b=0.75$.
Тогда IDF $\approx4.55738$, $L_d=0.25+0.75\cdot2=1.75$.
Знаменатель равен $3+1.2\cdot1.75=5.1$, числитель частоты —
$3\cdot2.2=6.6$. Вклад термина:

$$
4.55738\cdot\frac{6.6}{5.1}\approx5.89779.
$$

У документа длиной 100 при той же частоте $L_d=1$, знаменатель
равен 4.2, вклад — примерно 7.16160. Числа относятся к выбранной
формуле в вещественной арифметике. Продуктовая реализация может
иначе кодировать длины полей или опускать общий постоянный множитель.

В примере на вход подаются уже выделенные токены. Лингвистический
анализатор настраивается отдельно; здесь проверяются postings и оценки.

<details id="ret-code-bm25">
<summary>Построение postings и расчёт BM25</summary>

```python
from collections import Counter, defaultdict
import math

def make_bm25(documents):
    if not documents or any(not tokens for tokens in documents.values()):
        raise ValueError("need a nonempty corpus of nonempty token lists")
    lengths = {doc: len(tokens) for doc, tokens in documents.items()}
    postings = defaultdict(dict)
    for doc, tokens in documents.items():
        for term, count in Counter(tokens).items():
            postings[term][doc] = count
    return postings, lengths

def bm25_search(query, postings, lengths, k1=1.2, b=.75):
    if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("need k1 > 0 and b in [0, 1]")
    n, average = len(lengths), sum(lengths.values()) / len(lengths)
    scores = defaultdict(float)
    for term in sorted(set(query)):
        found = postings.get(term, {})
        if not found:
            continue
        idf = math.log1p((n - len(found) + .5) / (len(found) + .5))
        for doc, frequency in found.items():
            norm = 1 - b + b * lengths[doc] / average
            scores[doc] += idf * frequency * (k1 + 1) / (frequency + k1 * norm)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))

documents = {
    "A": ["ошибка", "e42", "ошибка"],
    "B": ["ошибка", "e17"],
    "C": ["порядок", "восстановления", "доступа"],
}
postings, lengths = make_bm25(documents)
ranked = bm25_search(["ошибка", "e42"], postings, lengths)
assert [doc for doc, _ in ranked] == ["A", "B"]
assert ranked[0][1] > ranked[1][1] > 0
assert bm25_search(["неизвестный_код"], postings, lengths) == []
assert ranked == bm25_search(["ошибка", "ошибка", "e42"], postings, lengths)
```

</details>

В таком индексе запрос «вернуть учётную запись» не совпадёт с терминами
«восстановления доступа» без расширения запроса или словаря соответствий.
Поиск по векторам позволяет сопоставить тексты без общих слов. Для E42
при этом важно не потерять и буквальное совпадение кода.

<span id="3-плотный-поиск"></span>

## Плотный поиск

<a id="ret-flat"></a>
<span id="что-остаётся-после-вычисления-эмбеддингов"></span>

### Точный поиск по векторам

Документный энкодер заранее вычисляет $e_d(d)$; при поступлении
запроса вычисляется $e_q(q)$. Для скалярного произведения выбирают

$$
C_k(q)=\operatorname{top-k}_{d\in D}
e_q(q)^\top e_d(d).
$$

Здесь top-k возвращает ID документов вместе с оценками. Модель,
инструкции, pooling, обрезка и нормировка должны соответствовать
обучению; они разобраны в
[[60 Embeddings и metric learning|главе об эмбеддингах]].
Смысловой поиск не устраняет необходимость проверять точный код,
дату, отрицание или версию продукта. Он меняет функцию оценки пары.

**Flat search** сравнивает запрос со всеми сохранёнными векторами.
Для $N$ документов размерности $d$ это $N$ скалярных произведений
и выбор $k$ наибольших значений. Для пакета запросов сравнения можно
выполнять матричным умножением. «Точный» здесь означает точный
относительно данной метрики и сохранённых представлений, а не
безошибочный относительно человеческой релевантности.

Для косинуса можно нормировать обе стороны и использовать
`IndexFlatIP`. `IndexFlatL2` возвращает квадраты евклидовых расстояний:
меньше означает ближе. Подстановка ненормированных векторов
в IP вместо L2 может изменить порядок. Название индекса не исправляет
неверно выбранную метрику.

<a id="ret-ann"></a>
### Где возникает приближение

При большом корпусе можно проверять не все векторы либо хранить
приближённые представления. Эти решения дают два разных источника потерь:

| Решение | Что может потеряться |
| --- | --- |
| Не посетить часть базы | Нужный вектор останется вне просмотренных кандидатов |
| Сжать координаты | Просмотренный вектор получит искажённую оценку |
| Сделать оба шага | Ошибки отбора и оценки могут складываться |

Поэтому IVF, HNSW и PQ не являются тремя взаимозаменяемыми названиями.
IVF и HNSW прежде всего сокращают просмотр; PQ сжимает представления.
Комбинации вроде IVF+PQ совмещают оба решения. Поддержка конкретных
комбинаций зависит от библиотеки и её версии.

В [[60 Embeddings и metric learning#emb-error-decomposition|примере о смене пароля]]
точный поиск поставил страницу восстановления выше мобильной инструкции.
Индекс, который идеально повторил эту выдачу, получил ANN recall@3,
равный 1, но нашёл лишь два из трёх нужных документов. Точность
приближения измеряет, насколько индекс воспроизводит оценки модели.
Релевантность отвечает на другой вопрос: нашёлся ли документ с ответом.
[Faiss, §3.2](https://arxiv.org/abs/2401.08281) прямо различает
приближение векторного поиска и качество конечной задачи.

В этой главе ANN recall@k означает размер пересечения приближённого
и точного top-k, делённый на $k$. В источниках встречается также
1-recall@k: попал ли один точный ближайший сосед в первые $k$.
Результаты с такими разными определениями нельзя сравнивать по
одному названию «recall».

<a id="ret-ivf"></a>
### IVF: сначала список, затем векторы

**Inverted File index** группирует векторы по ближайшим центроидам.
Центроиды обычно получают k-means на представительной выборке.
Для каждого центра хранится список ID и соответствующих векторов.
При запросе сначала выбирают несколько близких центров, затем
оценивают только содержимое их списков:

![[00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/ivf-build.png|IVF construction: centroids, assignments and inverted lists]]

*Milvus documentation contributors,
[IVF_FLAT — Overview](https://milvus.io/docs/ivf-flat.md),
[исходная схема](https://milvus-docs.s3.us-west-2.amazonaws.com/assets/IVF-FLAT-workflow.png).
Apache-2.0, [лицензия документации](/licenses/milvus-docs-9acdc999-Apache-2.0.txt).
Без изменений. Шаги 1–2: кластеры и списки; шаг 3: выбор списка для запроса.*

Здесь «инвертированный» означает отображение **центр → ID векторов**.
У лексического индекса выше было отображение **термин → ID документов**.
Название похоже, но ключи и правило выбора кандидатов различаются.
Граница кластера проходит по близости к центрам, а не по релевантности
запросу.

Параметр `nlist` задаёт количество списков, `nprobe` — сколько
списков посетить для одного запроса. В **IVFFlat** векторы внутри
списков не сжимаются; расстояния до просмотренных кандидатов считают
по сохранённым координатам. Потеря возможна из-за непросмотренного списка.

Рассмотрим одномерную базу: $A=4$, $B=5.1$, $C=-4$, $D=14.9$.
Заданы центры 0 и 10. К первому относятся $A,C$, ко второму — $B,D$;
средние каждой пары действительно равны этим центрам.
Для запроса $q=4.9$ центр 0 ближе, чем 10: расстояния 4.9 и 5.1.
При `nprobe=1` просмотрим $A,C$ и выберем $A$ на расстоянии 0.9.
Однако настоящий ближайший — $B$ на расстоянии 0.2, по другую
сторону границы.

![[00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/ivf-search.png|IVF search across one or several selected lists]]

*Тот же источник,
[оригинальный рисунок](https://milvus-docs.s3.us-west-2.amazonaws.com/assets/IVF-FLAT-workflow-2.png).
Слева сосед лежит вне выбранного кластера; справа посещаются три
списка, `nprobe=3`. Это двумерная схема источника, не наш одномерный пример.*

При `nprobe=2` наш пример посещает оба списка и находит $B$.
Полный просмотр всех списков IVFFlat восстанавливает точный поиск
при той же метрике и без дополнительных ограничений просмотра.
Для IVFPQ этого недостаточно: оценки всё ещё вычисляются по сжатым данным.

<details id="ret-code-ivf">
<summary>Выбор списков IVF</summary>

```python
import numpy as np

def ivf_flat(query, vectors, centers, nprobe, k):
    query, vectors, centers = [np.asarray(x, dtype=float) for x in (query, vectors, centers)]
    if vectors.ndim != 2 or centers.ndim != 2 or query.shape != (vectors.shape[1],):
        raise ValueError("need a query and two vector matrices")
    if len(vectors) == 0 or len(centers) == 0 or vectors.shape[1] == 0:
        raise ValueError("need nonempty vectors and centers")
    if centers.shape[1] != vectors.shape[1] or not all(np.isfinite(x).all() for x in (query, vectors, centers)):
        raise ValueError("incompatible or non-finite vectors")
    if not isinstance(nprobe, int) or isinstance(nprobe, bool) or not 1 <= nprobe <= len(centers):
        raise ValueError("invalid number of probed lists")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be positive")
    assignment = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    selected = np.argsort(((centers - query) ** 2).sum(axis=1), kind="stable")[:nprobe]
    candidate_ids = np.flatnonzero(np.isin(assignment, selected))
    distances = ((vectors[candidate_ids] - query) ** 2).sum(axis=1)
    order = np.argsort(distances, kind="stable")[:k]
    return candidate_ids[order], distances[order]

vectors = np.array([[4.], [5.1], [-4.], [14.9]])
centers, query = np.array([[0.], [10.]]), np.array([4.9])
one, d_one = ivf_flat(query, vectors, centers, 1, 1)
all_lists, d_all = ivf_flat(query, vectors, centers, 2, 1)
assert one.tolist() == [0] and np.allclose(d_one, [.81])
assert all_lists.tolist() == [1] and np.allclose(d_all, [.04])
```

</details>

При примерно равных размерах списков просмотр затрагивает порядка
$N\cdot\text{nprobe}/\text{nlist}$ векторов, дополнительно к поиску
центров. Это оценка числа сравнений, не формула задержки. Несбалансированные
списки, размер пакета запросов, размещение памяти и приближённый поиск
самих центров меняют стоимость. Увеличение `nlist` уменьшает списки,
но может потребовать большего `nprobe` для прежней полноты.
Фиксированного «правильного» `nlist` для всех корпусов нет.

<a id="ret-hnsw"></a>
### HNSW: поиск по графу

В **Hierarchical Navigable Small World** вершина соответствует вектору,
рёбра — переходам между векторами. Внизу находятся все элементы;
на верхние уровни случайно попадает меньшая часть. На верхнем уровне
поиск быстро приближается к нужной области, затем продолжает с найденной
вершины уровнем ниже:

![[00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/hnsw.png|HNSW search through upper layers and the base-layer graph]]

*Milvus documentation contributors,
[HNSW — Overview](https://milvus.io/docs/hnsw.md),
[исходная схема](https://milvus-docs.s3.us-west-2.amazonaws.com/assets/hnsw.png).
Apache-2.0; без изменений. Пунктир между жёлтыми точками связывает
появления одной вершины на уровнях, стрелки показывают поиск.
Фиолетовая точка — запрос,
а не дополнительный проиндексированный документ.*

На верхних уровнях используется жадное продвижение: перейти к соседу,
если он ближе к запросу, и спуститься, когда улучшения нет.
На нижнем уровне нельзя сводить алгоритм к одному такому маршруту.
Поиск поддерживает несколько перспективных кандидатов и очередь
следующих проверок. Это позволяет рассмотреть и переход, который
не улучшает текущий лучший ответ сразу.

Для иллюстрации возьмём условный граф-цепочку $A\leftrightarrow B
\leftrightarrow C$ с координатами $A=4$, $B=5$, $C=0.1$ и запросом 0.
Стартуем в $A$. Строго жадный шаг не пойдёт в $B$: 5 дальше, чем 4,
хотя за $B$ находится ближайший $C$. Поиск, сохранивший $B$ как
дополнительного кандидата, сможет перейти через него и обнаружить $C$.
Это пример ограничения жадного обхода, не утверждение, что построенный
HNSW обязательно имеет такую цепочку.

Три параметра описывают разные части работы:

- `M` ограничивает число связей и влияет на память и маршруты.
  На базовом уровне отдельная реализация может допускать больше связей,
  чем на верхних; надпись `M=5` на схеме не задаёт универсальную
  формулу числа рёбер.
- `efConstruction` задаёт ширину рассмотрения кандидатов при добавлении
  вершины. Большее значение требует больше работы при построении,
  но даёт больше возможностей выбрать полезные связи.
- `efSearch` — ширина поиска при запросе; в Milvus параметр называется
  `ef`. Это не число возвращаемых документов и не точное число
  вычислений расстояния.

Более широкий поиск обычно повышает полноту ценой работы, но эффект
зависит от построенного графа. Он не даёт гарантии точного top-k.
В памяти хранятся и векторы, и связи; к сравнению «HNSW быстрее Flat»
нужно добавить размер графа, время построения и нагрузку.
Обзор графового поиска с очередью кандидатов есть в
[The Faiss Library, §5.2](https://arxiv.org/abs/2401.08281);
исходный алгоритм — [Malkov & Yashunin](https://arxiv.org/abs/1603.09320).

<a id="ret-hnsw-queue"></a>
#### Candidate queues and the stopping rule

The authors distinguish the result count from the search width:

> ```ef``` - the size of the dynamic list for the nearest neighbors (used during the search). Higher ```ef```
> leads to more accurate but slower search. ```ef``` cannot be set lower than the number of queried nearest neighbors
> ```k```. The value ```ef``` of can be anything between ```k``` and the size of the dataset.

*hnswlib authors, [HNSW algorithm parameters, v0.8.0](https://github.com/nmslib/hnswlib/blob/3f3429661187e4c24a490a0f148fc6bc89042b3d/ALGO_PARAMS.md).
Original English retained; [Apache-2.0](/licenses/hnswlib-3f342966-Apache-2.0.txt).*

At the base layer, keep two priority queues. The frontier contains vertices
whose adjacency lists may still be inspected, with the closest vertex first.
The result queue retains at most `ef` current best vertices, with the farthest
one available for removal. A visited set prevents repeated distance calculations
for vertices reached through several edges.

Start both queues with the entry vertex. Pop the closest frontier candidate
and inspect its previously unseen neighbors. A neighbor enters the queues if
the result queue is not full or it is closer than the current worst result.
If the result queue grows beyond `ef`, remove its farthest member.
The two queues serve different purposes: a vertex removed from the results
can still have a pending frontier entry.

For the unfiltered, no-deletion search used here, stop when the closest remaining
frontier candidate is farther than the worst retained result and the result
queue is full. This is an approximate stopping rule. A far frontier vertex can
still have an edge to a very close, unseen vertex; its own distance is not
a lower bound on all vertices reachable through it.

Return to the chain above and use squared L2 distances. From query 0,
the distances to A, B and C are 16, 25 and 0.01.

| Width / expansion | Result queue and next step |
| --- | --- |
| `ef=1`, A | A: 16. Reject B: 25; the frontier becomes empty. |
| `ef=2`, A | A: 16, B: 25. Keep B because the result queue has space. |
| `ef=2`, B | C: 0.01, A: 16. Discover C and remove B from results. |
| `ef=2`, C | C: 0.01, A: 16. No unseen neighbors; return C for `k=1`. |

The wider search expands three vertices even though `ef=2`.
The width limits retained results, not the total number of distance calculations
or expansions. The final response can still contain just one neighbor.

The implementation below follows the two-queue logic in
[hnswlib `searchBaseLayerST`](https://github.com/nmslib/hnswlib/blob/3f3429661187e4c24a490a0f148fc6bc89042b3d/hnswlib/hnswalg.h#L311).
It uses a supplied one-dimensional graph to expose the search steps.
It does not implement insertion, upper layers, deletion, filtering or the
optimized memory layout of hnswlib. The Python code and worked graph are
Bookvar teaching additions, not a performance measurement of the library.

```python
import heapq
import math


def search_layer_demo(points, graph, query, entry, ef, k=1):
    if (not isinstance(ef, int) or isinstance(ef, bool)
            or not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= ef):
        raise ValueError("need integer 1 <= k <= ef")
    if entry not in points or set(points) != set(graph):
        raise ValueError("need an entry and an adjacency list for every vertex")
    if not math.isfinite(query) or any(not math.isfinite(x) for x in points.values()):
        raise ValueError("need finite one-dimensional coordinates")
    if any(v not in points for neighbors in graph.values() for v in neighbors):
        raise ValueError("every edge must point to a known vertex")

    distance = lambda v: (points[v] - query) ** 2
    frontier = [(distance(entry), entry)]       # nearest unexpanded first
    best = [(-distance(entry), entry)]          # farthest retained first
    visited, expanded = {entry}, []
    while frontier:
        d, vertex = heapq.heappop(frontier)
        if len(best) == ef and d > -best[0][0]:
            break
        expanded.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            dn = distance(neighbor)
            if len(best) < ef or dn < -best[0][0]:
                heapq.heappush(frontier, (dn, neighbor))
                heapq.heappush(best, (-dn, neighbor))
                if len(best) > ef:
                    heapq.heappop(best)
    return sorted((-d, v) for d, v in best)[:k], visited, expanded


points = {"A": 4., "B": 5., "C": .1}
graph = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
narrow, _, _ = search_layer_demo(points, graph, 0., "A", ef=1)
wide, visited, expanded = search_layer_demo(points, graph, 0., "A", ef=2)
assert narrow[0][1] == "A" and wide[0][1] == "C"
assert visited == set(points) and expanded == ["A", "B", "C"]
assert len(expanded) > 2  # ef is not a hard limit on graph expansions
assert math.isclose(wide[0][0], .01)
```

Increasing the width cannot create a missing graph connection. If a component
containing the true nearest neighbor is unreachable from the entry, even a
width equal to the whole database cannot reach it. The graph structure and
entry selection therefore affect recall separately from the query-time width.
Returning to all IVF lists has a different property:
with IVFFlat it explicitly visits the complete database.

<a id="ret-hnsw-links"></a>
#### Why construction does not keep only the closest links

In hnswlib, [getNeighborsByHeuristic2](https://github.com/nmslib/hnswlib/blob/3f3429661187e4c24a490a0f148fc6bc89042b3d/hnswlib/hnswalg.h#L443)
considers candidate neighbors in increasing distance from the new vertex.
After accepting a neighbor, it can reject another candidate that is closer
to an already accepted neighbor than to the new vertex. The candidate may
provide a redundant route through the same nearby region.

For a new vertex at 0 and candidates A=1, B=1.1 and C=−1.2, the two closest
candidates are A and B. With a two-link budget, the heuristic first accepts A.
It rejects B because the A–B distance is 0.1, smaller than B's distance 1.1
from the new vertex. It then accepts C: the A–C distance 2.2 exceeds C's
distance 1.2 from the new vertex. The two links point to opposite sides
rather than the same tight group.

The actual code compares the configured distance function; squared L2 gives
the same decisions in this example. This local heuristic does not guarantee
a globally optimal graph or complete connectivity. `efConstruction` controls
the candidate search used while adding vertices; `M` limits the retained
connections. A wide construction search and a large connection budget change
different parts of the graph-building procedure.

<a id="ret-pq"></a>
### Product quantization: код вместо координат

В обычном векторном квантовании каждый полный вектор заменяют
ближайшим представителем из кодовой книги. Для длинного кода одна
общая книга становится слишком большой. **Product quantization**
разбивает $d$ координат на $M$ непересекающихся частей и обучает
отдельную небольшую книгу для каждой части. При восстановлении
выбранные подвекторы соединяются обратно.

![[00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/pq-codebooks.png|Product quantization: subspaces, codebooks and vector codes]]

*Milvus documentation contributors,
[IVF_PQ — PQ](https://milvus.io/docs/ivf-pq.md),
[оригинальная схема](https://milvus-docs.s3.us-west-2.amazonaws.com/assets/ivf-pq-1.png).
Apache-2.0; без изменений. Полный вектор разбивается на части,
для каждой сохраняется индекс выбранного представителя.
В формулах ниже используются $M$ частей и $K$ представителей
с индексами $0,\ldots,K-1$.*

Если в каждой книге $K=2^b$ представителей, код требует $Mb$ бит
на вектор до округления и служебных данных. Книги общие для базы;
их не нужно записывать заново для каждого вектора.
Для $d=768$, $M=96$, $b=8$ код занимает 96 байт против
$768\cdot4=3072$ байт float32: сжатие самих координат в 32 раза.
Это не 32-кратное уменьшение всего сервиса. Остаются ID, центры,
кодовые книги, списки или граф, метаданные и копии для репликации.

Возьмём четырёхмерный вектор и две части по две координаты.
Пусть первая книга содержит $(0,0)$ и $(2,2)$,
вторая — $(0,0)$ и $(4,4)$.
Вектор $x=(0.1,0.2,3.9,4.1)$ получает код $(0,1)$.
Восстановленный $\tilde x=(0,0,4,4)$ отличается от исходного;
квадрат ошибки равен $0.01+0.04+0.01+0.01=0.07$.

Для запроса $q=(0.2,0,4,4)$ можно сначала вычислить таблицу
квадратов расстояний до представителей:

| Часть запроса | Код 0 | Код 1 |
| --- | ---: | ---: |
| $(0.2,0)$ | 0.04 | 7.24 |
| $(4,4)$ | 32 | 0 |

Для кода $(0,1)$ достаточно сложить 0.04 и 0.
Это **asymmetric distance computation, ADC**:
запрос остаётся несжатым, а документ сравнивается через восстановление.
Физически восстанавливать каждый полный вектор не нужно —
используются индексы в общей таблице:

$$
\begin{aligned}
T_{m,j}&=\|q^{(m)}-c_{m,j}\|_2^2,\\
\widehat D(q,x)&=\sum_{m=0}^{M-1}T_{m,\operatorname{code}_m(x)}.
\end{aligned}
$$

> The elements of `dt.dtable` are computed by comparing each sub-vector of the query
> to the codewords for each subspace.
> More specifically, `dt.dtable[m][ks]` contains the squared Euclidean distance between
> (1) the `m`-th sub-vector of the query and (2) the `ks`-th codeword
> for the `m`-th subspace (`pq.codewords[m][ks]`).

Yusuke Matsui, [nanopq tutorial](https://nanopq.readthedocs.io/en/latest/source/tutorial.html),
[MIT](/licenses/nanopq-898fad82-MIT.txt). В обозначениях главы эта таблица — $T$.

![[00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/pq-search.png|PQ distance lookup tables and summed subspace distances]]

*Тот же источник,
[исходная схема вычисления расстояний](https://milvus-docs.s3.us-west-2.amazonaws.com/assets/ivf-pq-2.png).
На рисунке в конце берётся квадратный корень для L2.
В нашей таблице и коде оставлен квадрат L2: он даёт тот же порядок,
но другие численные значения.*

Точное квадратное расстояние от $q$ до $x$ равно 0.07, приближённое —
0.04. Другой вектор $y=(0.3,0,4,4)$ имеет точное расстояние 0.01,
но получает тот же код и ту же приближённую оценку 0.04.
PQ уже не различит эти два документа. После сжатия малые изменения
координат могут исчезнуть или изменить порядок соседей.

<details id="ret-code-pq">
<summary>Коды PQ и таблица расстояний</summary>

```python
import numpy as np

books = np.array([[[0., 0.], [2., 2.]],
                  [[0., 0.], [4., 4.]]])  # M=2, K=2, d/M=2
vectors = np.array([[.1, .2, 3.9, 4.1], [.3, 0., 4., 4.]])
query = np.array([.2, 0., 4., 4.])

parts = vectors.reshape(len(vectors), 2, 2)
codes = ((parts[:, :, None, :] - books[None, :, :, :]) ** 2).sum(axis=3).argmin(axis=2)
reconstructed = books[np.arange(2)[None, :], codes].reshape(len(vectors), 4)
table = ((query.reshape(2, 2)[:, None, :] - books) ** 2).sum(axis=2)
approximate = table[np.arange(2)[None, :], codes].sum(axis=1)
exact = ((vectors - query) ** 2).sum(axis=1)

assert codes.tolist() == [[0, 1], [0, 1]]
assert np.allclose(table, [[.04, 7.24], [32., 0.]])
assert np.allclose(approximate, [.04, .04])
assert np.allclose(exact, [.07, .01])
assert np.allclose(approximate, ((reconstructed - query) ** 2).sum(axis=1))
```

</details>

В IVFPQ сначала отбираются списки, затем сравниваются сжатые векторы.
Часто кодируют остаток $x-c$ относительно центра списка, а не сам $x$.
Тогда для L2 сравнивают $q-c$ с восстановленным остатком.
Увеличение `nprobe` исправляет только часть ошибок отбора, но
не возвращает координаты, потерянные квантованием.
Точный пересчёт оценок по исходным векторам помогает на уже
найденном расширенном списке, но не возвращает пропущенный документ.

**OPQ** предварительно обучает поворот пространства, чтобы разбиение
на части было удобнее для квантования. Один ортогональный поворот,
применённый к запросам и документам, сохраняет исходные L2-расстояния;
качество меняется из-за последующего кодирования. Это отличается
от whitening из предыдущей главы, который в общем случае меняет
саму метрику. Объяснение таблиц ADC, восстановления и OPQ есть
в [учебном tutorial Yusuke Matsui](https://nanopq.readthedocs.io/en/latest/source/tutorial.html)
и [Faiss, §4.1–4.2](https://arxiv.org/abs/2401.08281).

<a id="ret-faiss"></a>
<span id="проверка-настоящего-индекса-faiss"></span>

### Сравнение Flat и IVF в Faiss

В учебном примере ниже нет языковой модели: используются 512
синтетических восьмимерных векторов. Он проверяет смысл `train`,
`add`, `search` и `nprobe`, а не качество поиска текстов.
Для запуска нужны NumPy и `faiss-cpu`.
Два официальных [tutorial Faiss](https://github.com/facebookresearch/faiss/tree/42aac2fb4e17a66a6b17e7a99e3df229bcf757b8/tutorial/python)
показывают те же этапы на более крупной базе.
Пример адаптирован к небольшой проверке; исходники Meta / Facebook
и contributors доступны по [MIT](/licenses/faiss-42aac2fb-MIT.txt).

<details id="ret-code-faiss">
<summary>Сравнение Flat и IVFFlat</summary>

```python
import faiss
import numpy as np

faiss.omp_set_num_threads(1)
rng = np.random.default_rng(61)
database = rng.normal(size=(512, 8)).astype("float32")
queries = rng.normal(size=(8, 8)).astype("float32")
flat = faiss.IndexFlatL2(8)
flat.add(database)
exact_distances, exact_ids = flat.search(queries, 3)

ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(8), 8, 8, faiss.METRIC_L2)
assert not ivf.is_trained
ivf.train(database)
assert ivf.is_trained and ivf.ntotal == 0
ivf.add(database)
assert ivf.ntotal == len(database)
ivf.nprobe = ivf.nlist  # visit all 8 lists; no compression
all_distances, all_ids = ivf.search(queries, 3)
assert np.array_equal(exact_ids, all_ids)
assert np.allclose(exact_distances, all_distances, atol=1e-5)
```

</details>

`train` здесь обучает геометрическое разбиение, не нейросеть.
После него база ещё пуста. `add` распределяет векторы по спискам.
Возвращаемые ID в этом примере — номера строк; для собственных
ID и удаления нужен соответствующий интерфейс индекса.
При небольшом `nprobe` равенство с Flat уже не гарантируется.

<span id="4-что-именно-индексировать"></span>

## Разбиение документов на фрагменты

<a id="ret-chunking"></a>
### Границы фрагмента меняют доступную информацию

Документ, раздел, абзац, предложение и ячейка таблицы дают разные
единицы поиска. Один вектор длинного документа должен представить
несколько тем сразу. У короткого фрагмента другая проблема:
«разрешается после проверки» без названия операции и версии
может быть бесполезно или вводить в заблуждение.

Пусть в документе A условие доступа записано так. Для примера считаем
слова и точку отдельными токенами:

> Сервис Atlas разрешает доступ только после проверки прав в версии v2 .

Разобьём его на окна по шесть токенов без пересечения.
Первое заканчивается на «после», второе начинается с «проверки».
Условие «только после проверки прав» разорвано.
При перекрытии в три токена появится окно
«доступ только после проверки прав в», которое сохранит условие целиком.
Но даже оно не содержит названия сервиса и номера версии:
их нужно сохранить в метаданных или получить из родительского раздела.

<details id="ret-code-chunks">
<summary>Окна с перекрытием</summary>

```python
def token_windows(tokens, width, overlap):
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError("width must be positive")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < width:
        raise ValueError("need 0 <= overlap < width")
    chunks = []
    for start in range(0, len(tokens), width - overlap):
        end = min(len(tokens), start + width)
        chunks.append((start, end, tokens[start:end]))
        if end == len(tokens):
            break
    return chunks

tokens = "Сервис Atlas разрешает доступ только после проверки прав в версии v2 .".split()
plain = token_windows(tokens, 6, 0)
overlapping = token_windows(tokens, 6, 3)
evidence_start, evidence_end = 4, 8  # "только после проверки прав"
contains = lambda chunks: any(start <= evidence_start and end >= evidence_end
                              for start, end, _ in chunks)
assert not contains(plain) and contains(overlapping)
assert overlapping[1][2] == ["доступ", "только", "после", "проверки", "прав", "в"]
assert sum(end - start for start, end, _ in overlapping) == 18
assert len(tokens) == 12
```

</details>

Это пример окон по готовым токенам, не промышленный splitter.
У реального токенизатора длина в токенах отличается от числа слов
и символов. Заголовок, инструкция и специальные токены тоже расходуют
входное окно энкодера. В
[Hugging Face Advanced RAG, §1.1](https://huggingface.co/learn/cookbook/en/advanced_rag)
разобрано, почему ограничение в символах не гарантирует соблюдения
токенного лимита.

Для длинного текста при постоянных ширине $w$ и перекрытии $o$
число записываемых токенов примерно увеличивается в $w/(w-o)$ раз.
При перекрытии половины окна асимптотический множитель — 2.
В коротком примере выше записано 18 токенов вместо 12, множитель 1.5:
на границах нет полного повторения. Это оценка дублирования данных,
не обещание такого же изменения задержки.

<a id="ret-parent"></a>
### Структура вместо одного размера окна

Если документ уже содержит заголовки, абзацы, таблицы или функции
кода, сначала используют эти границы. Слишком большой раздел можно
разбить дополнительно, сохраняя его заголовочный путь.
Строка таблицы должна оставаться связана с заголовками столбцов,
единицами измерения и примечаниями. Последовательное склеивание
ячеек без этой связи может изменить смысл найденных чисел.

**Child–parent retrieval** разделяет единицу поиска и единицу чтения.
Небольшие дочерние фрагменты помогают найти точное место, затем
система получает окружающий родительский раздел. Например,
поиск нашёл условие доступа из примера, а родитель возвращает
название Atlas и применимую версию v2. Повторные попадания
в одного родителя объединяют, чтобы не передавать его несколько раз.

Расширение контекста должно соблюдать права на весь получаемый материал.
Доступный дочерний фрагмент не даёт автоматически права читать
любого родителя или соседний раздел. Заголовки и соседние предложения
не должны добавляться после проверки ACL в обход этой проверки.

Иерархический поиск сначала выбирает документы, затем разделы
и фрагменты. Он сокращает область второго поиска, но ошибка
первого этапа исключает все дочерние ответы. Полезно сравнить
его с прямым поиском по фрагментам при сопоставимом бюджете,
а не считать иерархию бесплатным улучшением.

<a id="ret-provenance"></a>
### Происхождение фрагмента

Для цитирования и обновлений одного текста недостаточно. Хранят
`document_id`, версию источника, ID родителя, заголовочный путь,
начало и конец фрагмента, единицу смещения, дату и правила доступа.
Позиция в нормализованном тексте, номер PDF-страницы и смещение
в байтах исходного файла — разные координаты; их нельзя смешивать.

Размер фрагмента выбирают по запросам с размеченными достаточными
участками источника. Сравнивают, сохраняются ли нужные определения,
условия и ссылки, сколько дубликатов попадает в выдачу и сколько
контекста приходится передать дальше. Универсальное число вроде
«всегда 512 токенов» не учитывает ни структуру корпуса, ни входное
окно выбранного энкодера.

<span id="5-query-processing"></span>

## Обработка запроса

<a id="ret-query"></a>
### Сохранить условие запроса

В запросе «не использовать версию 1.2» отрицание и номер версии
задают ограничение. Удаление «не» как стоп-слова или расширение
«1.2» до «последняя версия» изменит задачу. Поэтому исходная строка,
нормализованная форма и дополнительные варианты хранятся отдельно.
Точные идентификаторы и фильтры не стоит поручать одному эмбеддингу.

Лексический канал можно расширить известными сокращениями:
например, «SSO» и «единый вход». Расширение увеличивает число
кандидатов, но неоднозначное сокращение может добавить чужую тему.
Правило должно быть привязано к домену и проверяться на случаях,
где исходная форма была правильной.

**Декомпозиция** нужна, когда вопрос содержит несколько самостоятельных
информационных потребностей. «Чем отличаются условия доступа в v1 и v2?»
можно разложить на два запроса по версиям, а затем сохранить оба
набора источников. Это отличается от **multi-query** с несколькими
перефразировками одной потребности. У каждого канала сохраняют,
какой вариант привёл к документу, и объединяют повторные ID.

<a id="ret-hyde"></a>
### HyDE: искать по гипотетическому документу

HyDE сначала просит генеративную модель написать предполагаемый
документ с ответом, затем кодирует его документным энкодером.
Полученный вектор используется для поиска **реальных** документов
корпуса. Гипотетический текст не становится источником фактов:

![[00 Учебник/Assets/Figures/curated/retrieval-review-2026-10/hyde-pipeline.png|HyDE: hypothetical documents used to retrieve real corpus documents]]

*Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan,
[Precise Zero-Shot Dense Retrieval without Relevance Labels](https://aclanthology.org/2023.acl-long.99/),
рисунок 1, PDF-страница 2. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Вырезана область рисунка с легендой; содержимое не изменено.
Жёлтые блоки — инструкции, зелёные — запросы, оранжевые —
выдуманные документы, голубые — результаты поиска в корпусе.
Примеры сгенерированных сведений на схеме не являются проверенными ответами.*

В исходной работе использовались InstructGPT и Contriever/mContriever
без дополнительного обучения на метках релевантности целевого набора.
Идея — заменить непосредственное сопоставление вопроса и документа
на генерацию образца и сравнение двух документных представлений.
Это постановка конкретной работы, не гарантия, что произвольная LLM
с любым энкодером улучшит поиск.

Для $n$ сгенерированных документов $\hat d_i$ исходный метод
рассматривает среднее представлений:

$$
\hat v_q=\frac1n\sum_{i=1}^{n}f(\hat d_i).
$$

Например, два условных вектора $(1,0)$ и $(0,1)$ дают $(0.5,0.5)$.
Скалярные произведения с документами $(1,0)$ и $(0,1)$ станут равны.
Среднее не сохраняет отдельное требование найти каждую из двух тем.
Для вопроса с несколькими обязательными частями может быть лучше
отдельно искать каждую и контролировать покрытие.

Если выдуманный текст ошибочно добавил название продукта или дату,
поиск может сместиться к неправильным документам. Плотное представление
не гарантирует удаления такой ошибки. Проверяют исходный запрос,
расширение, стоимость генерации и итоговую выдачу раздельно.
Сохранение исходного поискового канала — один из проверяемых вариантов,
а не обязательная часть оригинального алгоритма HyDE.

<a id="ret-filtering"></a>
### Фильтр до и после top-k

Пусть четыре документа имеют оценки 0.95, 0.90, 0.80, 0.70,
но пользователю разрешены только последние два.
Если сначала взять глобальный top-2, затем отфильтровать права,
результат будет пустым. Если поиск выполняется среди разрешённых,
вернутся документы с оценками 0.80 и 0.70.

Такая разница между pre-filtering и post-filtering описана, например,
в [документации Elasticsearch kNN](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-knn-query#knn-query-filtering).
Способ применения фильтра зависит от движка. Фильтрация внутри ANN
может изменить обход графа или привести к точному просмотру небольшого
разрешённого множества; её стоимость тоже нужно измерять.
Увеличение числа кандидатов перед post-filter помогает не всегда
и не заменяет корректного поиска по разрешённой области.

Права, дата действия и версия продукта проверяются явно.
Проверка доступа должна предшествовать передаче текста reader,
внешнему сервису или пользователю, в том числе при попадании в кэш.
Эмбеддинг и модель повторного ранжирования не являются механизмом авторизации.

<span id="6-гибридная-выдача"></span>

## Слияние лексической и векторной выдачи

<a id="ret-score-scale"></a>
### Почему сумма оценок зависит от масштаба

Пусть BM25 поставил документам $A,B$ оценки 20 и 10, а dense-поиск —
0.1 и 0.9. Суммы 20.1 и 10.9 сохраняют первый порядок.
Если тот же BM25-результат выразить в масштабе 0.01, суммы станут
0.3 и 1.0 — порядок поменяется, хотя внутри каждого канала он прежний.

Складывать оценки можно, но их масштаб и веса становятся частью
метода, которую нужно настраивать и проверять. Например, используют
взвешенную сумму нормированных оценок или обучают слияние по разметке.
Min–max по текущей выдаче тоже имеет ограничения: одна аномальная
оценка меняет шкалу остальных; у константного списка знаменатель
максимум минус минимум равен нулю. Нормированное число всё равно
не становится вероятностью релевантности.

<a id="ret-rrf"></a>
<span id="два-списка-и-одно-решение"></span>

### Слияние по рангам (RRF)

**Reciprocal Rank Fusion, RRF** использует позиции, а не исходные оценки:

$$
\operatorname{RRF}(d)=
\sum_{r:\,d\in r}\frac{w_r}{c+\operatorname{rank}_r(d)}.
$$

Ранг начинается с 1. Чем больше $c\ge0$, тем меньше различаются
вклады ранних и поздних позиций. В исходной
[работе Cormack, Clarke и Büttcher](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
все списки имели одинаковый вес, а $c=60$ подобрали заранее.
В показанном обобщении $w_r\ge0$ задаёт вес списка. Если документ
отсутствует в возвращённой части списка, этот канал не даёт ему баллов.

Вернёмся к [[#ret-e42-case|вопросу об Atlas v2]]. Пусть BM25 вернул
`A, B, C`: первое место заняла расшифровка точного кода E42.
Плотный поиск вернул `C, B, D`: первой оказалась инструкция смены пароля.
Это условные ранги, а не результаты запуска на реальном корпусе.
При $c=60$ и единичных весах:

| Документ | BM25 rank | Dense rank | RRF |
| --- | ---: | ---: | ---: |
| A | 1 | — | $1/61\approx0.01639$ |
| B | 2 | 2 | $2/62\approx0.03226$ |
| C | 3 | 1 | $1/63+1/61\approx0.03227$ |
| D | — | 3 | $1/63\approx0.01587$ |

Итог — `C, B, A, D`. Справка B получила вклад от обоих каналов и
обошла A, хотя только A объясняет E42. C тоже получила два вклада.
Разрыв между B и C очень мал; при равенстве баллов можно сортировать
по ID, чтобы порядок не зависел от реализации.

<details id="ret-code-rrf">
<summary>Слияние рангов</summary>

```python
from collections import defaultdict
import math

def rrf(lists, c=60., weights=None):
    weights = [1.] * len(lists) if weights is None else list(weights)
    if not math.isfinite(c) or c < 0 or len(weights) != len(lists):
        raise ValueError("invalid constant or weights")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must be finite and nonnegative")
    scores = defaultdict(float)
    for ranked, weight in zip(lists, weights):
        if len(set(ranked)) != len(ranked):
            raise ValueError("each input list must contain unique document IDs")
        if weight == 0:
            continue
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] += weight / (c + rank)
    return sorted(scores.items(), key=lambda row: (-row[1], row[0]))

combined = rrf([["A", "B", "C"], ["C", "B", "D"]])
assert [doc for doc, _ in combined] == ["C", "B", "A", "D"]
assert math.isclose(dict(combined)["C"], 1 / 63 + 1 / 61)
assert rrf([["B"], ["A"]])[0][0] == "A"  # explicit tie-break
```

</details>

RRF отбрасывает информацию о величине разрыва оценок. Документы,
которые первый канал разделял едва заметно, дают те же ранги,
что и документы с большим разрывом. Кроме того, отсутствие в коротком
списке не означает нерелевантности. Поэтому глубина каждого исходного
списка влияет на слияние.

Повторение одного канала под разными именами фактически увеличивает
его вес. Десять близких перефразировок запроса не являются десятью
независимыми свидетельствами. Число каналов, варианты запросов,
веса, $c$ и глубины списков фиксируют вместе с результатом.

<a id="ret-fusion-limits"></a>
<span id="когда-объединение-помогает-а-когда-мешает"></span>

### Полнота до и после отбора

В таблице RRF оба канала дали по три результата. В объединении
есть оба нужных документа, но если отдать дальше только первые два,
пройдут C и B. Полнота упадёт с 1 до $1/2$: модуль ответа не узнает
причину E42 из найденных страниц.

Теперь обрежем каждый канал *до* слияния. Останутся A, B у BM25 и C, B
у плотного поиска. Полнота каждого списка $1/2$, объединения — 1.
B по-прежнему получает два вклада. A и C получают одинаковые баллы
$1/61$, и по принятому правилу ID выше окажется A. Дальше уйдут B и A:
при той же полноте $1/2$ теперь теряется уже инструкция смены пароля.

В отчёте сохраняют полноту каждого канала на заданной глубине,
полноту объединения до отбора и полноту списка для модуля ответа.
Для вопроса из двух частей нужны ещё ID и покрытие сведений:
в нашем примере одна и та же итоговая метрика скрывает две разные потери.

В [эксперименте Anserini на BEIR](https://github.com/castorini/anserini/blob/07d380635f1719736c6042c6e971fd7d08ee125e/docs/experiments-beir-fusion.md)
RRF не выигрывает у лучшего одиночного канала на каждом наборе.
Например, nDCG@10 на `nq`: BM25 0.3055, BGE 0.5415, RRF 0.4831;
на `trec-covid`: 0.5947, 0.7815 и 0.8041 соответственно.
Это результаты указанных моделей, данных и глубин из источника,
не измерения нашей учебной системы. Они показывают, почему выбор
слияния нужно проверять по доменам и типам запросов.

<span id="7-обновления-и-версии-индекса"></span>

## Обновления и версии индекса

<a id="ret-version"></a>
### Что именно версионировать

Индекс зависит от снимка корпуса, парсера, разбиения, токенизатора
и модели, pooling, инструкций, обрезки, нормировки и метрики.
У IVF/PQ добавляются обученные центры и кодовые книги; у BM25 —
статистики коллекции. Один тег модели не описывает весь индекс.

Если изменился документ, надо найти все его старые фрагменты,
удалить или заменить их и обновить связанные записи.
Идентификатор, содержащий версию источника, помогает отличить
старый chunk от нового с тем же порядковым номером.
При добавлении документов меняются и статистики IDF/средней длины,
так что оценки BM25 могут измениться даже для неизменённого текста.

Изменение только правил обработки запроса при неизменной документной
ветви обычно не требует пересчитывать документы, но требует
проверки качества. Смена документного энкодера, pooling или инструкции
требует пересчёта затронутых представлений. Совпадение размерности
старого и нового векторов не означает совместимости пространств.

<a id="ret-cutover"></a>
### Переход на новую версию

Новый индекс можно построить рядом со старым. Проверяют наличие
документов, согласованность ID и метаданных, качество на отложенных
запросах, задержку и применение фильтров. Затем на копии запросов
сравнивают выдачу без изменения пользовательского результата.
Такой shadow-прогон должен соблюдать те же ограничения на данные,
что и основной поиск.

Переключение указателя на индекс не должно смешать в одном запросе
новый энкодер со старой базой. Удобно выбирать согласованный набор
`encoder configuration + index version + metadata snapshot`
один раз в начале запроса. Кэш ключуется этой версией, а для отката
сохраняют прежний согласованный набор. Атомарная замена alias сама
по себе не обновляет старые кэши и уже выполняющиеся запросы.

Для отзыва доступа одной фоновой переиндексации недостаточно.
До завершения удаления копий авторитетная проверка прав должна
не допустить выдачу отозванного текста. Отдельно измеряют срок
очистки индекса, родительских фрагментов, кэшей и других производных
копий. Техническое наличие старого вектора и право вернуть его
пользователю — разные состояния.

<span id="8-диагностика-retrieval"></span>

## Где потерялся документ

<a id="ret-trace"></a>
### Проследить один потерянный ответ

В [[#ret-e42-case|примере с Atlas]] возьмём слияние двух списков
по три результата. Начнём с A. Исходник есть в базе,
парсер сохранил условие доступа, BM25 поставил A первой, и она
вошла в объединённый список. После RRF она третья, а модуль чтения
получает первые две позиции. Значит, A потеряна при отборе после
слияния. Проверять стоит вклады каналов и лимит списка. Увеличение
`efSearch` может изменить входные ранги, но отсутствие A в ответе
само по себе ничего не говорит об ошибке графа.

В другом запросе точный поиск по векторам может поставить источник
на 12-е место, а ANN не вернуть его даже в первых 100. Тогда сравнивают
индексы при одной метрике и одних сохранённых векторах. Если же условие
доступа исчезло уже при извлечении текста PDF, никакая настройка поиска
не восстановит отсутствующий текст.

Для такого разбора сохраняют трассу, не только финальные ID:

| Этап | Что сохранять |
| --- | --- |
| Источник | ID, версию, исходный достаточный участок |
| Подготовка | Результат парсера, границы и метаданные chunks |
| Запрос | Исходную форму, варианты и явные фильтры |
| Каналы | Оценки и ранги до слияния, параметры ANN |
| Отбор | Причину фильтрации, вклад каждого канала, место обрезки |
| Результат | Версию индекса, итоговый список и переданный контекст |

Трасса тоже может содержать закрытые данные. Полные тексты запросов
и документов нельзя автоматически отправлять во внешний журнал
только потому, что это отладка. Объём, доступ и срок хранения
диагностических данных определяют отдельно.

<a id="ret-evaluation"></a>
### Разделить проверки

Сначала на фиксированном корпусе сравнивают лексический и точный
плотный поиск с одной разметкой. Затем добавляют ANN и измеряют
потерю относительно точного поиска. После этого проверяют слияние,
фильтры и повторное ранжирование. Если одновременно сменить модель,
разбиение и тип индекса, по одной итоговой метрике нельзя установить,
какое изменение дало эффект.

Оценку по qrels дополняют разбором точных идентификаторов,
перефразировок, отрицаний, длинных запросов, редких сущностей,
неотвечаемых запросов и смены версии документа.
Неполную разметку отмечают явно: новый хороший результат может
отсутствовать в прежнем наборе известных ответов.

Скорость проверяют вместе с размером индекса, временем построения,
числом запросов в пакете, параллелизмом, фильтрами и целевой полнотой.
Flat на GPU для большого пакета и графовый поиск на CPU для
одиночного запроса решают разные вычислительные задачи.
Число сравнений из учебного примера не является измерением задержки.

## Источники и дальнейшее чтение

- [Jurafsky & Martin, SLP3, chapter 11](https://web.stanford.edu/~jurafsky/slp3/11.pdf) —
  §11.1: разреженные представления, TF–IDF, BM25 и inverted index;
  §11.3: плотный поиск. Ссылка ведёт на обновляемый черновик;
  в этой редакции проверен PDF от 19 августа 2026 года.
- [ACL 2020 OpenQA Tutorial](https://github.com/danqi/acl2020-openqa-tutorial) —
  история retriever–reader и исходный TF–IDF DrQA.
- [The Faiss Library](https://arxiv.org/abs/2401.08281) —
  §3: метрики и стоимость; §4: сжатие; §5: IVF и графовые индексы.
- [Milvus: IVF_FLAT](https://milvus.io/docs/ivf-flat.md),
  [HNSW](https://milvus.io/docs/hnsw.md),
  [IVF_PQ](https://milvus.io/docs/ivf-pq.md) —
  использованные оригинальные пошаговые схемы.
- [Yusuke Matsui: nanopq tutorial](https://nanopq.readthedocs.io/en/latest/source/tutorial.html) —
  английское объяснение кодовых книг, таблиц ADC, восстановления и OPQ
  с исполняемым примером библиотеки.
- [HyDE, ACL 2023](https://aclanthology.org/2023.acl-long.99/) —
  §3.2 и рисунок 1; гипотетический документ служит для поиска,
  не заменяет источник ответа.
- [Anserini: Fusion on BEIR](https://github.com/castorini/anserini/blob/07d380635f1719736c6042c6e971fd7d08ee125e/docs/experiments-beir-fusion.md) —
  воспроизводимые конфигурации и отдельные результаты методов слияния.
- [Hugging Face Advanced RAG](https://huggingface.co/learn/cookbook/en/advanced_rag) —
  §1.1: структура и токенная длина фрагментов; весь конвейер рассматривается
  дальше, в главе RAG.
- [BEIR](https://arxiv.org/abs/2104.08663) —
  оценка переноса поиска между разными наборами и доменами.

← [[60 Embeddings и metric learning|Эмбеддинги и метрическое обучение]]

Дальше — [[62 Reranking — cross-encoder и late interaction|повторное ранжирование]].
Оно сможет сопоставить A с запросом внимательнее, если получит A
вместе с B и C, до того как список обрежут до двух документов.
