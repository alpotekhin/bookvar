---
title: "Переранжирование: cross-encoder и late interaction"
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://www.sbert.net/examples/cross_encoder/applications/README.html
  - https://arxiv.org/abs/2004.12832
  - https://web.stanford.edu/~jurafsky/slp3/11.pdf
---

# Переранжирование: cross-encoder и late interaction

Первый этап поиска сокращает миллион документов до сотен кандидатов, но независимые
векторы запроса и документа неизбежно теряют тонкие связи. Переранжировщик
получает уже небольшой список и может потратить больше вычислений на каждую
пару. Его задача — не повторить поиск, а восстановить точный порядок среди
похожих кандидатов.

## 1. Bi-encoder и cross-encoder решают разные вычислительные задачи

Bi-encoder вычисляет $e_q(q)$ и $e_d(d)$ независимо. Это позволяет заранее
индексировать документы, но взаимодействие сводится к одному скаляру.
Cross-encoder получает последовательность `[CLS] query [SEP] document [SEP]`.
Self-attention сопоставляет каждый токен запроса с токенами документа на всех
слоях, а классификационная голова выдаёт оценку релевантности:

$$
s(q,d)=w^\top h_{\mathrm{CLS}}(q,d)+b.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/sbert-bi-vs-cross-encoder.png]]

*Оригинальная схема из [Sentence Transformers, Cross-Encoder applications](https://www.sbert.net/examples/cross_encoder/applications/README.html), UKP Lab / Hugging Face. Слева предложения проходят через BERT раздельно и сравниваются после pooling; справа токены обеих сторон входят в один BERT и classifier. Поэтому только левую ветвь можно предварительно индексировать.*

В документации [Sentence Transformers](https://www.sbert.net/examples/cross_encoder/applications/README.html)
это различие объяснено через практическое ограничение: cross-encoder обычно
точнее, но не создаёт самостоятельного document embedding и должен заново
обрабатывать каждую пару. Поэтому схема retrieve $k$ → rerank $k$ является не
случайной инженерной привычкой, а следствием архитектуры.

## 2. Обучающие примеры должны приходить из настоящего retriever

Случайный отрицательный документ слишком прост: модель различит тему, но не
научится выбирать между двумя правдоподобными passages. Нужны кандидаты, которые
первый этап действительно ставит высоко. Для запроса сохраняют положительные
документы, hard negatives из BM25/dense/hybrid выдачи и, если возможно,
градуированную релевантность.

Pointwise обучение предсказывает метку или score для каждой пары. Pairwise
обучение требует $s(q,d^+)>s(q,d^-)$, например через

$$
\mathcal L=-\log\sigma(s(q,d^+)-s(q,d^-)).
$$

Listwise objective работает сразу с набором кандидатов и ближе к nDCG, но
дороже и чувствительнее к неполной разметке. Если неразмеченный документ
действительно отвечает на вопрос, обучение как на negative ухудшает модель.

## 3. Калибровка не равна ранжированию

Высокая оценка cross-encoder не обязательно является вероятностью
релевантности. Для порядка достаточно монотонности внутри запроса, но порог
«ответить или отказаться» требует отдельной калибровки на целевом распределении.
Сдвиг домена, длина документа и изменение candidate generator меняют
распределение score. Порог после замены retriever необходимо переоценить.

## 4. Late interaction: промежуточная точка ColBERT

Cross-encoder выполняет максимум взаимодействия и минимум переиспользования.
Один вектор на документ — наоборот. ColBERT сохраняет отдельный embedding для
каждого токена документа. Для каждого токена запроса выбирается наиболее
похожий токен документа, затем максимумы суммируются:

$$
s(q,d)=\sum_{i\in q}\max_{j\in d} E_{q_i}^\top E_{d_j}.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-60-64/colbert-late-interaction.png]]

*Figure 2 из [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832),
Khattab & Zaharia, SIGIR 2020. Первые три панели показывают цену перехода от
независимых представлений к полному взаимодействию query и document. ColBERT в
четвёртой панели оставляет кодирование раздельным, но откладывает сравнение до
уровня токенов: синие векторы документов вычисляются заранее, а MaxSim не
требует совместного Transformer для каждой пары.*

Рисунок важен именно расположением границы: contextualization происходит
раздельно, но взаимодействие отложено до токенных векторов, а не до одного
pooled embedding. Это улучшает точные и семантические совпадения, но индекс
становится значительно больше. ColBERT можно использовать и как полный
retriever, и как reranker кандидатов.

MaxSim объясним: для каждого query token можно показать победивший document
token. Но это не причинное объяснение решения модели — contextualized vector
уже содержит информацию других токенов, а сумма скрывает отрицательные и
композиционные взаимодействия.

## 5. Cascade и бюджет вычислений

Практический каскад может состоять из BM25 и dense retrieval, дешёвого
late-interaction или small cross-encoder, затем дорогого cross-encoder для
верхних кандидатов. Выбор $k$ определяется кривой:

- candidate recall растёт с $k$;
- стоимость reranker примерно линейна по числу пар и длине;
- качество перестаёт расти, когда добавляются только нерелевантные кандидаты;
- при слишком малом $k$ reranker не может вернуть потерянное доказательство.

Batching повышает пропускную способность, но p95 зависит от длины самого
длинного элемента пакета. Bucketing по длине и ограничение passage size могут
дать больший выигрыш, чем квантование модели.

## 6. Метрики и ablation

Переранжировщик оценивают только при фиксированном наборе кандидатов. Иначе улучшение
retriever ошибочно приписывается reranker. Сравнивают MRR, nDCG@k, precision@k
и долю запросов, где релевантный документ был в кандидатах, но выпал из
финального top-k.

Минимальный отчёт содержит четыре строки: retriever без reranking; oracle,
поднимающий любой размеченный relevant candidate наверх; новый reranker;
полный end-to-end ответ. Разрыв до oracle показывает потенциал reranking, а
разрыв между oracle и candidate recall — потолок, который исправляется только
retrieval.

## 7. Типичные ошибки

- **Position bias:** модель привыкает к месту документа в исходной выдаче.
  Кандидаты перемешивают при обучении и проверяют перестановками.
- **Length bias:** длинный passage получает больше совпадений. Нужны length
  slices и одинаковые правила обрезки.
- **Shortcut по источнику:** домен или шаблон документа заменяет содержание.
- **Temporal leakage:** обучающие clicks или labels появились после тестовой
  даты.
- **Answer-string bias:** наличие строки ответа не означает, что passage
  подтверждает утверждение.
- **Lost recall:** хороший reranker обвиняют в ошибке, хотя нужного документа не
  было среди кандидатов.

## Источники и дальнейшее чтение

- [Sentence Transformers: Cross-Encoder applications](https://www.sbert.net/examples/cross_encoder/applications/README.html) —
  наглядное сравнение bi- и cross-encoder и retrieve-and-rerank pipeline.
- [ColBERT](https://arxiv.org/abs/2004.12832) и
  [официальный репозиторий](https://github.com/stanford-futuredata/ColBERT) —
  late interaction и MaxSim.
- [Jurafsky & Martin, SLP3 chapter 11](https://web.stanford.edu/~jurafsky/slp3/11.pdf) —
  ranking, neural retrieval и метрики.
