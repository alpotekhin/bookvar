---
title: BERT, RoBERTa и DeBERTa
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://arxiv.org/abs/1810.04805
  - https://arxiv.org/abs/1907.11692
  - https://arxiv.org/abs/2006.03654
---

# BERT, RoBERTa и DeBERTa

Значение слова часто раскрывается только после него. В предложении «он подошёл
к банку реки» слово «банку» невозможно уверенно интерпретировать, прочитав лишь
левый контекст. BERT строит представление каждого токена по всему предложению:
в его encoder нет причинной маски, поэтому внимание направлено и влево, и
вправо.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-feature-extraction-contextualized-embeddings.png]]

*Одинаковое слово получает разные контекстные представления в разных фразах.
Иллюстрация: Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Полная видимость мешает обучать BERT обычным предсказанием следующего токена:
модель увидела бы правильный ответ справа. Поэтому часть входа скрывают и просят
восстановить по оставшемуся контексту. Для множества выбранных позиций $M$
оптимизируется

$$\mathcal L_{MLM}=-\sum_{i\in M}\log p(x_i\mid x_{\setminus M}).$$

## Как устроен обучающий пример

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/BERT-language-modeling-masked-lm.png]]

*Masked language modeling в изложении Jay Alammar: двунаправленный encoder
использует обе стороны контекста для восстановления скрытого токена. Источник:
[The Illustrated BERT](https://jalammar.github.io/illustrated-bert/), автор Jay
Alammar, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

BERT обучался на BooksCorpus и английской Wikipedia. Из выбранных для предсказания токенов 80% заменялись `[MASK]`, 10% — случайным токеном, 10% оставались прежними. Такая смесь уменьшает разрыв между предобучением и применением, но не устраняет его. Вторая исходная цель, Next Sentence Prediction, различала настоящую соседнюю пару сегментов и случайную. Именно здесь полезно читать рисунок выше не как «модель угадывает пропуск», а как карту потока: скрывается входной токен, весь encoder строит контекст, и только затем MLM head сравнивается с исходным id.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-input-output.png]]

*Вход BERT складывается из токенных, позиционных и сегментных векторов; верхний
слой возвращает контекстное состояние для каждой позиции. Иллюстрация: Jay
Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Последовательность начинается с `[CLS]`; пары текстов разделяются `[SEP]`.
К токенному и позиционному векторам прибавляется segment embedding, отмечающий,
из какого фрагмента пришёл токен. Вход `[B,T]` превращается в `[B,T,d]`, а
attention scores имеют `[B,h,T,T]`. Padding закрывается отдельной маской.

## Как одно предобучение приспосабливается к разным задачам

Fine-tuning не меняет основной стек. Для token classification линейная голова применяется к каждой строке `[B,T,d]`; для классификации пары — к состоянию `[CLS]`; в extractive QA две проекции дают logits начала и конца `[B,T]`. Все параметры обычно обновляются совместно. В отличие от feature extraction, это не использование замороженных embeddings.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/bert/bert-tasks.png]]

*Один encoder и четыре способа прочитать его выход: классификация пары,
классификация одного текста, ответы на вопросы и разметка токенов. Иллюстрация:
Jay Alammar, [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

`[CLS]` здесь не «смысл предложения по определению». Это специальная позиция,
состояние которой обучается быть полезным для выбранной головы. Без
дополнительного contrastive или supervised обучения оно не обязано быть хорошим
sentence embedding.

## RoBERTa: та же архитектура, другой рецепт

BERT показал, что один и тот же двунаправленный encoder после короткого
fine-tuning даёт сильные результаты на GLUE, MultiNLI и SQuAD. RoBERTa затем
повторила постановку при более тщательном обучении: больше данных и обновлений,
большие batch, более длинные последовательности, динамически выбранные маски и
без Next Sentence Prediction. Качество выросло без новой формулы attention.
Сравнение показывает, почему архитектуру нельзя оценивать отдельно от данных и
режима обучения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/roberta-dynamic-masking-table1.png]]

*Статическое и динамическое маскирование в контролируемом сравнении. Источник:
Liu et al., [RoBERTa, Table 1](https://arxiv.org/abs/1907.11692), 2019.*

В статическом варианте авторы заранее создали десять масок каждого примера; в
динамическом маска выбиралась заново при каждой подаче. Архитектура BERT при
этом не менялась. Следовательно, результат относится к режиму обучения, а не к
новому механизму внимания.

## DeBERTa: содержание и положение разделены

DeBERTa меняет сам attention. Для позиции $i$ хранится вектор содержания, а для
смещения $i-j$ — отдельный относительный позиционный вектор. До softmax оценка
пары имеет три слагаемых:

$$
\tilde A_{ij}=Q_i^cK_j^{c\top}+Q_i^cK_{\delta(i,j)}^{r\top}
+K_j^cQ_{\delta(j,i)}^{r\top}.
$$

Первое сравнивает содержание токенов, второе — содержание запроса с положением
ключа, третье — содержание ключа с положением запроса. Нормирующий множитель
равен $\sqrt{3d}$, поскольку складываются три скалярных произведения.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-audit-25-31/deberta-disentangled-attention-equation4.png]]

*Disentangled attention в записи авторов. Источник: He et al.,
[DeBERTa, Eq. 4](https://arxiv.org/abs/2006.03654), 2020.*

Абсолютная позиция возвращается ближе к MLM-голове в enhanced mask decoder:
одних относительных отношений недостаточно, чтобы различить некоторые варианты
на конкретном месте. Это архитектурное изменение, в отличие от
оптимизационного рецепта RoBERTa.

## Ограничения и наследие

Полная видимость делает encoder естественным для анализа известного текста, но не для свободной генерации. MLM предсказывает несколько скрытых мест условно, а не задаёт нормализованную вероятность всей строки слева направо. Квадратичная память ограничивает длину. Наконец, `[CLS]` обучался под конкретные objectives и не обязан быть хорошим семантическим pooling без дополнительного обучения.

Наследие BERT — не одна модель, а шаблон «общее двунаправленное предобучение → небольшая task head». RoBERTa научила отделять recipe от architecture; DeBERTa — явно разделять content и position. При сравнении результатов нужно фиксировать корпус, tokenizer, число обновлений и fine-tuning protocol.


## Источники

- [Devlin et al., BERT](https://arxiv.org/abs/1810.04805)
- [Liu et al., RoBERTa](https://arxiv.org/abs/1907.11692)
- [He et al., DeBERTa](https://arxiv.org/abs/2006.03654)
- [Stanford CS224N course: pretraining readings](https://web.stanford.edu/class/cs224n/)
- [Jurafsky & Martin, SLP3, Chapter 10: Masked Language Models](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — последовательное объяснение двунаправленного encoder, MLM и fine-tuning
- [Jay Alammar, The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — визуальная опора для MLM и переноса encoder representations
