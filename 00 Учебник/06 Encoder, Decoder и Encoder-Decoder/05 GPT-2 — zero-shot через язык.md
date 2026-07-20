---
title: GPT-2 — zero-shot через язык
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources: [https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf]
---

# GPT-2: zero-shot через язык

GPT-2 задаёт следующий вопрос: если задачи уже встречаются в естественном тексте, можно ли выполнять их без градиентной адаптации, задав контекст словами? Архитектурно это увеличенный causal Transformer, а концептуально — переход от fine-tuning как обязательного этапа к conditioning.

Модель обучается той же целью

$$\mathcal L=-\sum_t\log p(x_t\mid x_{<t}),$$

но на WebText и byte-level BPE. Семейство выросло до 1.5B параметров. LayerNorm была перенесена ближе к pre-norm-схеме, дополнительная нормализация поставлена после последнего блока, residual weights масштабировались при инициализации.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-autoregression-2.gif]]

*Пошаговая авторегрессия GPT-2: каждый сгенерированный токен становится частью
следующего префикса. Источник: [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
автор Jay Alammar, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

На анимации нет переключателя задач. На каждом шаге модель выполняет одну и ту
же операцию — продолжает префикс. Перевод, краткое изложение или ответ возникают
тогда, когда начало текста похоже на пример соответствующей задачи.

Например, формат `English text = French text` превращает перевод в продолжение последовательности. Это не отдельный переводчик: задача задаётся наблюдаемым текстовым паттерном.

Zero-shot результаты были неоднородны, а sampling чувствителен к temperature и truncation. Документы WebText несут смещения источников; memorization и правдоподобное продолжение нельзя путать с проверенным знанием. GPT-2 показал направление, но ещё не дал устойчивого few-shot интерфейса.

## Данные и масштаб как часть гипотезы

WebText собирался из страниц, на которые ссылались посты Reddit с достаточным числом голосов; Wikipedia исключили, чтобы упростить проверку benchmark contamination. Такая эвристика повышает долю связного текста, но переносит предпочтения англоязычного интернет-сообщества. Семейство 117M, 345M, 762M и 1.5B позволяло проверить, возникает ли zero-shot behavior плавно с ёмкостью.

Byte-level BPE избегает отдельного unknown token: любой текст представим байтами, а частые последовательности сливаются. Цена — длина для редких письменностей и зависимость разбиения от пробелов. Tokenizer поэтому является частью вычислительного бюджета, а не нейтральной предобработкой.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-sizes-hyperparameters-3.png]]

*Четыре размера GPT-2 и параметры их Transformer-блоков. Иллюстрация: Jay
Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Семейство из нескольких размеров было частью эксперимента: авторы проверяли, как
качество продолжения и перенос без fine-tuning меняются с ёмкостью. Нельзя
читать результат только по самой большой модели — именно ряд контрольных точек
показывает направление изменения.

## Что находится внутри одного шага генерации

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png]]

*Поток одного токенного вектора через блок GPT-2: masked self-attention,
остаточные связи и FFN. Источник: Jay Alammar,
[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/),
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).*

Анимация выше показывает центральный механизм лучше статической блок-схемы. После каждого шага распределение берётся из последней строки logits; выбранный id добавляется к входу; прежние K/V могут кэшироваться. «Zero-shot перевод» означает, что префикс формирует распределение следующего текста, а не включает скрытую translation head.

## Ключевые проверки

Авторы оценивали language modeling без task-specific training и перенос на reading comprehension, summarization, translation и question answering. Улучшение с размером было заметно, но лучшие supervised systems часто оставались впереди. Особенно важно различать objective и metric: likelihood обучает все токены корпуса, тогда как BLEU или F1 измеряют узкую форму ответа.

Paper также исследовал memorization через совпадения с training data. Это раннее предупреждение: впечатляющее продолжение может быть восстановлением увиденного фрагмента. Нельзя интерпретировать sample только по плавности языка.

## Ограничения и наследие

Задача задаётся неявным текстовым паттерном, поэтому небольшая смена формулировки меняет результат. Авторегрессия накапливает ошибки, а likelihood не запрещает токсичные или ложные продолжения. WebText не репрезентативен для мира и языков. Большая модель дороже на каждом токене и хранит больше редких фрагментов.

GPT-2 сделал prompting наблюдаемым интерфейсом, выпустил код и staged checkpoints и сформировал практику анализа generated samples. Но in-context learning с несколькими демонстрациями станет систематическим объектом только в GPT-3.

## Источники

- [Radford et al., Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenAI GPT-2 code and model card](https://github.com/openai/gpt-2)
