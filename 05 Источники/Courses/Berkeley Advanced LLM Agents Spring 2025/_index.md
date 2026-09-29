---
title: Berkeley Advanced LLM Agents — Spring 2025
type: source-note
status: verified
last_verified: 2026-09-15
---

# Berkeley Advanced LLM Agents — Spring 2025

[Официальная страница курса](https://rdi.berkeley.edu/adv-llm-agents/sp25) ·
[[05 Источники/Курсы|Все курсы и тематический маршрут]]

Курс связывает обучение моделей рассуждения с памятью и планированием агентов,
работой в браузере, программированием, формальными доказательствами и безопасностью.
Ниже можно выбрать главу учебника по теме или открыть оригинальные английские
слайды и список статей соответствующей лекции.

## Где читать тему

| Тема | В учебнике | Оригинальные лекции |
|---|---|---|
| Поиск решения, проверка и дополнительные вычисления при ответе | [[00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Test-time compute]], [[00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR и проверяющие программы]] | [1–2: reasoning](#meeting-01), [4: post-training](#meeting-04) |
| Память, планирование и модели мира | [[00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Память и планирование]] | [3: memory and planning](#meeting-03) |
| Программирование, браузер и графический интерфейс | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Coding, web и computer-use agents]] | [5: coding](#meeting-05), [6: web](#meeting-06), [7: GUI](#meeting-07) |
| Формализация и доказательство теорем | [[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Математические агенты]] | [8: AlphaProof](#meeting-08), [9: autoformalization](#meeting-09), [10: theorem proving](#meeting-10) |
| Научный поиск и построение абстракций | [[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Агенты научного поиска]] | [11: abstraction and discovery](#meeting-11) |
| Угрозы, недоверенные данные и ограничение полномочий | [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Безопасность агентов]] | [12: safe and secure agentic AI](#meeting-12) |

<a id="встречи-и-будущие-назначения"></a>
## Лекции, слайды и статьи

<a id="meeting-01"></a>
### 1. 27 января — inference-time reasoning

Xinyun Chen разбирает, как получить более надёжное решение, меняя способ
генерации и проверки ответа: разбить задачу на шаги, построить несколько
кандидатов, организовать поиск и проверить результат. Отдельная проблема —
самокоррекция: повторная просьба «проверь себя» без внешней проверки не
гарантирует исправления ошибки.

[Слайды, 74 страницы](Lectures/meeting-01-slides.pdf) ·
[вводная часть, 16 страниц](Lectures/meeting-01-intro.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=g0Dwtf3BH-0)

Статьи к лекции:

1. [Large Language Models as Optimizers](Readings/meeting-01-reading-01.md)
2. [Large Language Models Cannot Self-Correct Reasoning Yet](Readings/meeting-01-reading-02.md)
3. [Teaching Large Language Models to Self-Debug](Readings/meeting-01-reading-03.md)

<a id="meeting-02"></a>
### 2. 3 февраля — learning to reason

Jason Weston обсуждает обучение по предпочтениям и способы, которыми модель
может участвовать в оценке собственных ответов. DPO, Self-Rewarding и IRPO
показывают разные варианты цикла «сгенерировать решения — сравнить —
дообучить»; качество сравнения ограничивает качество следующего шага обучения.

[Слайды, 106 страниц](Lectures/meeting-02-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=_MNlLhU33H0)

Статьи к лекции:

1. [Direct Preference Optimization](Readings/meeting-02-reading-01.md)
2. [Iterative Reasoning Preference Optimization](Readings/meeting-02-reading-02.md)
3. [Chain-of-Verification](Readings/meeting-02-reading-03.md)

<a id="meeting-03"></a>
### 3. 10 февраля — reasoning, memory, and planning

Yu Su связывает рассуждение с устройством всей агентной системы. На примере
HippoRAG рассматривается поиск по связанной памяти, а модели мира нужны для
предсказания последствий действия до его выполнения. Это разные задачи:
вспомнить подходящий факт ещё не означает выбрать удачный план.

[Слайды, 80 страниц](Lectures/meeting-03-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=zvI4UN2_i-w)

Статьи к лекции:

1. [Grokked Transformers are Implicit Reasoners](Readings/meeting-03-reading-01.md)
2. [HippoRAG](Readings/meeting-03-reading-02.md)
3. [Is Your LLM Secretly a World Model of the Internet?](Readings/meeting-03-reading-03.md)

<a id="meeting-04"></a>
### 4. 24 февраля — open post-training recipes

Hanna Hajishirzi показывает последовательность дообучения Tülu: подбор
примеров для SFT, обучение по предпочтениям и RLVR. Разбор связывает методы
с данными и оцениванием, поэтому полезен не только как описание алгоритмов,
но и как пример воспроизводимой работы над моделью.

[Слайды, 155 страниц](Lectures/meeting-04-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=cMiu3A7YBks)

Статьи к лекции:

1. [Tulu 3](Readings/meeting-04-reading-01.md)
2. [Unpacking DPO and PPO](Readings/meeting-04-reading-02.md)
3. [OpenScholar](Readings/meeting-04-reading-03.md)

<a id="meeting-05"></a>
### 5. 3 марта — coding agents and vulnerability detection

Charles Sutton разбирает поиск уязвимостей с помощью агента, который читает
код, запускает инструменты и проверяет гипотезы. Разница между правдоподобным
сообщением об ошибке и найденной уязвимостью проявляется в воспроизводящем
примере и результатах проверки.

[Слайды, 55 страниц](Lectures/meeting-05-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=JCk6qJtaCSU)

Статьи к лекции:

1. [EnIGMA](Readings/meeting-05-reading-01.md)
2. [From Naptime to Big Sleep](Readings/meeting-05-reading-02.md)

<a id="meeting-06"></a>
### 6. 10 марта — multimodal web agents

Ruslan Salakhutdinov рассматривает агента, действующего в браузере: он должен
понять страницу, выбрать действие и проверить, приблизило ли оно его к цели.
Mind2Web, WebArena и VisualWebArena позволяют сравнить разные условия такой
работы; поиск по дереву действий помогает исследовать несколько вариантов
продолжения.

[Видеозапись](https://www.youtube.com/watch?v=RPINOYM12RU) ·
[официальная страница со слайдами](https://rdi.berkeley.edu/adv-llm-agents/sp25)

> [!caution]
> Страница 117 исходного PDF помечена **“NVIDIA CONFIDENTIAL. DO NOT DISTRIBUTE.”**
> Она исключена из переиспользования. Здесь намеренно нет ссылки на локальную
> копию полного PDF, содержащую эту страницу.

Статьи к лекции:

1. [Mind2Web](Readings/meeting-06-reading-01.md)
2. [WebArena](Readings/meeting-06-reading-02.md)
3. [VisualWebArena](Readings/meeting-06-reading-03.md)
4. [Tree Search for Language Model Agents](Readings/meeting-06-reading-04.md)

<a id="meeting-07"></a>
### 7. 17 марта — GUI agents from perception to action

Caiming Xiong разбирает переход от изображения интерфейса к действию:
найти нужный элемент, определить координаты и выполнить команду. OSWorld
задаёт среду для проверки компьютерных агентов; TACO и Aguvis показывают,
как собирать траектории и обучать взаимодействию с интерфейсом.

[Слайды, 106 страниц](Lectures/meeting-07-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=n__Tim8K2IY)

Статьи к лекции:

1. [OSWorld](Readings/meeting-07-reading-01.md)
2. [Aguvis](Readings/meeting-07-reading-02.md)

<a id="meeting-08"></a>
### 8. 31 марта — AlphaProof

Thomas Hubert объясняет, как формальный проверяющий инструмент меняет задачу
математического рассуждения. AlphaProof строит доказательства, которые можно
проверить в Lean; поиск и обучение с подкреплением помогают находить шаги,
ведущие к завершённому доказательству. Результаты на задачах IMO обсуждаются
вместе с условиями, в которых они получены.

[Слайды, 112 страниц](Lectures/meeting-08-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=3gaEMscOMAU)

Статьи к лекции:

1. [AI achieves silver-medal standard at the IMO](Readings/meeting-08-reading-01.md)
2. [Mastering Chess and Shogi by Self-Play](Readings/meeting-08-reading-02.md)
3. [The Future of Mathematics?](Readings/meeting-08-reading-03.md)
4. [Building the Mathematical Library of the Future](Readings/meeting-08-reading-04.md)

<a id="meeting-09"></a>
### 9. 7 апреля — autoformalization and theorem proving

Kaiyu Yang разделяет две задачи: перевести математическое утверждение в
формальный язык и найти его доказательство. LeanDojo связывает языковую
модель с состоянием доказательства и библиотекой теорем. Примеры
автоформализации показывают, почему проверка синтаксиса ещё не гарантирует,
что формальная запись выражает исходное условие.

[Слайды, 114 страниц](Lectures/meeting-09-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=cLhWEyMQ4mQ)

Статьи к лекции:

1. [LeanDojo](Readings/meeting-09-reading-01.md)
2. [Autoformalization with Large Language Models](Readings/meeting-09-reading-02.md)
3. [Autoformalizing Euclidean Geometry](Readings/meeting-09-reading-03.md)

<a id="meeting-10"></a>
### 10. 14 апреля — advanced theorem proving

Sean Welleck сопоставляет способы построения формального доказательства:
черновое рассуждение, план доказательства, генерацию проверяемых шагов и
использование инструментов поиска. Draft-Sketch-Prove, Lean-STaR и miniCTX
помогают различить вклад метода обучения, доступного контекста и проверяющей
системы.

[Слайды, 118 страниц](Lectures/meeting-10-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=Gy5Nm17l9oo)

Статьи к лекции:

1. [Draft, Sketch, and Prove](Readings/meeting-10-reading-01.md)
2. [miniCTX](Readings/meeting-10-reading-02.md)
3. [Lean-STaR](Readings/meeting-10-reading-03.md)
4. [ImProver](Readings/meeting-10-reading-04.md)

<a id="meeting-11"></a>
### 11. 21 апреля — abstraction and discovery

Swarat Chaudhuri рассматривает поиск, в котором агент не только предлагает
решения, но и накапливает пригодные для повторного использования понятия.
В символической регрессии это выражения и операции, из которых строятся
новые формулы; LaSR показывает связь между языковой моделью, поиском и
проверкой кандидатов на данных.

[Слайды, 93 страницы](Lectures/meeting-11-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=IHc0TEMrEdY)

Статьи к лекции:

1. [An In-Context Learning Agent for Formal Theorem-Proving](Readings/meeting-11-reading-01.md)
2. [Symbolic Regression with a Learned Concept Library](Readings/meeting-11-reading-02.md)

<a id="meeting-12"></a>
### 12. 28 апреля — safe and secure agentic AI

Dawn Song разбирает угрозы, возникающие, когда модель получает инструменты и
читает недоверенные данные. Рассматриваются внедрение инструкций, отравление
памяти и ограничение полномочий: даже ошибочное решение агента не должно
автоматически давать ему доступ ко всем данным и действиям.

[Слайды, 99 страниц](Lectures/meeting-12-slides.pdf) ·
[видеозапись](https://www.youtube.com/watch?v=ti6yPE2VPZc)

Статьи к лекции:

1. [Privtrans](Readings/meeting-12-reading-01.md)
2. [DataSentinel](Readings/meeting-12-reading-02.md)
3. [AgentPoison](Readings/meeting-12-reading-03.md)
4. [Progent](Readings/meeting-12-reading-04.md)

## Практика по темам курса

В Bookvar есть упражнения по
[[06 Практика/25 Воспроизводимо оценить и red-team компьютерного агента|оцениванию компьютерного агента]],
[[06 Практика/26 Поиск доказательства в Lean с verifier|поиску доказательства в Lean]]
и [[06 Практика/27 Проверяемый научный поиск на символьной регрессии|символьной регрессии]].
Это наши адаптации, **не официальные задания Berkeley**. Официальное
расписание упоминает лабораторную работу и проект, но проверяемого оригинала
лабораторного задания в сохранённой подборке нет.

<details>
<summary>Для редакторов: состав архива, проверка переноса и ограничения</summary>

### Состав и проверка

Сохранены 12 встреч Spring 2025: 13 оригинальных PDF, сведения о 12
видеозаписях и 37 записей о статьях. У первой встречи отдельно сохранены
введение и основная лекция. Ссылки на статьи ведут к первоисточникам; полный
текст статей здесь не воспроизводится.

Реестры [coverage.yml](coverage.yml) и [visuals.yml](visuals.yml) показывают,
какие фрагменты связаны с главами, а какие доступны только в оригинале.
Количество записей не равно количеству готовых объяснений или отдельных
картинок: запись может объединять последовательность слайдов.

- [source-manifest.yml](source-manifest.yml) — материалы и их происхождение;
- [source-units.yml](source-units.yml) — смысловые разделы лекций;
- [editorial-map.yml](editorial-map.yml) — решения о переносе в главы;
- [semantic-review.json](semantic-review.json) — постраничная разметка;
- [audit-contract.json](audit-contract.json) — независимая проверка полноты;
- [snapshot-lock.json](snapshot-lock.json) — контрольные суммы и версии.

Команда `python3 publishing/tools/import_berkeley_agents.py --check`
проверяет архив и соответствие реестров. Контракт аудита не перезаписывается
автоматически: изменение состава или разметки требует отдельной проверки
затронутых страниц. Само наличие связанного раздела ещё не подтверждает
полноту переноса или качество изложения.

<a id="practice-provenance"></a>
### Происхождение практических заданий

Статус официальной лабораторной работы:
`does not expose a verified official lab artifact`. Расписание и страницы
11–15 вводной лекции подтверждают наличие лабораторной работы и проекта,
но не дают проверяемого оригинала условий или стартового репозитория.

Неидентифицированный материал Google Drive не закреплён и не используется.
Для его включения нужны стабильный **file ID**, контрольная сумма
**export checksum** и дата получения **retrieval date**.

Студенческий репозиторий Precioux на версии
`16f7b26346ec4b8978199e6d35bcee96b128bd0a` сохранён лишь как указатель
для поиска (`discovery lead only`) и исключён из официальных материалов.
Собственные упражнения маркируются как **adaptation inspired by the course**.

### Права и ограничения

Запись `rights_status: permission-recorded` фиксирует пользовательское
подтверждение образовательного переиспользования, а не название лицензии
правообладателя. Условия использования сторонних рисунков проверяются
отдельно.

**Встреча 6, страница 117:** отметка
**“NVIDIA CONFIDENTIAL. DO NOT DISTRIBUTE.”** имеет приоритет над общей
записью о разрешении. Страница исключена из переноса
(`rights_scope: do-not-reuse`); локальная архивная копия не предлагается
читателю как скачиваемый полный PDF.

</details>
