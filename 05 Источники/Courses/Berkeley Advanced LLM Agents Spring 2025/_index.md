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
слайды и список статей соответствующей лекции. Материалы курса не образуют
отдельную обязательную последовательность внутри нашего учебника.

## Где читать тему

| Тема | В учебнике | Оригинальные лекции |
|---|---|---|
| Поиск решения, проверка и дополнительные вычисления при ответе | [[00 Учебник/13 Reasoning и Test-time Compute/01 Test-time compute|Test-time compute]], [[00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers|RLVR и проверяющие программы]] | [1–2: reasoning](#meeting-01), [4: post-training](#meeting-04) |
| Память, планирование и модели мира | [[00 Учебник/17 Tools и Agents/67 Память, планирование и оркестрация агентов|Память и планирование]] | [3: memory and planning](#meeting-03) |
| Программирование, браузер и графический интерфейс | [[00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents|Coding, web и computer-use agents]] | [5: coding](#meeting-05), [6: web](#meeting-06), [7: GUI](#meeting-07) |
| Формализация и доказательство теорем | [[00 Учебник/17 Tools и Agents/70 Формальные доказательства и математические агенты|Математические агенты]] | [8: AlphaProof](#meeting-08), [9: autoformalization](#meeting-09), [10: theorem proving](#meeting-10) |
| Научный поиск и построение абстракций | [[00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery|Агенты научного поиска]] | [11: abstraction and discovery](#meeting-11) |
| Угрозы, недоверенные данные и ограничение полномочий | [[00 Учебник/17 Tools и Agents/72 Безопасность агентных систем|Безопасность агентов]] | [12: safe and secure agentic AI](#meeting-12) |

## Что уже перенесено и что осталось

Материал уже связан с главами, а не только сохранён в архиве. В реестре
`coverage.yml` 140 фрагментов отмечены как `integrated`, ещё 6 — как
`covered-existing`: для них указаны конкретные разделы учебника и обратные
ссылки на источник. 38 фрагментов пока доступны только в исходных материалах,
28 исключены с указанием причины. Это числа смысловых фрагментов, не лекций
или страниц PDF.

В числе 140 записей `integrated` — 105 фрагментов слайдов и 35 записей
каталога статей. Последние дают ссылку на первоисточник и его описание;
полные тексты этих статей не перенесены в учебник.

Статус переноса подтверждает наличие текста и связей, но не заменяет
постраничную проверку объяснения и читаемости рисунков. Оставшиеся материалы
разбираются по темам: полезное дополнение расширяет соответствующую главу;
дублирование и административная информация не становятся новыми главами.

### Редакционная карта

Карта активна. Валидатор проверяет страницы назначения, якоря, обратные ссылки
`source_unit_id` и локальные файлы перенесённых иллюстраций. Из 161 записи
визуального реестра 47 отмечены `integrated`, 87 — `source-only`, 27 —
`excluded`. Запись может описывать последовательность слайдов, а не один рисунок.

Проверочные данные: [coverage.yml](coverage.yml), [visuals.yml](visuals.yml),
[editorial-map.yml](editorial-map.yml), [source-units.yml](source-units.yml),
[semantic-review.json](semantic-review.json), [audit-contract.json](audit-contract.json),
[source-manifest.yml](source-manifest.yml), [snapshot-lock.json](snapshot-lock.json).
Эти реестры нужны для обновления и аудита; для чтения достаточно ссылок в таблице.

## Что закреплено

- 12 встреч из официального Spring 2025 syllabus;
- 12 official recording metadata records: channel, video ID, точный title и
  duration;
- **13 неизменённых official PDF**, 1 254 physical pages и 13 детерминированных
  `pdftotext -layout` page indexes;
- 37 индивидуальных reading records с распределением
  `3, 3, 3, 3, 2, 4, 2, 4, 3, 4, 2, 4`;
- 161 visual rows, которые в точности закрывают все 1 254 страницы и связаны с
  semantic unit, parent SHA-256 и extraction provenance.

Числа «12 meetings» и «13 PDF» намеренно различаются. Встреча 27 января
содержит два отдельных официальных артефакта: 16-page **Intro** и 74-page
основной deck. Intro не схлопнут и не потерян ради искусственного равенства
«одна встреча = один PDF»; у него собственные manifest object, page closure и
visual disposition.

`semantic-review.json` — карта direct visual/semantic review. Каждый physical page был
просмотрен в упорядоченных contact sheets, сопоставлен с извлечённым текстом и
отнесён к именованному смысловому разделу или к явному reasoned exclusion.
Строки не создавались по ключевым словам и не являются generic one-page proxy.
Offline `python3 publishing/tools/import_berkeley_agents.py --check` заново
извлекает page indexes, строит все четыре ledger-документа из review spec,
проверяет locked syllabus membership, exact page closure и deterministic lock.

### Независимый completeness contract

`audit-contract.json` — отдельный вручную проверенный completeness anchor: он
фиксирует exact `semantic_id`, title, kind, visual kind, physical-page range и
disposition всех 161 смысловых секций, reason/evidence для exclusions и SHA-256
каждого из 13 исходных PDF. Importer дополнительно hard-pin’ит SHA-256 самого
contract. Поэтому изменение или схлопывание `semantic-review.json` не может
стать новым baseline даже после согласованной пересборки всех четырёх ledgers,
artifact inventory и snapshot lock.

`--refresh` и `--regenerate-ledgers` **никогда не записывают**
`audit-contract.json`. Для осознанного обновления нужно заново просмотреть все
затронутые physical pages в ordered contact sheets, сверить page text, вручную
изменить оба review-файла, проверить exact semantic/range/disposition diff,
вычислить новый contract SHA-256 и явно обновить `AUDIT_CONTRACT_SHA256` в
importer. Затем обязательны focused regressions, importer `--check`, ingestion
и coverage validators, full publishing suite и scoped diff review.

> [!danger]
> В неизменённом official deck встречи 6 physical page 117 явно маркирован
> **“NVIDIA CONFIDENTIAL. DO NOT DISTRIBUTE.”** Архивная копия сохраняется только
> ради точности official snapshot. Visual row имеет `excluded` и
> `rights_scope: do-not-reuse`; общая permission record курса не перекрывает
> более строгое ограничение конкретной страницы.

<a id="встречи-и-будущие-назначения"></a>
## Лекции, слайды и статьи

<a id="meeting-01"></a>
### 1. 27 января — inference-time reasoning

Артефакты: [Intro, 16 pages](Lectures/meeting-01-intro.pdf),
[основной deck, 74 pages](Lectures/meeting-01-slides.pdf), recording metadata и
3 readings. Темы: prompting, decomposition, self-consistency,
verifiers, Tree of Thoughts, reflection и self-correction failures.

Readings:

1. [Large Language Models as Optimizers](Readings/meeting-01-reading-01.md)
2. [Large Language Models Cannot Self-Correct Reasoning Yet](Readings/meeting-01-reading-02.md)
3. [Teaching Large Language Models to Self-Debug](Readings/meeting-01-reading-03.md)

<a id="meeting-02"></a>
### 2. 3 февраля — learning to reason

Артефакты: [deck, 106 pages](Lectures/meeting-02-slides.pdf), recording metadata
и 3 readings. Темы: DPO, self-rewarding models, IRPO,
meta-rewarding и evaluation-guided planning.

Readings:

1. [Direct Preference Optimization](Readings/meeting-02-reading-01.md)
2. [Iterative Reasoning Preference Optimization](Readings/meeting-02-reading-02.md)
3. [Chain-of-Verification](Readings/meeting-02-reading-03.md)

<a id="meeting-03"></a>
### 3. 10 февраля — reasoning, memory, and planning

Артефакты: [deck, 80 pages](Lectures/meeting-03-slides.pdf), recording metadata
и 3 readings. Visual ledger отдельно сохраняет реальные ordered sequences для
agent-first/LLM-first, HippoRAG memory и world-model planning.

Readings:

1. [Grokked Transformers are Implicit Reasoners](Readings/meeting-03-reading-01.md)
2. [HippoRAG](Readings/meeting-03-reading-02.md)
3. [Is Your LLM Secretly a World Model of the Internet?](Readings/meeting-03-reading-03.md)

<a id="meeting-04"></a>
### 4. 24 февраля — open post-training recipes

Артефакты: [deck, 155 pages](Lectures/meeting-04-slides.pdf), recording metadata
и 3 readings. Темы: Tülu, SFT mixtures, preference optimization,
RLVR, test-time scaling и OLMo openness. Physical page 155 сохранена как
содержательный chart `Human Preference Evaluation`, а не generic appendix.

Readings:

1. [Tulu 3](Readings/meeting-04-reading-01.md)
2. [Unpacking DPO and PPO](Readings/meeting-04-reading-02.md)
3. [OpenScholar](Readings/meeting-04-reading-03.md)

<a id="meeting-05"></a>
### 5. 3 марта — coding agents and vulnerability detection

Артефакты: [deck, 55 pages](Lectures/meeting-05-slides.pdf), recording metadata
и 2 readings. Visual ledger сохраняет полный multi-page vulnerability-discovery
loop Charles Sutton.

Readings:

1. [EnIGMA](Readings/meeting-05-reading-01.md)
2. [From Naptime to Big Sleep](Readings/meeting-05-reading-02.md)

<a id="meeting-06"></a>
### 6. 10 марта — multimodal web agents

Артефакты: [deck, 126 pages](Lectures/meeting-06-slides.pdf), recording metadata
и 4 readings. Темы: Mind2Web, WebArena, VisualWebArena, tree
search, verification и inference-time scaling. Page 117 — обязательное
`do-not-reuse` исключение, указанное выше.

Readings:

1. [Mind2Web](Readings/meeting-06-reading-01.md)
2. [WebArena](Readings/meeting-06-reading-02.md)
3. [VisualWebArena](Readings/meeting-06-reading-03.md)
4. [Tree Search for Language Model Agents](Readings/meeting-06-reading-04.md)

<a id="meeting-07"></a>
### 7. 17 марта — GUI agents from perception to action

Артефакты: [deck, 106 pages](Lectures/meeting-07-slides.pdf), recording metadata
и 2 readings. Темы: OSWorld, trajectory construction, TACO,
Aguvis, video understanding и generalist agents.

Readings:

1. [OSWorld](Readings/meeting-07-reading-01.md)
2. [Aguvis](Readings/meeting-07-reading-02.md)

<a id="meeting-08"></a>
### 8. 31 марта — AlphaProof

Артефакты: [deck, 112 pages](Lectures/meeting-08-slides.pdf), recording metadata
и 4 readings. Темы: formal mathematics, Lean, AlphaZero,
AlphaProof, IMO evidence и test-time RL.

Readings:

1. [AI achieves silver-medal standard at the IMO](Readings/meeting-08-reading-01.md)
2. [Mastering Chess and Shogi by Self-Play](Readings/meeting-08-reading-02.md)
3. [The Future of Mathematics?](Readings/meeting-08-reading-03.md)
4. [Building the Mathematical Library of the Future](Readings/meeting-08-reading-04.md)

<a id="meeting-09"></a>
### 9. 7 апреля — autoformalization and theorem proving

Артефакты: [deck, 114 pages](Lectures/meeting-09-slides.pdf), recording metadata
и 3 readings. Visual ledger сохраняет ordered Lean/theorem-proving pipeline
Kaiyu Yang, а также logical-gap и geometry sequences.

Readings:

1. [LeanDojo](Readings/meeting-09-reading-01.md)
2. [Autoformalization with Large Language Models](Readings/meeting-09-reading-02.md)
3. [Autoformalizing Euclidean Geometry](Readings/meeting-09-reading-03.md)

<a id="meeting-10"></a>
### 10. 14 апреля — advanced theorem proving

Артефакты: [deck, 118 pages](Lectures/meeting-10-slides.pdf), recording metadata
и 4 readings. Темы: Lean-STaR, Draft-Sketch-Prove, LeanHammer,
research workflows и miniCTX. Pages 116–117 остаются source-only как
содержательный accessibility/benchmarking и prover-method recap; только p.118
имеет administrative `excluded` disposition.

Readings:

1. [Draft, Sketch, and Prove](Readings/meeting-10-reading-01.md)
2. [miniCTX](Readings/meeting-10-reading-02.md)
3. [Lean-STaR](Readings/meeting-10-reading-03.md)
4. [ImProver](Readings/meeting-10-reading-04.md)

<a id="meeting-11"></a>
### 11. 21 апреля — abstraction and discovery

Артефакты: [deck, 93 pages](Lectures/meeting-11-slides.pdf), recording metadata
и 2 readings. Visual ledger сохраняет ordered LaSR/concept-library discovery
sequence Swarat Chaudhuri.

Readings:

1. [An In-Context Learning Agent for Formal Theorem-Proving](Readings/meeting-11-reading-01.md)
2. [Symbolic Regression with a Learned Concept Library](Readings/meeting-11-reading-02.md)

<a id="meeting-12"></a>
### 12. 28 апреля — safe and secure agentic AI

Артефакты: [deck, 99 pages](Lectures/meeting-12-slides.pdf), recording metadata
и 4 readings. Visual ledger сохраняет две самостоятельные Dawn Song sequences:
agentic threat model и privilege-control architecture.

Readings:

1. [Privtrans](Readings/meeting-12-reading-01.md)
2. [DataSentinel](Readings/meeting-12-reading-02.md)
3. [AgentPoison](Readings/meeting-12-reading-03.md)
4. [Progent](Readings/meeting-12-reading-04.md)

<a id="practice-provenance"></a>
## Граница practice provenance

Locked official syllabus и Intro pages 11–15 подтверждают lab, grading, final
project и project timeline. При этом public syllabus **does not expose a verified official lab artifact**:
на странице нет проверяемой официальной ссылки на lab handout или starter repository.

Неидентифицированный Google Drive artifact не закреплён и недоступен для
использования. До любого переноса обязательны stable Drive **file ID**,
immutable **export checksum** и **retrieval date**.

Pinned Precioux repository сохранён только как third-party discovery link на
commit `16f7b26346ec4b8978199e6d35bcee96b128bd0a`. Его disposition — `excluded`,
reason — `discovery lead only`: student mirror не может доказывать официальный
lab contract. Любое будущее Bookvar-упражнение следует маркировать как
**adaptation inspired by the course**, а не как официальный Berkeley lab.

## Права и пределы переиспользования

Для official course artifacts записан `rights_status: permission-recorded` на
основании пользовательского разрешения на образовательное сохранение и
атрибутированное дальнейшее использование. Это **permission record, not a
named license**. Для readings сохраняются только metadata и canonical links;
тела copyrighted papers не зеркалируются. Права upstream figures всё равно
проверяются item-by-item перед интеграцией. Ограничение meeting 6, page 117
всегда имеет приоритет над общей записью разрешения.
