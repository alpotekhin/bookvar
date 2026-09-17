---
title: Собрать воспроизводимый pretraining corpus
type: practice
status: canonical
last_updated: 2026-09-15
contract: Contracts/stanford-cs336-a4.yml
source_unit_id:
  - assignment-04-task-look-at-cc
  - assignment-04-deliverable-look-at-cc-001
  - assignment-04-deliverable-look-at-cc-002
  - assignment-04-deliverable-look-at-cc-003
  - assignment-04-deliverable-look-at-cc-004
  - assignment-04-task-extract-text
  - assignment-04-deliverable-extract-text-005
  - assignment-04-deliverable-extract-text-006
  - assignment-04-task-language-identification
  - assignment-04-deliverable-language-identification-007
  - assignment-04-deliverable-language-identification-008
  - assignment-04-deliverable-language-identification-009
  - assignment-04-task-mask-pii
  - assignment-04-deliverable-mask-pii-010
  - assignment-04-deliverable-mask-pii-011
  - assignment-04-deliverable-mask-pii-012
  - assignment-04-deliverable-mask-pii-013
  - assignment-04-deliverable-mask-pii-014
  - assignment-04-task-harmful-content
  - assignment-04-deliverable-harmful-content-015
  - assignment-04-deliverable-harmful-content-016
  - assignment-04-deliverable-harmful-content-017
  - assignment-04-deliverable-harmful-content-018
  - assignment-04-task-gopher-quality-filters
  - assignment-04-deliverable-gopher-quality-filters-019
  - assignment-04-deliverable-gopher-quality-filters-020
  - assignment-04-classifier-evaluation
  - assignment-04-task-quality-classifier
  - assignment-04-deliverable-quality-classifier-021
  - assignment-04-deliverable-quality-classifier-022
  - assignment-04-minhash-lsh-deduplication
  - assignment-04-task-exact-deduplication
  - assignment-04-deliverable-exact-deduplication-023
  - assignment-04-task-minhash-deduplication
  - assignment-04-deliverable-minhash-deduplication-024
  - assignment-04-task-filter-data
  - assignment-04-pipeline-runtime
  - assignment-04-deliverable-filter-data-025
  - assignment-04-deliverable-filter-data-026
  - assignment-04-task-inspect-filtered-data
  - assignment-04-deliverable-inspect-filtered-data-027
  - assignment-04-deliverable-inspect-filtered-data-028
  - assignment-04-deliverable-inspect-filtered-data-029
  - assignment-04-task-tokenize-data
  - assignment-04-deliverable-tokenize-data-030
  - assignment-04-model-validation
  - assignment-04-task-train-model
  - assignment-04-deliverable-train-model-031
  - assignment-04-test-tests-test-pii-py-8-test-mask-emails-single
  - assignment-04-test-tests-test-toxicity-py-8-test-classify-nsfw
  - assignment-04-test-tests-test-extract-py-9-test-extract-text-from-html-bytes
  - assignment-04-test-tests-test-langid-py-9-test-identify-language-english
  - assignment-04-test-tests-test-quality-py-9-test-classify-quality
  - assignment-04-test-tests-test-deduplication-py-11-test-exact-line-deduplication
  - assignment-04-test-tests-test-pii-py-18-test-mask-emails-multiple
  - assignment-04-test-tests-test-langid-py-21-test-identify-language-chinese-simplified
  - assignment-04-test-tests-test-pii-py-28-test-mask-emails-existing-string
  - assignment-04-test-tests-test-quality-py-31-test-gopher-valid-input
  - assignment-04-test-tests-test-toxicity-py-33-test-classify-toxic-speech
  - assignment-04-test-tests-test-quality-py-39-test-gopher-less-than-50-non-symbol-words
  - assignment-04-test-tests-test-pii-py-42-test-mask-phones-single
  - assignment-04-test-tests-test-deduplication-py-43-test-minhash-deduplication-exact-duplicates
  - assignment-04-test-tests-test-quality-py-47-test-gopher-more-than-100000-non-symbol-words
  - assignment-04-test-tests-test-pii-py-54-test-mask-ips
  - assignment-04-test-tests-test-quality-py-55-test-gopher-average-word-length-less-than-3
  - assignment-04-test-tests-test-quality-py-63-test-gopher-average-word-length-greater-than-10
  - assignment-04-test-tests-test-quality-py-73-test-gopher-more-than-30-percent-lines-ending-with-ellipsis
  - assignment-04-test-tests-test-deduplication-py-83-test-minhash-deduplication-fuzzy-duplicates
  - assignment-04-test-tests-test-quality-py-90-test-gopher-less-than-80-percent-words-with-alphabetic-character
primary_sources:
  - https://github.com/stanford-cs336/assignment4-data/tree/0555bea66369872d912652debf10b115ca0688c8
---

# Практика 23. Собрать воспроизводимый pretraining corpus

## Цель

Постройте две версии конвейера данных из одного неизменяемого набора сырых
документов и сравните их при одинаковом обучающем бюджете. Результат практики —
не только `.bin` с токенами, а доказательство того, откуда взялась каждая запись,
почему она сохранена или удалена и какое влияние рецепт оказал на held-out loss.

Практика адаптирует [Stanford CS336 Assignment 4](https://github.com/stanford-cs336/assignment4-data/tree/0555bea66369872d912652debf10b115ca0688c8).
Решения Stanford не переносятся. Используются его открытые интерфейсы адаптеров
и тестов, а также собственный небольшой локальный набор примеров.

## Что фиксируется до эксперимента

До просмотра модельного результата запишите:

- идентификатор и SHA-256 локального набора либо список объектов Common Crawl;
- две сравниваемые конфигурации конвейера;
- ревизию токенизатора и его хэш;
- model shape, optimizer, seed, context length и sampled-token budget;
- validation corpus/version;
- основную метрику и допустимые причины исключения запуска;
- максимум итераций конвейера, чтобы не переобучить рецепт под валидацию.

<a id="raw-inspection"></a>
## 1. Увидеть разницу между WARC и WET

<!-- source_unit_id: assignment-04-task-look-at-cc -->
<!-- source_unit_id: assignment-04-deliverable-look-at-cc-001 -->
<!-- source_unit_id: assignment-04-deliverable-look-at-cc-002 -->
<!-- source_unit_id: assignment-04-deliverable-look-at-cc-003 -->
<!-- source_unit_id: assignment-04-deliverable-look-at-cc-004 -->

Начните с готовых входов официального репозитория: `tests/fixtures/moby.html`
и ожидаемого текста `tests/fixtures/moby_extracted.txt`. Они позволяют
проверить одну стадию извлечения без загрузки Common Crawl. Подготовка
рабочей копии и запуск после реализации адаптера:

```bash
git clone https://github.com/stanford-cs336/assignment4-data.git
git -C assignment4-data checkout 0555bea66369872d912652debf10b115ca0688c8
cd assignment4-data
uv sync
uv run pytest -v tests/test_extract.py
uv run pytest -v tests/test_deduplication.py
```

На незаполненных адаптерах ожидается ошибка, а не успешный тест. Имена файлов
и ожидаемые результаты заданы в закреплённых
[тестах извлечения](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/tests/test_extract.py)
и [дедупликации](https://github.com/stanford-cs336/assignment4-data/blob/0555bea66369872d912652debf10b115ca0688c8/tests/test_deduplication.py).

Следующий этап — **подготовить**, а не найти уже готовый в Bookvar набор из
не менее 25 небольших HTML-документов и их текстовых производных с разрешённым
использованием. Это локальная имитация обработки веб-корпуса, не готовый
WARC/WET-пакет. Чтобы исследовать именно контейнеры WARC/WET, отдельно
возьмите закреплённый небольшой фрагмент Common Crawl и сохраните его адрес
и контрольную сумму. Для выбранных документов заполните таблицу:

| doc_id | URL/домен | язык | тип страницы | полезный текст | шаблонный мусор/потери извлечения | решение |
|---|---|---|---|---|---|---|

Отдельно ответьте, для какого приложения спорный документ полезен, а для какого
нет. «Качество» без целевой модели использования не является достаточной
меткой.

В full mode используется закреплённый список Common Crawl crawl/segments/paths.
Не подменяйте его запросом «последний crawl»: такой результат нельзя повторить.

<a id="adapters"></a>
## 2. Реализовать точные интерфейсы

<!-- source_unit_id: assignment-04-task-extract-text -->
<!-- source_unit_id: assignment-04-task-language-identification -->
<!-- source_unit_id: assignment-04-task-quality-classifier -->

В `tests/adapters.py` подключите собственную реализацию к одиннадцати функциям:

```python
run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None
run_identify_language(text: str) -> tuple[Any, float]
run_mask_emails(text: str) -> tuple[str, int]
run_mask_phone_numbers(text: str) -> tuple[str, int]
run_mask_ips(text: str) -> tuple[str, int]
run_classify_nsfw(text: str) -> tuple[Any, float]
run_classify_toxic_speech(text: str) -> tuple[Any, float]
run_classify_quality(text: str) -> tuple[Any, float]
run_gopher_quality_filter(text: str) -> bool
run_exact_line_deduplication(input_files, output_directory)
run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams,
                          jaccard_threshold, output_directory)
```

Оригинальные тесты не изменяются. Допишите собственные adversarial tests:
не-UTF-8 HTML, mixed-language/code, международный телефон, версия похожая на IP,
идемпотентное повторное masking, пустой документ, Unicode normalization,
транзитивный duplicate cluster и стабильный representative.

<a id="filter-ledger"></a>
## 3. Собрать конвейер как журнал решений

<!-- source_unit_id: assignment-04-task-gopher-quality-filters -->
<!-- source_unit_id: assignment-04-deliverable-gopher-quality-filters-019 -->
<!-- source_unit_id: assignment-04-deliverable-gopher-quality-filters-020 -->
<!-- source_unit_id: assignment-04-deliverable-quality-classifier-021 -->
<!-- source_unit_id: assignment-04-deliverable-quality-classifier-022 -->
<!-- source_unit_id: assignment-04-task-filter-data -->
<!-- source_unit_id: assignment-04-pipeline-runtime -->
<!-- source_unit_id: assignment-04-deliverable-filter-data-025 -->
<!-- source_unit_id: assignment-04-deliverable-filter-data-026 -->

Создайте два рецепта:

- **baseline:** извлечение, порог определения языка, прозрачные правила Gopher и
  точная дедупликация;
- **candidate:** обоснованное изменение извлечения, фильтра, классификатора или поиска приблизительных повторов
  при сохранении того же сырого входа.

Каждая стадия читает immutable input и пишет отдельный output. В
`decision_log.jsonl` для каждой записи сохраняются:

```json
{
  "doc_id": "fixture/000017",
  "stage": "language-id-v1",
  "action": "retained",
  "score_or_rule": {"label": "en", "score": 0.91, "threshold": 0.70},
  "reason": "above preregistered threshold",
  "input_sha256": "...",
  "output_sha256": "..."
}
```

`stage_counts.json` должен сходиться по документам, байтам и токенам. Для
каждого фильтра укажите число принятых, изменённых, удалённых и quarantined
записей. Измерьте wall time, CPU count, peak RSS и throughput; оцените время
обработки заявленного full snapshot, явно указав модель экстраполяции.

<a id="dedup"></a>
## 4. Подтвердить exact и fuzzy duplicates

<!-- source_unit_id: assignment-04-minhash-lsh-deduplication -->
<!-- source_unit_id: assignment-04-task-exact-deduplication -->
<!-- source_unit_id: assignment-04-deliverable-exact-deduplication-023 -->
<!-- source_unit_id: assignment-04-task-minhash-deduplication -->
<!-- source_unit_id: assignment-04-deliverable-minhash-deduplication-024 -->

Официальный `test_exact_line_deduplication` читает
`tests/fixtures/documents_with_line_duplicates/doc*.txt` и сравнивает пять
выходных файлов с `tests/fixtures/documents_line_deduplicated/doc*.txt`.
Это конкретный тестовый набор, не требование получать пять файлов из любого
корпуса. Проверьте его приведённой выше командой, затем добавьте свои случаи.
Стадия MinHash принимает `num_hashes`, `num_bands`, словесные `ngrams`,
`jaccard_threshold` и output directory. Сначала LSH создаёт candidates, затем
точный Jaccard подтверждает рёбра, connected components задают кластеры.

В `dedup_clusters.jsonl` сохраните правило построения подписей и seed, пары-кандидаты,
точный Jaccard, component ID, правило выбора и retained document. Постройте
небольшую precision/recall таблицу на вручную размеченных парах; тест с двумя
MIT licenses проверяет функцию, но не качество на реальном корпусе.

<a id="manual-audit"></a>
## 5. Измерить ошибки фильтров

<!-- source_unit_id: assignment-04-classifier-evaluation -->
<!-- source_unit_id: assignment-04-task-inspect-filtered-data -->
<!-- source_unit_id: assignment-04-deliverable-inspect-filtered-data-027 -->
<!-- source_unit_id: assignment-04-deliverable-inspect-filtered-data-028 -->
<!-- source_unit_id: assignment-04-deliverable-inspect-filtered-data-029 -->

Случайная выборка только retained records не показывает, что было ошибочно
удалено. Постройте страты по `stage × action × language/domain/length bucket` и
проверьте принятые, отклонённые и изменённые записи.

`manual_audit.csv` содержит фрагмент исходного текста, предсказание и оценку,
человеческую метку, FP/FN, причину и reviewer. Отдельно отчитайте:

- language-ID errors и mixed/unknown cases;
- false positives/negatives для email, phone и IP;
- NSFW/toxicity disagreements;
- Gopher-rule и quality-classifier disagreements;
- justified/unjustified duplicate removals.

Порог меняют только через новую версионированную конфигурацию конвейера. Старый запуск
и его ошибки сохраняются.

<a id="tokenize"></a>
## 6. Токенизировать, не потеряв происхождение

<!-- source_unit_id: assignment-04-task-tokenize-data -->
<!-- source_unit_id: assignment-04-deliverable-tokenize-data-030 -->

Закрепите GPT-2 tokenizer revision для совместимости с upstream full mode или
укажите собственный неизменяемый tokenizer. После каждого документа добавляйте
EOS. Для GPT-2 достаточно `uint16 .bin`; для другого токенизатора сначала
проверьте максимальный ID, включая служебные токены. Если он превышает 65 535,
используйте `uint32`, чтобы сериализация не обрезала старшие биты.
Зафиксируйте dtype и порядок байтов. Вместе с `.bin` запишите `tokenizer_manifest.json` с hash,
special-token IDs, числом документов/токенов и индексом диапазонов
`doc_id → [start,end)`.

Проверьте обратное чтение сериализованных ID на тестовом примере, границы EOS
и соответствие числа элементов заявленному числу токенов. Идентификаторы
валидационных **документов** не должны встречаться в обучающем манифесте;
это не запрет на общие ID токенов из одного словаря.

<a id="fixed-budget"></a>
## 7. Сравнить данные при фиксированном бюджете

<!-- source_unit_id: assignment-04-model-validation -->
<!-- source_unit_id: assignment-04-task-train-model -->
<!-- source_unit_id: assignment-04-deliverable-train-model-031 -->

Для базового и исследуемого вариантов должны совпадать:

- tokenizer и packing;
- model initialization/shape;
- optimizer и learning-rate schedule;
- context length, batch и число sampled tokens;
- seeds и evaluation cadence;
- validation corpus.

Сохраните learning curves, best и final validation loss, throughput и срезы
по доменам. Если исследуемый корпус меньше бюджета, выборку с повторением нужно явно
показать через expected/actual exposures. Не сравнивайте run с большим числом
токенов против baseline и не называйте разницу эффектом filtering.

Stanford full reference — примерно 430M parameters, context 512, 16,384 steps,
8 B200, batch 128 per device и около $2^{33}=8.6$B sampled tokens. Это описание
их инфраструктуры, а не требование или порог прохождения практики Bookvar.

<a id="evidence-bundle"></a>
## 8. Сдать доказательство, а не только данные

```text
evidence/
  environment.json
  source_manifest.jsonl
  pipeline_config.yml
  stage_counts.json
  decision_log.jsonl
  manual_audit.csv
  dedup_clusters.jsonl
  mixture_manifest.json
  tokenizer_manifest.json
  runtime.json
  learning_curve.csv
  validation.json
  artifact_hashes.json
  report.md
```

Практика пройдена, если собственная реализация подключена через upstream
adapters с неизменными сигнатурами, исходные тесты и fixtures не изменены,
повторный smoke run даёт те же hashes данных, балансы стадий сходятся, validation не
попадает в train, duplicate representatives детерминированы, manual audit
содержит обе стороны решений, а fixed-budget comparison воспроизводим. Улучшение
loss не является обязательным: корректно объяснённый отрицательный результат —
полноценный результат эксперимента.

Машиночитаемая спецификация: `06 Практика/Contracts/stanford-cs336-a4.yml`.

## Источники

- [Stanford CS336 Assignment 4, pinned revision](https://github.com/stanford-cs336/assignment4-data/tree/0555bea66369872d912652debf10b115ca0688c8)
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных]]
- [[02 Areas/ML & DL/00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса]]
