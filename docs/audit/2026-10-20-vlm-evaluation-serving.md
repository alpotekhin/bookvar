# Постраничная проверка: оценивание и serving VLM

Дата: 20 октября 2026 года.

Локально проверена глава
`00 Учебник/16 Multimodal Models/64f Оценивание, отказы и serving VLM.md`.
SHA-256: 7158736c550f13fbe8deca537fc369ef87c94a38586dae167fbdc78d215e625a.

[Собранная страница](http://127.0.0.1:4330/textbook/multimodal/evaluation-serving/).

Это результат по одной главе, не завершение учебника.
Следующая глава 64g о диффузионных и flow-моделях ещё не принята.
Переводы, staging, commit, push и deployment не выполнялись.
Работа осталась в editorial-route-alignment; прежний dirty diff сохранён.

## Чтение и редактура

Прочитана вся исходная глава, 191 строка, затем весь переработанный текст:
диапазоны 1–220, 221–442, 443–675; итоговый файл — 656 строк.
После этого текст не менялся. Проверка слога отделена от проверки механизмов:
сохранены подробные объяснения, ограничения, формулы и разобранные примеры.

Route и одиннадцать основных разделов сохранены. Два прежних якоря
`2-как-winoground-изолирует-композиционность` и
`4-проверка-причинной-опоры-на-изображение` сохранены явно,
новые заголовки точнее описывают возможности тестов.

## Прочитанные источники

Архив: `05 Источники/Originals/vlm-evaluation-serving-2026-10-20/`.
references.json содержит URL и SHA-256 27 загрузок и четырёх повторно
использованных источников. Картинки, page snapshots, crop и лицензии
зафиксированы отдельно; raw не изменён.

| Источник | Прочитанные разделы |
|---|---|
| Winoground | §3.1–3.2 полностью, определения text/image/group score |
| POPE, EMNLP2023 | §5.1–5.2 полностью, Figure3 и подпись |
| CHAIR, EMNLP2018 | §2.1 полностью, Figure1 и подпись |
| MM-Vet, PMLR235 | §3.1–3.2 полностью, Figure1, Table1 |
| MMStar v1 | §3 полностью, §4.3, протокол §5.1 |
| EPD, PMLR267 | §3.1–3.2.4 полностью, Figures1/3/4 |
| Molmo v2 | Figure11 и полная подпись, версия5 декабря2024 |
| Stanford CS231n2025 | Lecture16, полный slide50, авторство в footer/metadata |
| vLLM | mm_processing.md полностью; multimodal/hash/salt в prefix caching; online image schema; media/UUID/processor security |
| EncoderCacheManager | docstring и полные __init__, reset, check_and_update_cache, get_cached_num_encoder_embeds |
| OWASP | SSRF, redirects, DNS rebinding и привязка соединения к проверенному IP; visual prompt injection и границы полномочий |

vLLM закреплён на 7b569ec8953cf8099638a6d2ee1e915b75867c3a.
Официальный POPE evaluate.py прочитан целиком; Git tree
08d957b917e5a378a2f99d35b6293c536a66298b не назван commit SHA.

Полностью прочитано двухстраничное соглашение PMLR:
CC BY4.0 и требования к атрибуции. Оно подтверждает права опубликованных
объектов MM-Vet/EPD; лицензии отдельных arXiv-версий не подменялись.
Для POPE/CHAIR сохранена политика ACL для публикаций с2016 года.

Два первоначально предположенных вложенных PMLR PDF URL вернули404.
Скачаны PDF по фактическим ссылкам страниц издателя; неудачные URL
сохранены в references.json. Molmo v1 из другого архива содержит8 страниц,
не Figure11: использован проверенный 30-страничный v2.

## Механизмы и проверенные примеры

| Тема | Результат |
|---|---|
| Состав теста | Взвешенная оценка A=.905>B=.89; средняя по категориям A=.725<B=.85 |
| Winoground | Матрица [[.9,.8],[.1,.2]] даёт text=true, image=false, group=false; равенства не проходят |
| Случайная базовая линия | Все24 строгих ранжирования:6 text,6 image,4 group →1/4,1/4,1/6 |
| POPE | TP45/FP15/FN5/TN35 →precision.75, recall.9, F1=9/11, accuracy.8 |
| Парсер POPE | “I am not sure.” →no, “I am unsure.” →yes по исходному коду; это не семантический судья |
| CHAIR | Три описания с тремя категориями каждое и одним неподтверждённым объектом →CHAIRi=1/3, CHAIRs=1 |
| Контрфактические пары | Перестановка строк сохраняет12480; изменение налога даёт13480; .75 на каждом варианте не исключает .5 на паре |
| MM-Vet | Судья получает вопрос, эталон и ответ, не изображение; три оригинальные строки Table1, AND/OR сохранены |
| MMStar | .70/.45/.30 →MG=.25, ML=.15; это не доказательство утечки конкретного обучающего примера |
| Входной бюджет | Два изображения →3456 visual +256 text +4 boundaries =3716; output reserve200 →3916 |
| Условная память |27MiB features;464.5MiB input KV;489.5MiB с резервом, не параметры Qwen |
| TTFT |460мс server trace +40мс входной путь =500мс клиентских;195мс engine-only пропускают305мс |
| Перекрытие этапов |475мс critical path против710мс суммы работ при заданных независимых ветвях |
| Head-of-line blocking | Среднее завершение коротких запросов255мс в FIFO против55мс при другом порядке; fairness отдельно |
| EPD | MM-features E→P и KV P→D; MM-cache lifetime до подтверждения передачи, не обещание постоянного кеша |
| Сетевой нижний предел |27/464.5MiB при25GiB/s →1.0546875/18.14453125мс, не измерение реальной сети |
| Распределение ролей | Четыре исполнителя E/P/D; изменение encoder capacity меняет оптимальное разбиение |
| Cache identity | Processor, encoder features и prefix cache различены; salt не назван изоляцией всех мультимодальных кешей |
| Отказ | Coverage.4, наблюдаемый risk.25; два безошибочных принятых ответа не дают гарантии риска |
| Качество и deadline |80 правильных,70 вовремя,60 одновременно; за10с →6 пригодных/с, не произведение долей |

В Figure1 MM-Vet эталон14 соответствует7×2, а не придуманному правилу
для последовательности чисел. Соседнее11−2 даёт9, но относится к другому
вопросу. Выбор неверной области, ошибка OCR и ошибка арифметики различены.
Маскирование итоговой ячейки не обязано требовать отказа, если итог
можно вычислить по оставшимся строкам.

## Иллюстрации

Семь оригинальных иллюстраций: пять новых и два восстановления.
Новых Mermaid, SVG или AI-рисунков нет.

| Файл | Источник и crop |
|---|---|
| mmvet-integrated-capabilities.png | PMLR MM-Vet p1, [1930,1164,3540,2570],1610×1406 |
| pope-question-construction.png | ACL POPE p6, [406,406,3141,1391],2735×985 |
| chair-caption-errors.png | ACL CHAIR p1, [1836,1328,3141,2242],1305×914 |
| epd-stage-interference.png | PMLR EPD p1, [1930,1160,3445,2025],1515×865 |
| epd-system-architecture.png | PMLR EPD p4, [469,398,3289,1703],2820×1305 |

Новые crops лежат в Assets/Figures/curated/vlm-evaluation-review-2026-10/.
Снимки PDF высотой5000px сохранены рядом с источниками.
Начальные MM-Vet/EPD crops уточнены до регистрации:
в итоговых файлах не обрезаны вопросы, подписи блоков и оба расписания.

CS231n slide50: прежний2210×1430 full-slide PNG архивирован;
новый2970×850 получен из render3200×2071, crop[110,830,3080,1680].
Панели Winoground/CREPE/ARO и сравнение сохранены.
Footer указывает Ranjay Krishna, PDF metadata — Zane Durante;
оба сигнала записаны в registry, подпись не приписывает слайд последнему.

Molmo Figure11: прежний1124×1005 PNG архивирован; новый2005×1800,
page19 v2, render2473×3200, crop[200,290,2205,2090].
Это историческая таблица авторов, не текущий рейтинг открытости моделей.
Для двух старых объектов сохранены owner confirmation и raw locator,
они не выданы за независимо подтверждённую CC-лицензию.

## Исполнение и сборка

publishing/tools/check_vlm_evaluation_serving_examples.py:

- 17 Python-блоков отдельно и в общем namespace:34 запуска.
- Одна JSON schema с image_url object.
- Все24 Winoground ранжирования и100 проверок перестановок/пересечений.
- Исполнен реальный AST-фрагмент цикла официального POPE parser.
- Исполнены четыре исходных метода vLLM cache manager в минимальном
metadata-only окружении: miss, hit, повторный hit без двойного списания, reset.
- Проверены хеши27 загрузок, четырёх reused PDF, page snapshots и лицензий.
- Все семь crops пиксельно совпадают с зафиксированными областями snapshots.

Предобученные модели, GPU inference и реальные serving-нагрузки
не запускались. Примеры не названы измерениями качества или скорости.

Повторный запуск через системный python3 не нашёл numpy.
Проверка затем успешно выполнена bundled Python
`~/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`;
он сообщил указанный SHA-256 и34 успешных запуска.
Системный interpreter и зависимости не изменялись.

- Unit:261 тест,18 файлов.
- Astro check:12 файлов,0 errors/warnings/hints.
- Финальная сборка:1961 страница, Pagefind завершён.
- Output:20 тестов, включая картинки, KaTeX, старые якоря и AND notation.
- Link check:0 broken routes/fragments/files.
- Source coverage passed; это не доказательство полноты учебника.

Первый output run:19 pass/1 fail. Новый тест ожидал &lt;AND&gt;,
Astro корректно сериализует &amp;#x3C;AND&gt;. Исправлено только ожидание
теста; DOM и таблица осмотрены, после исправления20/20.
Unit/Astro check не повторялись после последующих текстовых правок:
компоненты и registry не менялись. Финальные build/output/link/source
coverage и example checker относятся к принятому тексту.

## Визуальная проверка

Все семь рисунков осмотрены при1365×900 и390×844.
На телефоне документ375px не выходит за viewport; содержимое343px.
Пять display-formulas имеют clientWidth=scrollWidth=343.
Формула TTFT дополнительно осмотрена: строки не обрезаны.
Пять таблиц имеют локальный overflow:auto, четыре требуют прокрутки.
AND в Table1 виден как текст, не скрытый HTML-элемент.

Для всех семи рисунков проверены viewer и увеличение4×.
Для Molmo и EPD architecture проверено горизонтальное перемещение.
EPD дополнительно осмотрен в desktop viewer.
Плотные таблицы на мобильной миниатюре не названы читаемыми:
для изучения мелких подписей используется viewer.
Console errors не обнаружены.

Финальный git diff --check чист; staging, en/ и raw не изменены.

Свидетельства:

- /tmp/bookvar-vlm-evaluation-final-desktop-2026-10-20.jpg
- /tmp/bookvar-vlm-evaluation-desktop-viewer-2026-10-20.jpg
- /tmp/bookvar-vlm-evaluation-mobile-0-2026-10-20.jpg … mobile-6
- /tmp/bookvar-vlm-evaluation-mobile-viewer-0-2026-10-20.jpg … viewer-6
- /tmp/bookvar-vlm-evaluation-mobile-viewer-4-pan-2026-10-20.jpg
- /tmp/bookvar-vlm-evaluation-mobile-viewer-6-pan-2026-10-20.jpg
- /tmp/bookvar-vlm-evaluation-mobile-ttft-2026-10-20.jpg
- /tmp/bookvar-vlm-evaluation-mobile-table-2026-10-20.jpg
- /tmp/bookvar-vlm-evaluation-build-2026-10-20.log
- /tmp/bookvar-vlm-evaluation-output-tests-2026-10-20.log
