# Отбор данных: стоимость, шарды и проверка качества

2026-10-28. Добавлена и проверена читательская глава 41i:
`00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества.md`.
691 строка, SHA-256
`b0e6e4621d64b1583c7522a3bb5cdc51b6f4ddaed461c7ebe1df0ad188e2feaf`.
Маршрут: `/textbook/pretraining/selection-engineering-evaluation/`.

Это содержательное продвижение общей задачи: дочитан источник и добавлено
изложение недостающих инженерных и измерительных разделов. Готовые RL-главы
повторно не редактировались. Приёмка 41i не закрывает весь учебник,
Harvard H-V1-09/H-F1-09 или общий аудит происхождения материалов.

## Чтение и границы источников

Harvard QMD при commit `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`
прочитан до конца: cumulative 1–4741, включая TikZ и Python.
В этом проходе закончены Selection Engineering, Cost Modeling,
Distributed Selection, Cross-Layer Interactions, Measurement Framework,
Fallacies/Pitfalls и Summary. Прежние частичные курсоры сохранены как история
в старых отчётах, но больше не определяют следующий unread cursor.

Исходный файл:
`05 Источники/Originals/data-selection-2026-10-28/chapter-2bd97c5.qmd`,
SHA `7bdfce0af42c461eb4277da4f61f3a18442af22360b65ce57a8073fef7de8f5c`.
Новый packet: `05 Источники/Originals/selection-engineering-2026-10-28/`.

Для 41i использованы полностью прочитанные разделы QMD 2761–4482.
Choi et al., arXiv 1907.05550v3, HTML §§1–3.2 прочитаны для определения
echo point, Eq. 1, различия fresh visits/presentations и конкретного
сетевого эксперимента. Последующие разделы paper не объявлены необходимой
частью этой приёмки. Нет переноса его рисунков или длинных цитат.

Существующая 41b прочитана целиком, 615 строк, и не переписана.
У 41c проверен заключительный раздел об оценивании/затратах; у Data pipeline 06 —
предпосылки и оглавление. Это не новая полная приёмка этих двух глав.
После правок 41i прочитана целиком по сохранённому файлу: 1–240,
241–480, 481–691. Объяснения проверены отдельно от no-ai-slop/eval.

## Приёмка объяснений

| Механизм | Раздел источника | Разобранный случай и ограничение |
| --- | --- | --- |
| Полная стоимость отбора | Selection bottleneck, 2765–3014 | 100 против 5+40 и 60+40; запас 60, а не универсальное правило 10%; inference-time не принят за training-time |
| Повторное оценивание | Selection engineering; consistency challenges | 100 эпох, 0.7 часа scoring, 0.4 training: интервалы 1/10/100 дают 110/47/40.7; serial cost отделён от overlapped wall time |
| Эпохи и шаги | Selection inequality; existing 41b | N=10000, B=100, E=10; subset 20%: 200 против 1000 шагов; 50 subset epochs возвращают исходное число шагов |
| Label/lifecycle accounting | Cost Modeling, 3230–3439 | 5000×10 против (1000+10×100)×10+10×50; 59% saving с перечисленными неучтёнными расходами |
| Амортизация | 3440–3549 | I=55000, net recurring saving 10000: пять запусков ещё −5000, шесть +5000; равенство и положительная прибыль разделены |
| I/O access pattern | Random-access penalty, 3015–3112 | 4 KiB, 1M/100K records, assigned 100 MiB/s и 100 IOPS: 39.0625/1000 seconds; отношение bandwidth 256, времени 25.6 |
| Sequential filtering/materialization | Selection Engineering; unchanged 06 | Полный scan с ID-фильтром всё ещё читает discarded bytes; новые компактные шарды требуют upfront pass |
| Distributed scoring | Distributed Selection, 3550–3735 | Разделены embedding payload 3.072 GB, packed scores/IDs 12 MB и final IDs 0.8 MB; фактическая передача зависит от реализации |
| Global top-k | Strategies for distributed selection | A=[9.9,9.8,9.7,9.6], B=[2,1.9,1.8,1.7]; quota 2 теряет 9.7; local top-k содержит global top-k при общей полной сортировке; это не coreset diversity theorem |
| Версии и повторные IDs | Consistency challenges; existing 41b | Два checkpoint p_label=0.1/0.9 дают EL2N 1.273/0.141; sampler padding, label/reduction versions и cross-shard duplicates оговорены |
| Distributed overhead | Архивный восьми-worker example | 20+15+30+2=67 минут — assigned values, не benchmark; F>74.44 minutes; 24h→211min, overall 6.82× при условном 0.1F |
| Data echoing | Harvard 3113–3229; Choi §2, §3.2 | Tau=max(t_u,e*t_d); e=1/2/3/4, 300/800 rates; разные итоги при равных presentations и fresh visits |
| PPD/AULC/DCR | Core metrics, 4145–4175 | Общая hypothetical таблица, PPD .0026; trapezoids 63+79+165=307; normalized .7675; DCR 2 при target .82, а не storage ratio 4 |
| Subgroup quality | Measurement framework; pitfalls | Majority/rare 99/1%; average .949→.9648 при rare .85→.45; раздельные evaluation/test и выбор thresholds |
| Cross-layer savings | 3736–4140; frontier diagnostic | 20 fixed +80 model hours →40 при сокращении только model work вчетверо; corpus reduction не сокращает параметры/KV/inference автоматически |

В самостоятельной проверке рассуждения для каждого основного механизма
пройдены исходные величины, промежуточные операции и итог. Таблица метрик
явно hypothetical; настоящие кривые или новые ML-результаты не заявлены.

## Уточнения архивного курса

Не перенесены как установленные факты:

- Названный «measured» восьми-GPU пример, заданный константами сценария.
- Ускорение полного обучения 10× только из сохранения 10% данных.
- Порог 10% overhead как необходимое условие положительной экономии.
- Одинаковая стоимость forward scoring и training pass.
- Автоматическое совпадение локальных квот с глобальной сортировкой.
- Гарантированный перенос proxy ranking между моделями и precision modes.
- Неизменная скорость fresh visits после насыщения downstream стадии.
- Универсальный полезный echo factor 4 и обязательная поломка BatchNorm.
- Универсальные D/P=10/20/40 и plateau как однозначный диагноз нехватки данных.
- Пропорциональное сокращение serving/compression стоимости вслед за corpus size.
- Назначенные в pitfalls числа synthetic accuracy, domain redundancy,
  «150 minimum samples» и смеси 50–80% как общие эмпирические результаты.

Прочитанные Decision Framework и distillation 2602–2760 всё ещё требуют
явной сверки с существующими главами; их общие обещания не импортированы.

## Исполнение и ограничения

`python3 publishing/tools/check_selection_engineering_examples.py`:

- Все восемь actual Python-блоков исполнены в отдельных namespaces.
- 450 вариантов serial refresh accounting с неполными последними интервалами.
- 12 точных strict break-even случаев.
- 360 shard/top-k случаев с ties, пустыми/неравными шардами.
- 48 независимых конечных two-stage schedules с учётом fill/drain.
- 24 точных рациональных AULC/first-observed-budget случая.
- Три исходных рисунка byte-identical; SHA, referring HTML, PNG chunk CRC,
  IDAT decompression и размеры проверены.

Итоговый лог:
`/var/tmp/bookvar-selection-engineering-examples-final-2026-10-28.log`,
SHA главы совпадает с указанным в начале. Только standard-library CPU
арифметика: нет GPU, обучения, датасета, замера IOPS, speed benchmark
или воспроизведения качества из course/paper.

## Рисунки и происхождение

Три полных оригинала Harvard web edition:

- selection-inequality.png: 3394×2233,
  SHA `0bf8c01cae14cbde4a57aa17015537706c839cc7bee5be1c6310e141cc1d35c2`.
- distributed-selection.svg: viewBox 0 0 354.23468 143.04933,
  SHA `4b00bd46c5d58c272e24fcb998580fe0c8bf1fa4f89a06c8667efc84ad49f12c`.
- performance-data.png: 3391×2257,
  SHA `033f5554b7a0efd0d48986b06e374f00cfa5dd30f61ba5b022b6c8f93ab52d21`.

Referring HTML SHA:
`9f31e8439bd840b3f6993851089acd305b230070bdd4fd6474c68fa39192f09e`.
Footer текущей страницы подтверждает CC BY-NC-ND 4.0.
Рисунки перенесены неизменёнными как атрибутированные объекты коллекции;
не названы CC BY-NC-SA только потому, что старая QMD имела эту лицензию.
В bytes рисунков нет crop, redraw, перевода, recompression или upscaling.
Для SVG задана только CSS display width min(100%,40rem): исходный вектор
мелкий при своём print-размере. Он не изменён.

Рядом с каждым — автор, прямая ссылка, лицензия и объяснение illustrated
versus empirical. assets.json, ATTRIBUTION, central registry и две публичные
license-копии согласованы. Tiny echo margin SVG сохранён только в raw packet:
весь SVG-код проверен, оси без названий, поэтому он не заменяет объяснение
fresh visits/presentations и не считается новым читательским рисунком.

## Вёрстка, навигация и сборка

Реальный IAB QA на desktop 1280×900 и mobile 360×900:

- Все три рисунка загружаются и осмотрены на странице/в просмотрщике.
- SVG после CSS-правки занимает 632 px desktop, 328 px mobile.
- Все 8 display-формул: 328/328; две таблицы: 328/328.
- document width/scrollWidth 360/360; ошибок KaTeX 0; inline math >328 px нет.
- Валюта отображается обычным текстом: dollar signs не захватывают prose в math.
- При 4× mobile все три рисунка доступны по горизонтали, x=0 и x=1032.
  У performance plot проверены y=0 и y=221; у cost plot y=0 и около210.5;
  у SVG вертикального overflow нет. Fit/reset и закрытие работают.
- Фактические footer clicks: 41i→MAE→41i→42→41i.
- EN route показывает явный fallback с 3 рисунками; ошибок рендера нет.
  Authored EN body не создавался и не объявлен готовым переводом.
- Итоговая RU-глава оставлена в tab5 как deliverable. Screenshot:
  `/var/tmp/bookvar-selection-engineering-2026-10-28.png`.

Проверки:

- Полный unit suite: 625 tests / 98 files PASS до финальных правок
  introductory attribution/alt/CSS; после них 27 affected tests / 8 files PASS.
- Финальный Astro build: 1985 pages, 27.33s, exit0.
- После завершения build: 164 output tests PASS, links 0 broken, source coverage PASS.
  Source coverage — структурная проверка, не свидетельство полноты курса.
- Scoped git diff --check PASS.

Логи имеют префикс `/var/tmp/bookvar-selection-engineering-`:
unit, build-final, affected-final, output-final, links-final, coverage-final,
examples-final, дата 2026-10-28.

Из принятых MAE/42 изменены только navigation edges. Обратная замена в памяти
восстанавливает принятые SHA, что отдельно проверяет новый test.
Текущие SHA после footer changes:
MAE `6cfa1df55e37e02515ce19faefd57f77dcaa9aed2cb4bb913b199b956cc83b64`;
42 `c200aa127c6ce1ebb37cf95b34faa82479ae601d1b43579f591247348f8e6527`.

## Остаток общей задачи

H-V1-09/H-F1-09 остаются re-review. Следующий шаг — сопоставить source
2388–2601 с существующей 53 и augmentation readers, затем закрыть явную
карту Decision Framework/distillation и individual native/generated figure
decisions. Не перечитывать источник снова с начала и не повторять готовые
PPO/DPO/RLVR/GRPO, SimCLR, MoCo, MAE или новую 41i без конкретного замечания.

Остальные Harvard/HSE/EDLS units, атлас, source/rights и отложенный
EN-паритет остаются в общей очереди. Staging, commit, push, deploy,
EN-body edits, делегирование и live infra в этом проходе не выполнялись.
