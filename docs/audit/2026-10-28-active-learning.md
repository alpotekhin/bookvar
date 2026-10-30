# Active learning: explanation, source reuse and screen checks

2026-10-28. Новая глава
`00 Учебник/11 Pre-training и Scaling/41c Active learning — запрос меток и выбор примеров.md`:
545 строк, SHA-256
`c1970f259e599a045fb49e6f47745a80e7e4d7adca764e4ff500b1bec7862f81`.
Маршрут: `/textbook/pretraining/active-learning/`.

Принимается изложение active learning и его локальное отображение, не вся
H-V1-09, геометрический coreset или весь учебник. Публикация не выполнялась.

## Источники и полное чтение

Пакет `05 Источники/Originals/active-learning-2026-10-28/README.md` сохраняет
pins, точные границы чтения и source findings. В этом проходе:

- Harvard QMD `2bd97c5`: 1507–2240, включая TikZ и economics/plot code;
  cumulative 1–2240. Следующий курсор 2241, внутри economics of amortization.
  Semi/self-supervised sections прочитаны, но не объявлены интегрированными.
- Settles 2010 survey: выбранные §§2.1–2.3, 3.1–3.3, 6.1–6.3, 6.7 целиком.
  PDF SHA `4ad9785751a840f794ac4c81996d3cf45a61c7326a5545a6625a1be912b170bf`.
  Схемы scenarios, uncertainty simplex и version spaces осмотрены на физических
  страницах 11, 16, 18; не воспроизведены под выдуманной открытой лицензией.
- Sener–Savarese `1708.00489v4`: §§1–4.4 целиком, §5 только начало.
  PDF SHA `93562d0894759407c0ed795fbc58e9d270967e789be787e2ef85f777ee008482`.
  Figures 1–2 осмотрены на страницах 5 и 7. Poppler реконструировал xref при чтении;
  исходный PDF не менялся. Не перенесены обобщения CNN results на любые deep models.
- modAL pin `bba6f6fd00dbb862b1e09259b78caf6cffa2e755`: весь MIT LICENSE,
  все 192 строки overview RST, 157 строк disagreement RST, все markdown/code cells
  uncertainty notebook, включая plotting code. Четыре кандидата-рисунка осмотрены;
  дополнительно просмотрен motivating-example, выбран вместе с workflow.
- Google `kcenter_greedy.py`, все 123 строки, прочитан как будущий geometry
  reference. Код не импортировался, геометрическая глава ещё не готова.
- Вся итоговая глава прочитана после содержательных дополнений; последние
  две редакторские правки убрали повторный вводный пример и уточнили unweighted
  evaluation. Они перечитаны отдельно. Код и иллюстрации при этом не менялись.

## Сопоставление объяснений

| Механизм | Источник | Пример и сохранённые границы |
| --- | --- | --- |
| Pool/stream/synthesis | Settles §§2.1–2.3; Harvard AL; modAL overview | L/U/Q, accepted labels, pending items, retraining policy; 200/10000/100 — условная задача, не benchmark |
| Least confidence, margin, entropy | Settles §3.1; modAL notebook | Исходная трёхклассовая матрица: U выбирает 2, margin — 3, H — 2; бинарная эквивалентность, calibration, 0 log 0 |
| Query by committee | Settles §3.2; modAL disagreement | Version-space motivation и её ensemble approximation; пять исходных vote rows, равные/противоположные вероятностные committees |
| Consensus vs disagreement | Settles average KL; modAL consensus description | Оба mean predictions (0.5,0.5), но average KL 0/0.6371456; дано определение KL, арифметика и entropy identity |
| Expected gradient length | Settles §3.3 | Unknown-label expectation → logistic derivation → реальные p=0.5/0.9 и G=0.5/0.9; feature magnitude, convergence approximation, cost |
| Annotation round | Settles pool setting; собственный полный CPU пример | Fit → score current pool → oracle for selected ID only → move ID; два раунда, no duplicate request, stable IDs |
| Batch diversity | Settles §6.1; Sener–Savarese §§3–4 | Сто похожих документов, перекрытие информации и latency; k-center упомянут как другой objective, не заменён строчкой определения в готовой geometry chapter |
| Evaluation/cost/oracles | Settles §§6.2–6.3,6.7 | 95% traffic accuracy vs 50% hard-only query accuracy; 100×2 vs 60×5 minutes; absent classes, noisy oracle, independent test, budget stopping |

Два source passages и числовые примеры modAL сохранены на английском.
В margin definition явно объявлена замена символа predicted class с x-hat
на y-hat; не скрыто под заявлением о буквальной копии. Vote explanation
воспроизведено без стилистической переработки. Матрица и `partition` calculation
проверены как полные byte-exact spans notebook cells 3 и 12.

Неправильная нормировка первого probability row в pinned modAL consensus example
не перенесена как рабочий input; source warning объясняет проверку 0.9 != 1.
Entropy не описана как expected guess count. Из Harvard не импортированы
assigned 20x speedup, 4x illustrative curve, неаргументированное исчезновение
редких pathology при random sampling и обещание каждого раунда улучшать модель.
Не создавался фиктивный benchmark или новый AL learning curve.

## Фактическое исполнение

`publishing/tools/check_active_learning_examples.py` исполнил все 5 actual
Python-программ в отдельных namespaces. Затем проверены:

- 1001 binary probability points: three-score equivalence и entropy monotonicity;
- 90 normalized committees: прямой KL, entropy identity, bounds, identical members;
- 8 invalid committee cases: empty, wrong shape, negative, nonfinite, unnormalized;
- 60 independently differentiated EGL cases, dimensions 1/2/4;
- exact source notebook spans, vote distributions и source example outputs;
- actual annotation trace: query 101 → label 0 → L/U=3/3;
  query 205 → label 1 → L/U=4/2; no unseen-label access before query,
  no duplicate oracle calls; fit after both updates;
- evaluation-mix and annotation-time arithmetic.

NumPy 2.3.5, torch 2.14.1, CPU. Финальный лог с SHA:
`/var/tmp/bookvar-active-learning-examples-accepted-2026-10-28.log`.
Это toy logistic training, а не trained-ensemble benchmark. Нет modAL/GPU
запуска, pretrained weights, dataset download или внешней разметки.
Google/Gurobi/MIP, primary-paper experiments и active-learning superiority
локально не воспроизводились и не заявлены воспроизведёнными.

## Иллюстрации и происхождение

`00 Учебник/Assets/Figures/curated/active-learning-2026-10-28/`:

- `active-learning.png`: 858×302, SHA
  `b39fc6c587bcc5f6a5914ba36d49721bec155cff06baca58fd527211ff73f732`;
- `motivating-example.png`: 3000×3000, SHA
  `5d9c6acd6dadcd8eb88f08500a4612b0cf06d78a48378226456be4efdfc8af9e`.

Полные оригиналы modAL, byte-for-byte; без crop, redraw, translation или recompression.
Исходники, pinned URLs, dimensions, source/license/asset hashes согласованы
в assets.json и центральном registry; NOTICE и полный MIT text доступны с сайта.
License SHA `e9ad97986d03270bf7fdc39826ccfec4cc48357837500c8706a20157f2706c38`.
Оба рисунка объяснены рядом с формальными определениями и статусом illustration,
а не приписаны измеренному ускорению. Нечёткие gray maps с недостаточным
intensity legend остались reference-only, не опубликованы ради количества.
Открытый доступ Settles/Sener не заменён разрешением на перенос их figures.

## Стиль, навигация и UI

No-ai-slop/eval: убрано вступление о том, чему «следует глава», повторная
абстрактная подводка к evaluation и редакционная самопрезентация.
Сохранены связные выводы, реальные source examples, limitations и целый loop;
лицензированные excerpts не переписывались для стиля. Проверка стиля не подменяет
приведённое выше сравнение с источниками и самостоятельные вычисления.

Переход 41b → 41c → 42 внесён в navigation/sidebar/index и footer edges.
Регрессионный тест обратной заменой только next link восстанавливает принятый
SHA 41b `e9b361…def2`; её объяснения/код/рисунки повторно не редактировались.
Фактические footer clicks 41c → 41b → 41c открыли нужные страницы.
H-V1-09 остаётся re-review, notes теперь указывают обе реальные главы и source cursor.

Фактический IAB QA:

- desktop 1280: оба изображения загружены и осмотрены; inline aspect ratios
  соответствуют исходникам (632×222.45 и 632×632), без file crop;
- mobile 360: body 360/360, все 8 display equations 328/328, errors 0;
  проверка bounding boxes inline math не нашла обрезанных формул;
- workflow viewer 2.5x: pan 0,180,516 из 516; height/scrollHeight 611/611,
  caption до конца, MIT/source links доступны;
- probability figure viewer 3x: width/scrollWidth 360/1048, x=0/688;
  height/scrollHeight 611/1048, y=0/437; легенда, оба нижних угла и оси осмотрены;
  подпись прокручена до конца;
- EN route открыт: видимый explicit fallback, не новая EN translation;
- viewport reset, main preview tab 5 отмечен deliverable; source gallery не запускался.

Screenshots: `screenshots/2026-10-28-active-learning-*` (desktop workflow/plot,
mobile workflow edges, probability legend/bottom, EGL formula).

## Проверки и остаток

607 unit / 92 files PASS; после финальных prose-only corrections повторены
6 affected tests. Финальная сборка 1973 страниц — 23.30s;
159 output tests — 4.39s; links — 0 broken. Source coverage PASS, 210 rows /
59 primary destinations: это structural check, не доказательство всей интеграции.
Логи `/var/tmp/bookvar-active-learning-{build,output,links,examples,scoped}-accepted-2026-10-28.log`,
unit-current и coverage-final. Scoped diff-check прошёл при финальной передаче.

Открыты полноценная geometry/coreset chapter (включая farthest-first proof,
outliers, representations и herding), semi/self-supervised reconciliation,
source 2241–4741 и native figure inventory. Текущая глава не закрывает их.
Готовые RL/Training/synthetic curricula не переписывать; EN bodies,
stage/commit/push/deploy и делегирование не выполнялись.
