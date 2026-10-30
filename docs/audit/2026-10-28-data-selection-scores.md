# Data selection: EL2N, GraNd, forgetting

2026-10-28. Каноническая новая глава:
`00 Учебник/11 Pre-training и Scaling/41b Отбор обучающих примеров — EL2N, GraNd и forgetting.md`.
615 строк, SHA-256
`e9b361be8d70274f446dd3f2c7607c96e2c26b3cf74862942e5db5d3bc05def2`.
Маршрут: `/textbook/pretraining/data-selection-scores/`.

Изложение, примеры и рисунки этой главы проверены. Это не приёмка всей
H-V1-09 Data Selection и не завершение учебника. H-V1-09 и H-F1-09 переведены
из прежнего `audited` в `re-review` с конкретной границей оставшейся работы.
Готовые PPO/DPO/RLVR/GRPO не переписывались; их SHA совпали с приёмкой 24–25 октября.

## Что прочитано

- Harvard, закреплённый QMD `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`:
  строки 1–1506 включительно, в том числе все TikZ в этом диапазоне.
  Файл 4741 строка, SHA
  `7bdfce0af42c461eb4277da4f61f3a18442af22360b65ce57a8073fef7de8f5c`.
  Следующее чтение начинается с Active learning, строка 1507.
- Глава 53: все 947 строк, SHA
  `8b162e5380772ff0066fc7241061f4d6af8aa7e536d1b0172a753211bf6e855e`.
  Синтетическая генерация, смеси, curriculum ordering и model collapse уже
  объяснены. Глава не изменена; новый дубль этих разделов не создан.
- Data Diet `2107.07075v2`: §§1–4 целиком, не весь paper. Выбрана исправленная
  версия 28 марта 2023, а не первоначальное утверждение о GraNd at initialization.
- Kirsch `2303.14753v1`: весь paper, включая приложения и библиографию.
- `mansheej/data_diet`, pin `aee462eab93aa0fb14c1ddfeba45d645b85a5fbb`:
  целиком README, LICENSE, `scores.py`, `gradients.py`, `forgetting.py`,
  `scripts/get_mean_score.py`. Остальные скачанные файлы не объявлены прочитанными.
- `BlackHC/pytorch_datadiet`, pin `fda604dc34eade5918a2c98568c2dc0f7861ced3`:
  целиком README, LICENSE.md и 326 строк `data_diet_repro.py`; оба опубликованных
  исходных рисунка осмотрены отдельно и в читательской странице.
- Итоговая глава прочитана целиком, 1–615, после содержательных и мобильных правок.

Архив: `05 Источники/Originals/data-selection-2026-10-28/`.
Data Diet PDF SHA `842b7d259d98e6c900d87f730d08b689f347df70d764bf72e4a503daca213058`;
Kirsch PDF SHA `4327a29b97b6ed02162e186603ae03a099860fed136f5003413b0d65d78de2f4`.

## Приёмка объяснений

| Механизм | Прочитанный источник | Промежуточные шаги и проверенный случай |
| --- | --- | --- |
| Labelled pruning vs acquisition | Data Diet §§1–3; Harvard 1131–1355 | Пул с известными метками, бюджет m, score → selected IDs → fresh target run; отдельно эпохи и optimizer steps |
| EL2N | Data Diet §2.2, reference `scores.py` и `get_mean_score.py` | CE → p−y → L2; A/B/C показывают уверенную ошибку с низкой энтропией; R=2 показывает mean(norm) ≠ norm(mean) |
| GraNd | Data Diet §§2.1–2.2; Kirsch §2.1 | Jacobian chain rule; W=0 и признаки h=(1,0)/(10,0); равный EL2N и десятикратная разница norm(grad) |
| Influence bound | Data Diet Lemma 2.2 и оговорки | Суммированный SGD, first-order target loss, Cauchy–Schwarz; одинаковые нормы и противоположные направления дают −0.1/+0.1, вместе ноль; перенормировка mean-batch оговорена |
| Forgetting | Data Diet §2.3 и исходный `forgetting.py` | Четыре наблюдения: counts [1,0,1], never-correct ID 1 отдельно; его exported score ∞, не easy/unforgettable |
| Stable IDs | Harvard Listing 1, строки 1317–1352; собственный проверяемый пример | Loader C/A/B; неверный positional join выбирает 101, правильный ID join — 55; отдельные eval/inference и GraNd boundaries |
| Проверка результата | Data Diet §§3–4; Kirsch §§2–3 и полный reproduction script | Разделены preliminary/target seeds, trained/initial scores, random same-size control, ошибочный checkpoint zero; качество не выведено из корреляции |
| Noise/coverage | Data Diet §4; Harvard guardrail paragraph 1137 | Высокая ошибка не доказывает неверную метку; условный minority-class пример, validation/test separation, квоты и sampling distribution |
| Cost amortization | Собственный вывод, не непрочитанный раздел 3230+ Harvard | C_P+K C_S против K C_F; (100,40,180): 220/100, 300/300, 340/400; assigned ledger, не измеренное ускорение |

Не перенесены необоснованные общие утверждения закреплённого курса:
EL2N как unlabeled uncertainty, «все высокие scores около boundary»,
безусловное превосходство gradient methods, гарантированный proxy transfer,
«10× faster» из выбора 10% данных. Coreset geometry и её гарантии не выданы
за автоматически доказанную neural-network accuracy guarantee.

Проблема Listing 1 условная: без привязки к ID позиции scores соответствуют
dataset positions только при согласованном порядке scoring loader. Не написано,
что любой loader неверен. Современная вебверсия уже уточняет error/coverage и
unshuffled-order condition: findings относятся к архивному pin, не автоматически
к текущему upstream. Старый `/contents/vol1/...` URL дал HTTP 404; в главе стоит
проверенный HTTP 200 `https://mlsysbook.ai/vol1/09_data_selection/09_data_selection.html`.

У Kirsch Eq. 2 напечатан squared norm; для определения EL2N использованы
Definition 2.3 первичного v2 и код с `ord=2`, а не эта запись. Исходник не исправлялся.

## Фактическое исполнение

`publishing/tools/check_data_selection_examples.py` исполнил все пять actual
Python-блоков в независимых пространствах имён, а затем независимые проверки:

- 300 вероятностных случаев, диапазон EL2N и autograd CE = p−y;
- 40 линейных голов, outer-product gradient и norm factorization;
- 32 полные correctness histories, forgetting transitions и never-correct;
- 24 сочетания перестановки IDs и batch size, включая неполный хвост;
- 5 ошибочных ID/score cases, missing/unknown/duplicate/nonfinite, восстановление mode;
- точные исходные spans обоих воспроизведённых forgetting helpers;
- противоположные градиенты и ledger из фактических блоков главы.

NumPy 2.3.5, PyTorch 2.14.1, CPU. Финальный лог с SHA главы:
`/var/tmp/bookvar-data-selection-examples-final-layout-2026-10-28.log`.
Нет предварительного обучения, весов, CIFAR downloads, GPU, JAX/Flax запуска
или воспроизведённого performance benchmark. Опубликованные кривые — результат
автора, а не локального запуска. JAX training script и reproduction training
pipeline не исполнялись.

## Рисунки, оригиналы и права

В `Assets/Figures/curated/data-selection-2026-10-28/` два полных оригинала:

- `figure11_repro_cifar10_hlb.svg`, SHA
  `7c735a5bf5b4c888702a4ece7c3f2c005dddd94e1a6635d85d890c2a38e19710`;
  viewBox [0,0,496.005937,309.87588]. Это repository version, не идентичный
  crop более раннего Figure 2(b). Тип shaded interval не приписан неподтверждённым
  defaults Seaborn. Проверены весь script и его `sns.lineplot` без параметра interval.
- `joost_train_input_norm_vs_grad_norm_sorted.png`, SHA
  `2124a93db5fae6ce108e4b3dcf372b1a9d19cccde4b1b4947c1e3e91fe10b810`;
  2051×1354, original rank correlation 0.97. Ранги не выданы за качество pruning.

Byte-for-byte copies: без crop, redraw, перевода подписей или recompression.
Для каждого — источник, pin, author, source/asset/license SHA, adjacent attribution,
NOTICE и запись центрального registry. Apache-2.0 обоих репозиториев проверена;
две публичные копии лицензии распространяются рядом с кодом/рисунками.
Исходные два helper function spans сохранены без стилистической переработки.

Разрешение на короткую цитату Kirsch §3 — CC BY 4.0; evidence:
`reproduction/arxiv-abs-v1.html`, SHA
`8c93cfaf437bc8fcf2d95c2ab955425a5683d4501d02091137bbe10849023c3b`.
Удалён только footnote marker, соответствующий Flax commit сохранён ссылкой.
Harvard guardrail paragraph перенесён в оригинальном English, CC BY-NC-SA 4.0.
Не смешаны Apache кода, CC BY reproduction paper и nonexclusive arXiv license
первичного Data Diet. Рисунки последнего без подтверждённого разрешения не копировались.

## Стиль, навигация и экран

No-ai-slop и eval применены к полному итоговому изложению отдельно от приёмки
объяснений. Сохранены выводы, числовые промежуточные шаги, оговорки и полный
исходный код. Английское изложение не переведено; лицензированные excerpts
не переписаны для стиля. Новая chapter source_language=en не объявлена готовой EN translation.

Глава вставлена в navigation.yml, sidebar и `_index.md`; переходы 41a → 41b → 42
согласованы. В соседях изменены только footer edges. Обратная замена ссылки 41a
в памяти восстановила принятый SHA `b85e4a7a…929ae`, без изменения файла.
Два source-unit rows отражают частичный review и не считают structural coverage
доказательством полноты обучения.

Фактический IAB QA, основной tab 5:

- desktop 1280: оба полных inline images и captions осмотрены;
- mobile 360: body 360/360; 11 display equations по 328/328, KaTeX errors 0;
- обе таблицы в финале 328/328. Первоначально скрытый столбец устранён краткими
  математическими заголовками и явно указанным округлением таблицы; код печатает 6 decimals;
- SVG viewer 2.5×: pan 0/516 и 516/516, viewport height/scrollHeight 611/611;
  вся легенда и оба края доступны, caption отдельно прокручена до 147/147;
- PNG viewer 2×: оба края, pan до 344/344, height/scrollHeight 611/611;
- EN route открыт фактически: видимый explicit fallback, не готовый перевод;
- временный source gallery 4331, session 55071, остановлен с terminal exit 0;
  viewport reset, основной reader preview оставлен. Посторонние вкладки не закрывались.

Screenshots: `screenshots/2026-10-28-data-selection-*` — desktop SVG/PNG,
mobile left/right обоих viewer, таблица и исправленная стоимость.

## Проверки публикации и оставшийся объём

604 unit tests / 91 files прошли; после последних локальных layout-правок
повторены 3 affected tests. Финальная сборка 1971 страниц — 24.87s;
158 output tests — 4.32s; links — 0 broken. Source coverage прошёл после
исправления формата destination cell (210 rows / 59 destinations).
Scoped diff-check прошёл. Логи:
`/var/tmp/bookvar-data-selection-{build,output,links}-final-layout-2026-10-28.log`,
`...-unit-accepted-...`, `...-scoped-final-layout-...`, `...-coverage-final-...`.

Открыто: source lines 1507–4741, геометрический отбор, active learning,
semi/self-supervised selection, engineering/distributed/cost comparison,
индивидуальный native-figure inventory. Это следующие реальные работы,
а не повторная приёмка готового RL или переписывание главы 53.
Stage/commit/push/deploy, EN bodies и делегирование не выполнялись.
