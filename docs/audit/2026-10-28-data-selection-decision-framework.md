# Data Selection: выбор метода и native figures

## Результат и границы

Предыдущий ответ о готовом RL был статусной сверкой, не продвижением содержания.
В этом проходе дополнена существующая глава 41i:
`00 Учебник/11 Pre-training и Scaling/41i Отбор данных — стоимость, шарды и проверка качества.md`.
Итог — 820 строк, SHA-256
`b4ecbaa30f6367ae38d9a4e0cdfa0d89c8b17957073be3fed3a9b9d583a09009`.
Маршрут: `/textbook/pretraining/selection-engineering-evaluation/#selection-method-choice`.

Добавлены выбор по ограничениям, оригинальная английская таблица предпосылок,
неизменённое дерево Harvard и разобранный выбор при двух ограничениях качества.
Прежняя инженерная часть сохранена: удаление только нового раздела возвращает
SHA `b0e6e4621d64b1583c7522a3bb5cdc51b6f4ddaed461c7ebe1df0ad188e2feaf`.
Это проверяет сохранность текста, а не заменяет его содержательное чтение.

Полнота Harvard H-V1-09/H-F1-09 и всего учебника не объявлена. Полное объяснение
GAN и индивидуальные решения для generated QMD figures ещё нужны. RL, EN bodies,
stage/commit/push/deploy, делегирование и live infra не затронуты.

## Источники и сопоставление объяснений

Полностью прочитаны исходные 691 строка 41i и итоговые 820 строк. Финальная
новая часть перечитана после удаления лишней редакционной фразы перед таблицей.
Harvard QMD 2602–2760 прочитан целиком: knowledge distillation, обе сводные
таблицы, полный TikZ дерева, prerequisites, ROI, combination и переход к
Selection Engineering. Весь QMD 1–4741 уже был прочитан в предыдущих проходах;
повторное чтение всей книги не заявляется.

| Механизм | Источник | Проверенное объяснение |
| --- | --- | --- |
| Binding constraint и доступность метода | Decision Framework, 2616–2760 | Запрос меток требует oracle, coreset не создаёт отсутствующий класс; таблица prerequisites сохранена на английском |
| Ограничения дерева | Полный TikZ и original web Figure 12 | Нет oracle не означает готовый SSL; unlabelled pool не оплачивает pretraining; low redundancy не доказывает полезность curriculum |
| Разные цели distillation | Source 2602–2615 и существующие 08/57b | Teacher probabilities и generated sequences ведут к разным целям; повышенная entropy сама по себе не доказывает полезность |
| Допустимость по качеству перед минимизацией затрат | Decision Framework и Measurement Framework; новый условный пример | При порогах overall .90 / rare .80 вариант за 45h отклонён (.91/.60), за 60h принят (.905/.84); baseline 100h тоже допустим |
| Privacy | Сверка source constraint table и уже принятой 53a | Synthetic не считается privacy guarantee; универсальная рекомендация из таблицы не перенесена |
| Полная стоимость и комбинации | Прежнее принятое изложение 41i | Preparation, recurring scoring, I/O, fixed cost и quality gates сохранены; gains не объявлены автоматически мультипликативными |

Архивная таблица prerequisites воспроизведена под CC BY-NC-SA 4.0 с удалением
только index commands и типографического выделения. Живая веб-версия рисунка
имеет отдельный CC BY-NC-ND 4.0; она не перелицензирована старой лицензией QMD.
Диапазоны generic gains из исходной сводной таблицы не объявлены эмпирическими
гарантиями. Все числа нового примера явно hypothetical.

Для distillation повторная глава не создана. Текущие SHA совпадают с ранее
принятыми редакциями:

- 08 Reasoning distillation —
  `e7ea39b04663903565f4e9796adc5718ab1f7c3a23dee13d0c768faaba3bba4c`,
  отчёт `2026-10-25-distillation.md`.
- 57b Model compression —
  `e818fccc317a2025b020f453a2e578e43ea1a2ba39ae893a86c6ee0a89bed5d3`,
  отчёт `2026-10-19-model-compression.md`.

В этом проходе сопоставлены их необходимые distillation-фрагменты; новая
полная приёмка обоих неизменённых текстов не заявляется. Новые читательские
ссылки на обе главы проверены реальными кликами в preview.

## Полный native inventory

Индивидуальный машиночитаемый журнал:
`05 Источники/Originals/data-selection-reconciliation-2026-10-28/native-figure-decisions.json`.
Все 14 файлов (6 SVG, 6 PNG, 2 WebP), 6,684,684 bytes, лично просмотрены
по отдельности. Все Git blob SHA-1 совпали с закреплёнными GitHub API listings
при `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`; SHA-256 записаны отдельно.
Совпадение хеша не выдано за визуальную проверку.

Шесть margin SVG не приняты как новые смысловые рисунки: нет нужных осей,
единиц, условий или механизма. PNG frontier повторяет неподписанный мотив.
Обложки различены по композиции и формату: PNG blueprint не тождественен
позднему print/WebP. Они не добавлены в повествование ради счётчика картинок.
Существующая labelled WebP в source landing note сохранена.
`running_out_of_data.png` оставлен link-only: исторические точки и
экстраполяция не доказывают фактическую дату исчерпания или диапазон доступных
данных без модели оценки stock. Для каждого файла в JSON есть отдельная причина.

SVG временно растеризованы Sharp/libvips для осмотра. Это QA, а не замена
исходников. Ни один native/source файл не перерисован или обрезан.
Данная проверка закрывает native-directory disposition, не generated figures.

## Новый читательский рисунок

Полная original web Figure 12 скопирована byte-for-byte:
`00 Учебник/Assets/Figures/curated/selection-engineering-2026-10-28/selection-decision-tree.svg`.
SHA `ab3d0407cfbae771637bd61169b783284b6c2cc3b502db7e846b79cad6eacd63`,
viewBox `[0,0,761.21869,352.48932]`.
В curated assets.json и общем registry одинаковая запись с прямым URL,
исходным SVG, referring HTML/hash и действующей CC BY-NC-ND 4.0.
В главе теперь четыре оригинальных web-рисунка. Собственные SVG/Mermaid
и перевод подписей не создавались.

## Проверка примеров, стиля и публикации

`python3 publishing/tools/check_selection_engineering_examples.py` выполнил
все девять actual reader programs на итоговом SHA. Проверены 108 отдельных
вариантов quality/prerequisite gates, пустой допустимый набор и провал overall
threshold. Прежние проверки сохранены: 450 refresh cases, 12 break-even,
360 distributed top-k, 48 finite echo schedules и 24 metric cases.
Это CPU arithmetic, не новый training/hardware benchmark.

No-ai-slop SKILL.md и eval.md прочитаны целиком. После редактуры новая часть
проверена по checklist: нет generic opener, staged revelation, vague attribution,
выдуманного эмпирического улучшения или перечня определений вместо механизма.
Технические ограничения и полезные сравнения сохранены по правилам AGENTS.
Оригинальная таблица не переписана ради стиля.

- 154 affected unit tests PASS; после последней правки подписи/прозы ещё
  5 selection-engineering tests PASS на итоговом файле.
- Final build: 1989 pages, 27.18 s, exit 0.
- 164 site-output tests PASS.
- Built links: 0 broken routes, fragments, files.
- Source-coverage structural check PASS: это наличие связей, не доказательство
  содержательной полноты 210 units или всего курса.
- Scoped `git diff --check` PASS.
- Полный test:unit: 633 tests / 100 files PASS, 52.99 s, exit 0.
  В проверках прежних PDF остались non-fatal Poppler closepath warnings;
  они не относятся к новому SVG и не объявлены исправленными этим проходом.

Реально осмотрены desktop 1280 и mobile 360: четыре таблицы и четыре рисунка
помещаются, все восемь display-формул без KaTeX errors и page overflow.
Новая таблица prerequisites и числовая таблица отдельно просмотрены при 360 px.
В новом viewer 4× текст векторный и читаемый. На обеих ширинах достигнуты
точные крайние scroll positions: desktop `(3129,1502)` и `(0,0)`;
mobile `(1032,42)` и `(0,0)`. Полная подпись/автор/источник/лицензия доступны.
Непроверенные старые viewer не выданы за повторно принятые.

Английский маршрут осмотрен: сохраняет явную fallback-пометку, новый EN body
не создавался. Footer targets остались MAE → 41i → Next-token objective;
их сохранность отдельно проверяет regression с восстановлением соседних SHA.
Screenshot: `/var/tmp/bookvar-selection-method-choice-2026-10-28.png`.
Tab 5 отмечен deliverable; временный viewport сброшен.

## Следующая работа

Harvard 2480 лишь называет GAN; отдельное упоминание GAN-loss в MAE не заменяет
объяснение обучения генератора и дискриминатора. Далее подобрать и прочитать
полное оригинальное изложение GAN/DCGAN с хорошими авторскими рисунками,
сопоставить его с текущим учебником и встроить недостающее. Затем завершить
индивидуальные решения для generated QMD figures и секционную карту H-V1-09.
Готовые RL/Training/53/60/64c/SimCLR/MoCo/MAE/41i/06a/53a без конкретного
нового пробела в повторную очередь не возвращать. Общая цель активна.
