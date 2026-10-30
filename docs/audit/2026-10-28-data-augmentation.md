# Аугментация данных: объяснения, оригиналы и проверка страницы

Дата: 2026-10-28. Checkout `editorial-route-alignment`, без stage/commit/push/deploy.

Глава: `00 Учебник/01 Основы нейронных сетей/06a Аугментация данных — Mixup, CutMix и политики преобразований.md`.
Маршрут: `/textbook/neural-networks/data-augmentation/`.
Итог: 697 строк, SHA-256
`d35a256a67948d010ba6a0e3feae9a3d3d2c7856d47be695f9570c8750d3ddd7`.

Исходный 656-строчный текст прочитан целиком, после добавления оригинальных
фрагментов повторно прочитаны все 698 строк. Затем удалены два повторяющих
следующий абзац предложения и уточнён множитель шума в примере KWS;
эти итоговые фрагменты перечитаны отдельно. Примеры повторно выполнены
на указанном итоговом SHA, сайт собран после последних правок.

## Проверка объяснений по источникам

| Механизм | Прочитанные источники и сохранённые шаги |
|---|---|
| Преобразование входа и метки | D2L Image Augmentation прочитан целиком, 464 строки, оба framework-варианта. Разделены сохранение класса, преобразование координат и смешивание меток. Пример отражения связывает half-open box с пиксельной маской; crop/clip и интерполяция категориальных масок объяснены отдельно. |
| Случайные crops | Сохранены оригинальный абзац D2L, helper `apply`, PyTorch RandomResizedCrop и весь восьмипанельный вывод. Площадь crop отличена от длины стороны, shape от содержания, галерея от последующего training recipe. Общая фраза источника про uniform sampling не перенесена как утверждение о распределении aspect ratio в PyTorch. |
| Средняя augmented loss | Определено ожидание по преобразованию пары. Два входа дают predictions 1 и 3, среднюю MSE 1, loss среднего 0 и gradient (2,−2). Число производных файлов не названо числом независимых наблюдений. |
| Cutout и заполнение | Оригинальная Figure 4 SimCLR связывает crop, цвет, размытие и маски. Разобраны contiguous input mask, отличие от dropout и единицы заполнения до/после нормализации: при mean .4/std .2 значения 0 и −2 различаются. |
| Mixup | Zhang et al., 1710.09412v2, §2 прочитан целиком. Сохранены vicinal distribution, Beta(α,α), permutation/fixed points, input и target mixtures. Показаны мягкая CE, линейность по target и общий forward. Для p=(.8,.2), q=(.25,.75) loss 1.262864, gradient (.55,−.55); hard argmax меняет loss на 1.609438. |
| CutMix | Yun et al., 1905.04899v1, §§3.1–3.2, protocol/Table 3; полный README и все 409 строк train.py прочитаны. Строки 230–240 сохранены без изменений. Отдельный CPU helper использует BCHW и реальную площадь после rounding/clipping. На 4×4 номинальная .5 превращается в .75 в центре и 15/16 в углу. Площадь не объявлена точным измерением семантической массы. |
| Полный update | Используется preceding CutMix helper; mixed target имеет размерность logits и сумму 1. Проверены forward, soft CE = weighted hard CE, backward, ненулевой конечный gradient и SGD step. Fixed-batch loss 1.200975 → 1.190955 не выдан за validation gain. |
| AutoAugment | 1805.09501v3, §3 целиком. Отличены search candidate с пятью sub-policies и final concatenation из 25, probabilities/magnitudes, child validation и PPO controller. Пример вероятности отсутствия вызова .365 разобран по ветвям. Исторические GPU-hours не названы универсальной текущей ценой. |
| RandAugment | 1909.13719v2, §3 целиком и proxy-task motivation. Показаны pool14, N/M, replacement и 196 возможных последовательностей при N=2; это не 196 обязательных trials. Разные единицы magnitude и риск чрезмерной силы преобразований сохранены. Самих 14 image operators локальная программа не реализует. |
| Текст и back-translation | Sennrich et al., 1511.06709v4, §§3.1–3.2 целиком. Synthetic source из real monolingual target отличён от roundtrip paraphrase. Примеры со сменой отрицания, spans и вопросом/ответом не допускают автоматической гарантии семантики. Иллюстративная языковая пара не выдана за результат запущенной MT-модели. |
| Speech/KWS и split | Harvard Data Selection 2388–2601 сопоставлен с готовой главой 53. Для 50 speakers ×10 recordings явно введены 10 transforms ×2 noise settings: 10,000 derived records, но 500 parents и 50 speakers. Код проверяет identities/split, не создаёт аудио. |
| Validation и TTA | Разобраны train/eval transformations, пределы model.eval(), обратное преобразование пространственных predictions, вероятность против logits. Пример (.9,.1),(.6,.4) даёт arithmetic (.75,.25), geometric-normalized (.7861,.2139). Выбор протокола отделён от финального test. |

Полный Harvard QMD уже прочитан в предыдущих проходах (4741 строка),
но его unit не объявлен полностью перенесённым. Непроверенные заявления
«маленькие модели всегда сильнее переобучаются», «synonym replacement
сохраняет смысл», «synthetic автоматически обеспечивает privacy» не импортированы.

## Стиль и связность

Применены `no-ai-slop/SKILL.md` и `eval.md` с местными академическими
оговорками. Вход главы начинается с cat/left-arrow target semantics;
формулы связаны с числовыми примерами. Не добавлены общие вступления
о значимости технологии или новые авторские схемы. Содержательные
контрасты и ограничения сохранены. Из заключения убран служебный
комментарий о подготовке текста; рядом с примерами указаны зависимости.

Глава стоит после Dropout и перед CNN, с дальнейшими тематическими ссылками
на pipeline, selection engineering, FixMatch, SimCLR и synthetic data.
Текст остаётся английским; `en/` bodies не редактировались. `/en/` показывает
явное предупреждение об отсутствии локализованной версии, а не считается
готовым переводом.

## Рисунки и происхождение

Два новых original assets и один уже существовавший:

- D2L: полный SVG с восемью встроенными raster-панелями,
  viewBox 0 0 465.600016 237.203471, CC BY-SA 4.0. Это сохранённый вывод
  книги, не наш повторный запуск и не новый результат. Естественная
  детализация его небольших исходных фотографий сохранена; увеличением
  не заявлено восстановление отсутствующих деталей.
- NAVER CutMix: целый img1.PNG, 649×447, MIT, byte-for-byte. Это overview
  репозитория, а не CAM Figure 1 статьи. Нижние строки показывают разные
  исторические задачи и метрики. Сохранены исходные подписи и числа;
  более поздний результат в README не подставлен в старый рисунок.
- SimCLR: существующая полная Figure 4, 2690×1135, CC BY 4.0.
  Credit Von.grzanka/CC-BY сохранён в подписи и viewer. Новый дубликат
  файла не создан; в adjacent metadata и central registry добавлен used_in.

Новые `assets.json`, `ATTRIBUTION.md` и source packet README содержат
точные upstream URL/commits, снимки страниц, лицензии и SHA.
Оригинальные license notices отдельно доступны в `site/public/licenses/`.
Два новых оригинала не обрезаны, не перерисованы, не переведены
и не перекодированы. SimCLR ранее извлечён из PDF как полный рисунок;
его существующий crop/provenance сохранён, новых преобразований нет.
Архив paper HTML используется для фактической сверки, не как заявление
о разрешении копировать все изображения этих papers.

## Выполнение и независимые случаи

Команда использует существующее локальное окружение PyTorch, без установки
пакетов, сети, загрузки датасета или GPU:

```sh
/Users/aleksan.potekhin/.cache/uv/environments-v2/035f72b8e8383b33/186c9c62a44ded93/bin/python publishing/tools/check_data_augmentation_examples.py
```

PASS на итоговом SHA: девять reader programs, 630 half-open boxes,
12 augmented objectives, 372 Mixup CE/gradient cases, 405 CutMix cases,
8 исходов выбора sub-policy и 9 TTA случаев. Проверены nonsquare/odd/even,
углы/края/центр, nominal λ0/1, self-pairing и неизменность родителей.
PNG декодируется с проверкой CRC/zlib; SVG содержит восемь внутренних
изображений и не зависит от внешних href. Все asset/page/license hashes
и оригинальные public license copies совпадают.

Полный unit run: 99 files / 628 tests PASS. После двух последних локальных
правок главы: 3 affected files / 12 tests PASS и actual checker повторён.
Полная сборка после них: 1987 pages, 27.09 s, exit0.
164 output tests PASS; built link check — 0 broken routes/fragments/files;
structural source coverage PASS. Scoped git diff --check PASS.
Покрытие source registry не подменяет смысловую приёмку всей Harvard unit.

Логи текущего запуска:

- `/var/tmp/bookvar-augmentation-examples-2026-10-28.log`
- `/var/tmp/bookvar-augmentation-unit-2026-10-28.log`
- `/var/tmp/bookvar-augmentation-focused-2026-10-28.log`
- `/var/tmp/bookvar-augmentation-final-build-2026-10-28.log`
- `/var/tmp/bookvar-augmentation-output-2026-10-28.log`
- `/var/tmp/bookvar-augmentation-links-2026-10-28.log`
- `/var/tmp/bookvar-augmentation-coverage-2026-10-28.log`

## Проверка интерфейса

Actual preview `http://127.0.0.1:4330/`, desktop1280×900 и mobile360×800:
все три изображения осмотрены в тексте; пропорции сохранены и края
не обрезаны. На desktop ширины 465.59/631.98/631.99 px; на mobile все
328 px. Все шесть display-formulas имеют clientWidth=scrollWidth=328,
нет inline overflow/KaTeX errors; document width=scrollWidth=360.

Мелкие подписи многопанельных оригиналов требуют viewer на телефоне.
Все три viewer реально открыты, увеличены до4× и перемещены к обоим
горизонтальным краям x0↔1032. Вертикальные края: CutMix0↔352.5
(scroll range353), D2L0↔106; SimCLR при этой ширине не требует vertical pan.
Управление закрытием работает. Source attribution остаётся в viewer.

Actual footer clicks: Dropout → augmentation → CNN → augmentation.
EN fallback осмотрен: предупреждение видно, три изображения загружены,
нет формульных ошибок/переполнения, footer остаётся в /en/.
После final build страница перезагружена; новые Harvard link/noise sentence
подтверждены в DOM. CSS не изменён, оригиналы не изменены.

Proof: `/var/tmp/bookvar-data-augmentation-2026-10-28.png`.
Preview tab5 оставлен deliverable; временный viewport reset.

## Сохранённые материалы и оставшаяся работа

Dropout изменён только в трёхстрочном next-footer. Regression test обратимо
восстанавливает прежний SHA
`ca5d07f0d27f27f8143f53882014a629e0f50c936e448ed8658be64acde7863c`.
Текущий SHA с новой ссылкой
`e30be3213d51d1f0276b560422ec1ee116ef8b5112339053871636e86d5aa742`.
Текст CNN не менялся; его generated previous edge обновлён manifest.
Глава53 не изменена: SHA
`8b162e5380772ff0066fc7241061f4d6af8aa7e536d1b0172a753211bf6e855e`.
Готовый RL не редактировался повторно.

H-V1-09/H-F1-09 остаются re-review. Следующие конкретные пробелы:
simulator/synthetic images, domain randomization/adaptation и privacy;
DecisionFramework/distillation mapping2602–2760; индивидуальные решения
по native/generated figures. Общая Harvard/HSE/EDLS очередь, атлас,
source/rights и отложенный EN-паритет также не закрыты.
Эта приёмка относится к augmentation reader и его интеграции, не ко всему учебнику.
