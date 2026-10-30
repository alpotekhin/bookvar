# Синтетические изображения и sim-to-real: приёмка

Дата: 2026-10-28. Checkout `editorial-route-alignment`.
Stage/commit/push/deploy, EN-body translation, делегирование и live-инфра не выполнялись.

Глава: `00 Учебник/11 Pre-training и Scaling/53a Синтетические изображения и перенос из симуляции.md`.
Маршрут: `/textbook/pretraining/synthetic-images-sim-to-real/`.
Итог: 587 строк, SHA-256
`63368115b8d0dd1f51ba8d16950b572be3fa57b08c491953fdf175ba5c85957a`.

Черновик прочитан полностью; после правок весь итоговый текст повторно
прочитан последовательно, строки1–200,201–400,401–587. Примеры выполнены
на этом SHA. Это приёмка данной главы, не всего учебника или Harvard unit.

## Объяснения и факты

| Механизм | Источник, промежуточные шаги и проверка |
|---|---|
| Разметка из сцены | Полные README/main.py BlenderProc semantic_segmentation. Сохранены оригинальная настройка map_by, два полных multi-panel вывода и связь category/instance/name. Числовой пример объединяет две instance-маски в один класс, не выдавая RGB preview за фактический ID-map. Видимые и amodal boxes, окклюзия и отсутствие видимых pixels различены. |
| Проекция | Pinhole-пример с world point(1,2,5), camera centre(1,0,0), positive-depth camera frame, f=100 и principal point(320,240) даёт pixel(320,280). Различены transform/inverse, camera axes и visibility; формула не объявлена применимой напрямую к Blender camera-to-world без basis conversion. |
| Материалы | Полные README/main.py material_randomizer и оригинальные before/after JPEG. Исходный код сохранён. При replacement probability.5 и K=4 материалов вероятность другой identity.375, поскольку допустим повтор исходного материала. Identity не отождествлена с видимым отличием. |
| Камеры | Полные README/main.py camera_sampling, исходный loop и два оригинальных RGB/normal/depth вывода. Показана зависимость направления от point of interest, а не независимая uniform выборка ориентаций. Указаны единицы сцены/radians и новые labels для нового ракурса. |
| Randomization | Tobin1703.06907v1, §III целиком; §IV-A/B/C, TableI/II прочитаны. Сохранены приблизительный viewpoint и fixed table height, различены textures/distractors/camera/noise. Гипотеза покрытия real world не выдана за гарантию. Пример p=.001,n=1000 даёт miss.367695; для95% вероятность встретить событие требует2995 независимых draws. При p=0 новые draws не помогают. |
| Source/target risk | Ganin§3 и Sugiyama§2.1/начало2.2 с finite known ratio и finite-sample instability. Пример risks.5/.2, importance weights4/.25. Указаны неизменный conditional law, target support и variance weights; без них identity не переносится на иной label rule. |
| DANN | Ganin JMLR2016, §4.2 целиком (p11–13), equations13–15, §5.2.2/p21–22 и footnote6. Сохранена source-label/target-unlabelled постановка, два head, opposing gradients, mapping F/C/D к Gf/Gy/Gd. Scheduled extractor multiplier отделён от fixed head multiplier1. |
| GRL | Identity forward и custom backward проверены отдельно для features и domain weights. Сохранена positive domain CE, показано, почему дополнительный минус меняет оба update неправильно. GRL не выдан за обычную производную identity. Полный CPU update использует source class labels и domain membership обеих частей, без target class labels. |
| Ограничение alignment | Ganin§3.3/Theorem2, joint-error term. Собственный конечный пример: одинаковые feature marginals, domain accuracy.5, source error0, target error1. Chance discriminator accuracy не объявлена достаточным свидетельством правильных признаков. |
| FID/MMD | Heusel1706.08500v6, p6, весь FID passage/equation6; Gretton JMLR2012, population expression/Lemma6 и Gaussian-kernel discussion. В собственном counterexample mean0/variance1 совпадают, Gaussian-moment distance0, но RBF MMD²=.0363477. Раскрыта трёхчленная population формула; отличена от off-diagonal unbiased sample statistic. Это не image FID benchmark. |
| Смеси и real evaluation | Harvard2482–2601 и текущая53 сопоставлены. Не перенесена таблица50/50–80/20 как универсальное ранжирование. Выбор смеси/силы преобразований/checkpoint отделён от final real test; split units включают mesh family, environment и acquisition session. |
| Privacy | Carlini2301.13188v1, §4.2.1 generate/filter и §7.1/7.2 defense context прочитаны; связанная privacy/DP часть Responsible systems прочитана. Synthetic origin не превращена в гарантию, failed extraction не выдан за proof, post-processing ограничен отсутствием повторного доступа к private records. Формальный DP вывод не продублирован. |

### Выбор между расходящимися формулами

В печатной equation18 DANN стоит отрицательный domain-loss term, при этом
GRL определён с отрицательным backward. Механическое совместное применение
дало бы двойной разворот. Исполняемый пример следует updates13–15 и
§5.2.2, а не копирует эту запись как код. Оригинальная Figure1 не изменена:
lambda у domain head сохранена, экспериментальная convention с head multiplier1
объяснена в тексте. Положительный loss, отрицательный upstream gradient и
положительный head gradient проверены на фактическом autograd graph.

## Стиль

Прочитаны no-ai-slop/SKILL.md и eval.md. Удалена служебная подводка о том,
каким порядком автор будет объяснять тему. Вход начинается с наблюдаемого
различия renderer state и camera RGB, определения связаны с mask/pixel/
probability/gradient примерами. Не добавлены собственные Mermaid/SVG или
сгенерированные изображения. Исходные BlenderProc excerpts не переводились
и не переписывались ради стиля. Лицензионные подробности вынесены в metadata,
сокращённая подпись DANN сохраняет автора, paper/page, license link и derivation.

## Оригинальные материалы

Семь новых registered figures:

- два material JPEG512×512;
- два camera JPEG1039×1039;
- два segmentation JPEG1039×1039;
- DANN Figure1 PNG2200×940, complete diagram at400dpi из физической p12,
  crop[640,470,2840,1410], page3400×4400, с сохранением всех branches/labels/legend.

Все шесть JPEG byte-identical pinned BlenderProc76bd8b82 originals, GPL-3.0.
Сохранены source README/main.py и полный LICENSE, public copy доступна отдельно.
DANN metadata указывает реальную JMLR CC-by publication policy без выдуманной
версии4.0 или заявления о просмотренном подписанном соглашении. Главный
HTML host частично отдавал policy; неполный HTML не использован как evidence.
Проверенная официальная формулировка/URL записаны отдельно. Арxiv nonexclusive
license Tobin/Carlini не использована для разрешения переноса их рисунков.

Манифесты adjacent/central совпадают, hashes source/page/license проверены.
JPEG реально декодируются в asset tests; PNG byte-for-byte восстанавливается
из PDF тем же crop через pdftoppm. Source packet16MB, curated figures684KB;
не загружались модели, датасеты или новая среда Blender.

## Выполнение и автоматические проверки

Использовано существующее локальное PyTorch-окружение:

```sh
/Users/aleksan.potekhin/.cache/uv/environments-v2/035f72b8e8383b33/186c9c62a44ded93/bin/python publishing/tools/check_sim_to_real_examples.py
```

PASS:9 reader programs,9 ID maps,27 projection/reconstruction cases,
48 material cases,16 coverage thresholds,54 exact weighted risks,
30 actual two-head GRL graphs (lambda0,.1,.7,1,2; unequal source/target counts),
9 translated/scaled metric pairs и7 source figures.

Первый полный unit run:630 PASS и1 failure — underscores в новых curated IDs
нарушали schema. Исправлены IDs, не ослаблено правило. После этого3 affected
files/158 tests PASS, включая полный asset-registry test со всеми новыми
оригиналами. После последней правки подписи и viewer:3 affected files/14 tests
PASS, включая пять geometry/modifier-click tests viewer. Эти unit tests сами
по себе не проверяют перенос подписи; это подтверждено actual browser ниже.

Final build после viewer fix:1989 pages,25.93s,exit0. Затем164 output tests
PASS,0 broken routes/fragments/files,structural source-coverage PASS.
Scoped git diff --check PASS. После этого chapter body не менялся.

Логи:

- `/var/tmp/bookvar-sim-to-real-examples-2026-10-28.log`
- `/var/tmp/bookvar-sim-to-real-unit-2026-10-28.log` — исторический run с ID failure;
- `/var/tmp/bookvar-sim-to-real-focused-2026-10-28.log` —158 PASS после исправления;
- `/var/tmp/bookvar-sim-to-real-final-focused-2026-10-28.log` —14 PASS после viewer fix;
- `/var/tmp/bookvar-sim-to-real-viewer-build-2026-10-28.log`
- `/var/tmp/bookvar-sim-to-real-output-2026-10-28.log`
- `/var/tmp/bookvar-sim-to-real-links-2026-10-28.log`
- `/var/tmp/bookvar-sim-to-real-coverage-2026-10-28.log`

## Actual UI

Preview4330, desktop1280×900/mobile360×800. Все7 originals осмотрены
в source view и на странице. JPEG proportions сохранены; desktop comparison
cells около300px, mobile около148px, два больших multi-panel вывода632/328px.
DANN632×270 desktop,328×140 mobile; мелкие labels доступны в viewer.
Все5 display formulas clientWidth=scrollWidth=328, обе таблицы328/328,
нет oversized inline math/KaTeX errors; pageWidth=scrollWidth=360.

Все7 viewer реально открыты, увеличены до4× и перемещены до x1032;
обратный pan к0 проверен. Square outputs достигают y≈780–781;
DANN при таком размере не требует vertical pan. Для второго segmentation
после промежуточного smooth-scroll readout(0.5,4) выполнена дополнительная
проверка: final left0/top0, не объявлено преждевременное положение точным краем.

В actual browser обнаружено, что comparison-table images не наследуют
общую подпись в viewer: closest('p') не существует на таком уровне.
Исправлены2 строки в `site/src/components/figure-viewer.ts`: следующий
caption paragraph ищется после whole table или standalone image paragraph.
После пересборки все4 table images сохраняют source attribution в dialog;
три standalone изображения также повторно проверены. Caption links не заменены
на plain text, source image bytes не изменены. CSS не менялся.

Actual footer clicks:53→53a→54→53a. EN fallback осмотрен: видимый disclaimer,
все7 images,5 formulas328/328,нетerrors/overflow,footer остаётся в /en/.
Готовый перевод этим не объявлен; EN bodies не редактировались.

Proof: `/var/tmp/bookvar-sim-to-real-2026-10-28.png`.
Preview tab5 оставлен deliverable; temporary viewport reset.

## Сохранение и остаток задачи

У53 изменены только next metadata и footer. Regression test восстанавливает
прежний полный SHA
`8b162e5380772ff0066fc7241061f4d6af8aa7e536d1b0172a753211bf6e855e`.
Текущий53 SHA
`df616fb7536981209a3fdef659003fb3f5b336c1983132f75f7d126b95fc3dd4`.
Принятая06a не изменена: d35a256a…750d3ddd7. Готовый RL не редактировался.

H-V1-09/H-F1-09 всё ещё re-review. Дальше — source2602–2760:
DecisionFramework/distillation mapping к уже готовым главам, затем individual
native/generated figure keep/link/omit decisions (14-file native listing уже
проверен, не все images visually accepted). Отдельно сверить generative-model
paragraph2480: подробная diffusion/flow глава64g уже есть и связана; достаточное
покрытие GAN пока не установлено. Не выводить decision boundary из source
unconditional density sketch и не переносить generic50–80% synthetic recipe.
Остальная Harvard/HSE/EDLS очередь, атлас/source/rights и отложенный EN-паритет
не закрыты. Общая цель остаётся активной.
