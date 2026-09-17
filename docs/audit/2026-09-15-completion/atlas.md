# Task 4: атлас моделей — закрытие замечаний

Статус: реализация завершена, готово к независимой редакторской проверке. Это не финальная приёмка основного агента.

Проверены все 28 страниц с префиксом `02 Атлас моделей/` из исходного `reference-practice.json`: 43 замечания, 38 исправлены в этом проходе, 5 уже были исправлены до него, 0 отклонены, 0 открыты. Авторских английских соответствий этих страниц в `publishing/navigation.yml` нет.

## Метод и границы

Каждая страница целиком прочитана до правок и повторно после них. Точные исходные тексты сохранены через apply_patch в `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/`. Ни одна прежняя иллюстрация не удалена. Изменённый заголовок Gemma сохраняет прежний якорь. Новые объяснения связывают механизм, его ограничения и пример; готовые архитектурные рисунки переиспользованы без перерисовки. Подход llm-wiki повлиял на сохранение исходных материалов и явное отделение источника от редакторского вывода; executing-plans удерживал границы назначенной области.

- This pass resolves the 43 assigned findings; it is not an exhaustive latest-release inventory or a claim that every inherited model-card statement was re-researched.
- Primary acceptance and independent editorial review are pending.
- Jamba architecture crop has a partially clipped caption; retained unchanged, correct figure link and description provided. Any replacement/registry work remains with primary.
- No authored EN counterparts were mapped for the 28 pages; fallback copies were not treated as authored translations.

## Проверки

- snapshots: 28/28 exact SHA-256 matches; preserved all original embeds and headings (renamed Gemma heading has compatibility anchor).
- local_links: 182 wiki targets and explicit heading references checked: zero errors.
- unit_tests: 2 files / 12 tests passed: links.test.ts, check-built-links.test.ts.
- pages_base: Executed actual publicationHref/convertWikiSyntax at /bookvar. Raw root-relative links are unchanged; wiki links acquire base. Timeline's 11 agent links now use wiki syntax.
- diff_check: git diff --check -- '02 Атлас моделей' 'docs/audit/2026-09-15-completion/atlas.json': exit 0.
- full_site_build: Not run: primary owns full build and final generated HTML checks.
- runtime: Node scalar checks and PyTorch 2.8 CPU in /private/tmp/bookvar-editorial-venv/bin/python; no weights downloaded or training run.

## Новая оригинальная иллюстрация

Kimi Linear: `00 Учебник/Assets/Figures/curated/kimi-linear/arch.jpg`, 237810 байт, 1272x1398, JPEG/JFIF. Файл скачан неизменным; расширение .jpg соответствует реальному формату, хотя upstream использует .png. Визуально проверены стек 3 KDA : 1 MLA, ветвь MoE и внутренний блок KDA.

- Автор: Moonshot AI / Kimi Linear authors.
- [Закреплённый оригинал](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3/figures/arch.png).
- Commit: `8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3`.
- SHA-256: `132ae021fa4661ed39e7be784d46f05f22b82aabb9afd2bab8dbdc0a5a61cba0`.
- [MIT License](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3/LICENSE), Copyright (c) 2025 Moonshot AI.
- Primary reported registry entry and full unchanged MIT notice in site/public/licenses/kimi-linear-MIT.txt; 18 registry tests passed on primary side.

## Индивидуальные результаты

### 1. 02 Атлас моделей/_index.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/_index.md`.

SHA-256 до: `9d013be8ee207d7b43490f04ca12dde2e6e8a216844e927a4abbc7990481bc32`.

SHA-256 после: `1a08660ffc2fdada88d6f0f77bbcba14658e4c55919710e6d1777f264d68f189`.

1.1. **fixed** — P3, исходная строка 23. Таблица топологий смешивает класс self-attention, альтернативный оператор и мультимодальность, которая совместима с разными топологиями.

Результат: Перед первой таблицей явно разделены топология внимания, оператор последовательности и состав модальностей; приведён совместимый гибридный мультимодальный decoder. Полная карта и её ссылки сохранены.

Иллюстрации: No new figure; original embeds retained

### 2. 02 Атлас моделей/Семейства/Мультимодальные семейства.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Мультимодальные семейства.md`.

SHA-256 до: `f445464666d1a84bd8ee90625f83034c4ddea885ca1eec88169558ea913d2d56`.

SHA-256 после: `fc9420308d6dcc9d9d7dfbef3b6bae6c62946107c046f3eb12ec79f964b79e3c`.

2.1. **fixed** — P2, исходная строка 88. Предыдущий абзац сам описывает thumbnail InternVL, поэтому наличие глобального обзора не является указанным отличием.

Результат: Убрано ложное отличие по наличию обзора. На одном изображении сопоставлены подбор неперекрывающейся сетки/thumbnail InternVL, перекрытие Molmo, удаление дублирующих признаков и row-major последовательность с маркерами.

2.2. **fixed** — P2, исходная строка 45. Подпись требует сверки с конкретной фигурой Thinker–Talker: не вся стрелка между компонентами обозначает backprop.

Результат: Просмотрен фактический PNG: легенда прямо обозначает solid Forward Propagation и dashed Backward Propagation. Подпись теперь различает эти стрелки и не объявляет всю схему потоком градиентов.

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/2412.05271v2#S3.SS1)
- [Источник](https://arxiv.org/html/2409.17146v2#S2)
- [Источник](https://arxiv.org/html/2503.20215v1#S2)

Численные/исполняемые проверки:

- Qwen2-VL arithmetic: (224/14)^2=256; 256/(2*2)=64; unchanged local scalar derivation.

Иллюстрации: Viewed actual Qwen2.5-Omni and Molmo PNGs; all 4 original figures retained, no new assets.

### 3. 02 Атлас моделей/Семейства/Baichuan.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Baichuan.md`.

SHA-256 до: `12076e0c27cb6541ea66a164abf980461201fdb47038867645709f90c74796c1`.

SHA-256 после: `a0b2e20ce5c1dbe4da8d3107be295b79ac1a279bdf665d7a5874b2e0a40ae06f`.

3.1. **fixed** — P2, исходная строка 40. Современная ветка перечислена через назначения, но ключевые архитектурные/обучающие отличия M3 фактически отложены за ссылку.

Результат: M3 привязана к фактическому Qwen3 MoE config (94 слоя, 128/top-8, 64Q/4KV); post-trained checkpoint не назван base. Объяснены три стадии RL/distillation, четыре сегмента консультации, quality gate и fact verifier. Добавлен явно синтетический inquiry-loop и граница медицинской пригодности.

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/2602.06570v1)
- [Источник](https://huggingface.co/baichuan-inc/Baichuan-M3-235B)
- [Источник](https://huggingface.co/baichuan-inc/Baichuan-M3-235B/blob/main/config.json)

Численные/исполняемые проверки:

- Проверены config values; примеры генерации M3 не запускались, clinical validation не заявлена.

Иллюстрации: No new figure; original embeds retained

### 4. 02 Атлас моделей/Семейства/BERT.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/BERT.md`.

SHA-256 до: `ada9dfa4fa9a3bc7282f0439db7b39e9bdf862bc1764b1c9250336ef73978434`.

SHA-256 после: `fd48361e66b15d397ae913532826bc371e3211f8c57aeaa19ed56f189ded7ee9`.

4.1. **fixed** — P2, исходная строка 31. Фраза читается как разные параметры по слоям, хотя имеется в виду совместное использование весов.

Результат: Все три упоминания ALBERT исправлены на совместное повторное использование весов между слоями; явно отделено количество параметров от последовательных вычислений.

4.2. **fixed** — P2, исходная строка 46. Карточка не ведёт к соответствующей главе учебника, хотя главы BERT и паттернов уже существуют.

Результат: Добавлены существующие учебные главы MLM/NSP и архитектурных паттернов; пути и разделы проверены через rg. Объяснён переход от векторов encoder к task-head.

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/1909.11942v6#S3.SS1)

Численные/исполняемые проверки:

- 768/12=64 и 1024/16=64: согласованные размеры голов; полноразмерная модель не запускалась.

Иллюстрации: No new figure; original embeds retained

### 5. 02 Атлас моделей/Семейства/Command R.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Command R.md`.

SHA-256 до: `ba4654209a2adb6909988ef8fd20a2ce0f6fc3d3f14b93e424169cedb71731fa`.

SHA-256 после: `9f1b8422be29e84b8a7e89887c6911d0dbc5cfcc61720fad556f90f868361eeb`.

5.1. **fixed** — P2, исходная строка 35. Заявлен latest verified A+, но существенные признаки нового релиза читателю предлагается добыть самостоятельно.

Результат: Добавлены проверенные 218B/25B, 128000 context, 64000 max output, text+image→text; расхождение 24B в анонсе vs25B в документации явно раскрыто. Переходы к MoE и VLM, W4A4-условие железа и нижняя оценка памяти; частичные сведения vision не домыслены.

Проверенные первичные материалы:

- [Источник](https://cohere.com/blog/cohere-releases-command-a-plus)
- [Источник](https://docs.cohere.com/docs/command-a-plus)

Численные/исполняемые проверки:

- Executed Node scalar: 218e9*2=436GB BF16; 218e9*0.5=109GB ideal4bit. Not deployment measurement.

Иллюстрации: No new figure; original embeds retained

### 6. 02 Атлас моделей/Семейства/DBRX.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/DBRX.md`.

SHA-256 до: `ec67cd05b4c142adf095a7e4ab37769dd69dee3f8b7b41917359ea6d42159daa`.

SHA-256 после: `8411fd99a021db39c1636ef793742ad0fa10491f1d12df43e8101f9e577bcdfd`.

6.1. **fixed** — P2, исходная строка 23. Единственная иллюстрация — release benchmark, при этом центральная заявленная тема16/top4 и132B/36B не имеет наглядного разбора.

Результат: Сохранена оригинальная release-иллюстрация. Добавлены компактная таблица 16/top4 vs8/top2, C(16,4)=1820 vs28, два примерных набора маршрутов и выведение S≈4/E≈128 из округлённых total/active.

6.2. **fixed** — P3, исходная строка 39. Негативное утверждение о релизах быстро стареет и не имеет точного проверенного каталога.

Результат: Удалено недоказуемое отрицание существования DBRX-2. Охват ограничен датированным Base/Instruct 2024, перепроверка не заявляет исчерпывающую новизну.

Проверенные первичные материалы:

- [Источник](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)

Численные/исполняемые проверки:

- Executed Node: combinations1820,28; ratio65; shared4/expert128; active36; 132*2=264GB (scalar arithmetic). HF model card/config inaccessible (401); did not claim weight tensor verification.

Иллюстрации: No new figure; original embeds retained

### 7. 02 Атлас моделей/Семейства/DeepSeek.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/DeepSeek.md`.

SHA-256 до: `8efd7dd87c6e41f2f42dadaa08ae7dd057a29a4381def3945d080f53e2c524dd`.

SHA-256 после: `922432e4f44a0ec5360475bb561bcc15df5438ef78be2c0651cb08e7a1d42206`.

7.1. **fixed** — P2, исходная строка 34. Ключевые механизмы описаны по абзацу без ссылок к уже существующим подробным главам; читателю приходится искать их отдельно.

Результат: Добавлены точные переходы к MLA, MoE, DSA, RLVR и GRPO. Для MTP дан прямой §2.2 первичного отчёта: отдельной подробной локальной главы MTP не найдено. Глава о спекулятивном декодировании связана только с применением MTP при выводе.

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/2412.19437v2#S2.SS2)

Численные/исполняемые проверки:

- No new numeric example; local targets verified by rg --files/headings.

Иллюстрации: Actual R1 PNG viewed: AIMEaccuracy and response-length learningcurves; original two figures preserved.

### 8. 02 Атлас моделей/Семейства/Falcon.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Falcon.md`.

SHA-256 до: `7164820964baf3655cd8c640a78f8717673c79dd7c081943825626b6317c5280`.

SHA-256 после: `4558710667de04eb5d12a963e6d507774be92ba46c285ddf3611be39151a1128`.

8.1. **fixed** — P2, исходная строка 21. Ранние модели7B/40B объединены под одним названием без размеров KV-групп, хотя распределённая40B имеет особенности multi-group MQA.

Результат: По официальным конфигурациям разведены Falcon 7B (71 Q, одна KV-голова) и Falcon 40B (128 Q, 8 KV-голов). Объяснено отличие глобальной и групповой схем; вычислен объём K+V на токен и слой.

8.2. **fixed** — P3, исходная строка 48. Метаданные и конец текста повторяют датированную новизну, но нет ссылок к механизмам гибрида.

Результат: Убрано абсолютное утверждение о последнем релизе. Указаны границы охвата и переходы к гибридам/Jamba. По Table 1 Falcon-H1 объём данных уточнён для разных размеров: 18T не приписывается всем моделям.

Проверенные первичные материалы:

- [Источник](https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json)
- [Источник](https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json)
- [Источник](https://arxiv.org/html/2507.22448v1)

Численные/исполняемые проверки:

- Executed Node: hd64 both;128/8=16;2*1*64*2=256B,2*8*64*2=2048B,2*128*64*2=32768B;16x cache ratio. No full-model inference.

Иллюстрации: No new figure; original embeds retained

### 9. 02 Атлас моделей/Семейства/Gemma.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Gemma.md`.

SHA-256 до: `8cb650c81c45afb6d433a51a4209792e12aa2b00c03a01beb8ad5a8cb8749e87`.

SHA-256 после: `5e3bbedfc74ab90acfee105a010dad6be8eb87a545259c56a6dc0c194b188bf2`.

9.1. **already-fixed** — P1, исходная строка 23. В Gemma1 виды внимания переставлены местами: официальный обзор Google указывает MQA2B иMHA7B.

Результат: В исходном снимке уже правильно указаны MQA у 2B и MHA у 7B; сведения повторно подтверждены официальным обзором Google.

9.2. **fixed** — P2, исходная строка 44. Сравнительный benchmark не позволяет приписать выигрыш именно данным без абляции.

Результат: После просмотра графика убрана причинная интерпретация о выигрыше от данных. Новый заголовок описывает сравнение результатов; прежний slug сохранён явным якорем.

9.3. **already-fixed** — P2, исходная строка 24. Не отделены дистиллированные2B/9B от27B; читатель может отнести один рецепт ко всему поколению.

Результат: В исходном снимке уже разделены дистиллированные 2B/9B и 27B без этой дистилляции; подтверждено официальным обзором Google.

Проверенные первичные материалы:

- [Источник](https://developers.googleblog.com/gemma-explained-overview-gemma-model-family-architectures/)

Иллюстрации: Viewed actual Gemma performance PNG, no edits to image; old heading slug retained as explicit anchor.

### 10. 02 Атлас моделей/Семейства/GLM.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/GLM.md`.

SHA-256 до: `a23f78663a32562944f8c8ef292a42adbb997a7ef965c540f2291b7fd7fbf0db`.

SHA-256 после: `153c71b1ddd2d4f23578ee6550c9d246c48691cc362b6320d83fdc63b17ac8f9`.

10.1. **already-fixed** — P1, исходная строка 28. GLM4.5 в официальном config имеет96query и8KVheads (GQA); термин MLA-like приписывает иное представление KV без основания.

Результат: В исходном снимке уже была исправлена GQA. Проверенная закреплённая официальная конфигурация содержит 96 Q и 8 KV-голов; MLA не приписывается GLM-4.5.

10.2. **fixed** — P2, исходная строка 17. Вступление категорично, а раздел неизвестного ниже говорит, что не установлено новое pretraining или posttraining.

Результат: Устранено противоречие между введением и строкой 5.1. Равенство 744B/40B не названо доказательством идентичности архитектуры или предобучения; охват ограничен рассмотренными версиями при наличии более новых поколений в каталоге.

Проверенные первичные материалы:

- [Источник](https://huggingface.co/zai-org/GLM-4.5/blob/8b91a96cb5e3a6dde04be29567b87b06f3dd61dc/config.json)
- [Источник](https://github.com/zai-org/GLM-5)

Численные/исполняемые проверки:

- 96/8=12 heads per group; direct scalar relation, no weight loading.

Иллюстрации: No new figure; original embeds retained

### 11. 02 Атлас моделей/Семейства/GPT.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/GPT.md`.

SHA-256 до: `e1e2375987f859833f8a8cb8cce90a85d78f733e6f0b517930eae620d49df729`.

SHA-256 после: `8a800269739b1010b48f43373e8106edb05d1fbdcc6aa84cf8814c585f50e63e`.

11.1. **fixed** — P2, исходная строка 15. Публичное описание архитектуры не делает полное обучение GPT3 воспроизводимым: данные и веса не опубликованы целиком.

Результат: Документированность отделена от воспроизводимости: по официальным репозиториям раздельно описаны код, веса и данные GPT-1/2/3. Синтетические оценочные данные GPT-3 не названы обучающим корпусом.

11.2. **fixed** — P2, исходная строка 24. Объединение скрывает раскрытые размеры1.3B/6B/175B InstructGPT и закрытость конкретных ChatGPT.

Результат: Разделены исследовательские политики InstructGPT 1.3B/6B/175B и продукт ChatGPT; модель награды 6B выделена отдельно. Добавлены главы SFT, reward modeling и PPO; охват продукта ограничен указанным поколением.

Проверенные первичные материалы:

- [Источник](https://github.com/openai/finetune-transformer-lm)
- [Источник](https://github.com/openai/gpt-2)
- [Источник](https://github.com/openai/gpt-3)
- [Источник](https://arxiv.org/html/2203.02155v1)
- [Источник](https://openai.com/index/gpt-5-6/)

Численные/исполняемые проверки:

- No numeric toy, model sizes verified in paper; no training reproduction claimed.

Иллюстрации: No new figure; original embeds retained

### 12. 02 Атлас моделей/Семейства/InternLM.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/InternLM.md`.

SHA-256 до: `2c447a1e7673c76145aa9b2871594146cfcf49dc39459374670921a9233c3171`.

SHA-256 после: `ca848ea21c14cf463618be77731358bbaa8339c363f38d8747c6a3f889771b4f`.

12.1. **fixed** — P2, исходная строка 38. Техническая особенность названа, но не показано, что изменено в форме/порядке и почему это влияет на shard.

Результат: Порядок QKV взят из официального InternLM2Attention: 8 групп × (4Q+K+V) × 128 = 6144. Добавлены формы тензоров и пример для 4 Q/2 KV с объяснением смежного группового разбиения. Изменение числа KV-голов требует преобразования весов, а не только конфигурации.

Проверенные первичные материалы:

- [Источник](https://huggingface.co/internlm/internlm2-7b/blob/main/config.json)
- [Источник](https://huggingface.co/internlm/internlm2-7b/blob/main/modeling_internlm2.py)

Численные/исполняемые проверки:

- Actual PyTorch2.8 CPU reshape/slice toy:Q=[0,1,2,3,8,9,10,11],K=[4,5,12,13],V=[6,7,14,15],shapes[1,4,1,2]/[1,2,1,2];PASS. Configscalar6144PASS. No downloaded model.

Иллюстрации: No new figure; original embeds retained

### 13. 02 Атлас моделей/Семейства/Jamba.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Jamba.md`.

SHA-256 до: `3279904018239b991ef0689a68aeadc0d5dc082646ed6ee406414489b6455999`.

SHA-256 после: `bd62715b6a6030359446d06c9d98a40d423d9ef0f39dc5b64b9abbb6c5d15ada`.

13.1. **fixed** — P3, исходная строка 15. Начальное противопоставление сообщает, как не объяснять модель, вместо краткого описания её устройства.

Результат: Введение описывает конструкцию: семь Mamba-слоёв и один слой внимания, MoE в каждом втором слое, выбор двух из шестнадцати экспертов. По фактическому изображению исправлена подпись: слева стек, справа типы слоёв. Attention MoE показан как вариант, не выбранный в исходной конфигурации. Существующий crop сохранён, обрезка подписи отмечена, дана прямая ссылка на Figure 1 и CC BY-SA 4.0.

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/2403.19887v2)

Численные/исполняемые проверки:

- Counted depicted8layers=7Mamba+1attention;4MoEpositions; no inferencebenchmark.

Иллюстрации: Actual local PNG inspected; bottom originalcaption clipped; parent owns bestoriginalregistry decision.

### 14. 02 Атлас моделей/Семейства/Kimi.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Kimi.md`.

SHA-256 до: `4cc44f22b81657e203113336f43d7de1322b856d6cc730a0886a61076d461512`.

SHA-256 после: `df641246623d8079e327145c37dc799cff5a4025c64c0019c457d1bcb215b951`.

14.1. **already-fixed** — P1, исходная строка 24. Moonlight ошибочно назван плотной 3B-моделью: официальный репозиторий описывает 16B total / 3B active MoE.

Результат: До этой задачи строка Moonlight уже правильно различала 16B всего и 3B активных параметров; полное чтение и официальная таблица подтверждают исправление.

14.2. **fixed** — P2, исходная строка 37. Единственная иллюстрация показывает RL-систему Kimi 1.5, не главную архитектуру K2/KDA.

Результат: Добавлена неизменённая оригинальная схема Kimi Linear с проверенной MIT-лицензией и закреплённым commit. Визуально сверены левый стек, KDA и MoE; объяснены отношение 3:1, фиксированное состояние, периодический MLA и независимость маршрутизации экспертов. Сохранена схема k1.5, добавлены ссылки на локальные главы.

Проверенные первичные материалы:

- [Источник](https://github.com/MoonshotAI/Moonlight)
- [Источник](https://github.com/MoonshotAI/Kimi-Linear)
- [Источник](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/8c1d85eb6b5f8fcefb15758691b0ce50b0827ce3/LICENSE)

Иллюстрации: Viewed original 1272×1398 JPEG, 237810 bytes, SHA256 132ae021fa4661ed39e7be784d46f05f22b82aabb9afd2bab8dbdc0a5a61cba0. Upstream arch.png saved as arch.jpg without changing bytes; primary owns registry and full MIT notice.

### 15. 02 Атлас моделей/Семейства/Llama.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Llama.md`.

SHA-256 до: `9c2dcc1cc12d1dabe268626ee92e9dfb4af2f638cfedbf384eacfcf80a095f23`.

SHA-256 после: `8d5ae02e6f5b818e4a6eb6507458edaa30d97fcfb54614ef6ec3c84d7a8069b0`.

15.1. **fixed** — P2, исходная строка 48. Изменения блока вроде MHA→GQA обесцениваются как неархитектурные, чтобы противопоставить им Llama 4. Это смешивает семейство decoder-only с конкретной архитектурой.

Результат: Разделены общая причинная decoder-топология и архитектура подслоёв. GQA прямо названа архитектурным изменением; Llama 4 также сохраняет decoder, меняя FFN и мультимодальный вход. Исходная иллюстрация и ссылки сохранены.

Численные/исполняемые проверки:

- Conceptual classification correction; no new numeric claim.

Иллюстрации: No new figure; original embeds retained

### 16. 02 Атлас моделей/Семейства/Mamba.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Mamba.md`.

SHA-256 до: `af8d052acfd833ea2bc32f8546e0e36139d8c130fd23038b77aa61591ed284c7`.

SHA-256 после: `117146c73d4572c087c392f543428a55c87b7aa0c07f12f0a80d609ce16ec0f3`.

16.1. **fixed** — P3, исходная строка 43. Сравнение prefix caching слишком общее: snapshot состояния возможен, но granular reuse и branching требуют иной реализации.

Результат: Убрано универсальное утверждение о сложности prefix caching. Объяснены сохранение состояния всех слоёв и свёрток, независимые ветви, невозможность получить произвольный более ранний префикс из единственного позднего снимка, отличие от KV-блоков; даны ссылки и вычисленный скалярный пример.

Численные/исполняемые проверки:

- Executed scalar recurrence: prefix [2,4] -> 5; branches +1 -> 3.5 and +3 -> 5.5. This is not a Mamba benchmark.

Иллюстрации: No new figure; original embeds retained

### 17. 02 Атлас моделей/Семейства/Mistral и Mixtral.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Mistral и Mixtral.md`.

SHA-256 до: `b381fecdd75409daff602d1afc97cfb24f1bc4343583c3a68812a3eb299c0565`.

SHA-256 после: `5930e31a119d674c0a696cd8b86f05d0e946255ea3fbaf5a85f34b4311fafb2d`.

17.1. **fixed** — P2, исходная строка 31. Противоречивая классификация публикации в одной ячейке, которую нельзя использовать как справочный факт.

Результат: Medium 3.5 теперь однозначно обозначена как open-weight под Modified MIT, с датой и прямой официальной карточкой. Исправлены противоречивые метаданные latest_open_generalist; доступность весов отделена от полного рецепта обучения.

17.2. **fixed** — P3, исходная строка 27. Объединённая строка может приписывать Tekken и один контекст разным семействам без различения checkpoints.

Результат: NeMo и Ministral 2024 разделены: Tekken относится к NeMo, обе исходные линии заявляли 128K, но права на веса различались. Для Ministral 8B указана исследовательская публикация, коммерческое развёртывание не приравнено к Apache-2.0.

Проверенные первичные материалы:

- [Источник](https://docs.mistral.ai/models/mistral-medium-3-5-26-04)
- [Источник](https://mistral.ai/news/ministraux/)
- [Источник](https://mistral.ai/news/mistral-nemo/)

Иллюстрации: No new figure; original embeds retained

### 18. 02 Атлас моделей/Семейства/Nemotron.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Nemotron.md`.

SHA-256 до: `e8b07dc6761963065c1d49326fe8c8137a71ae88f58b33a0bfc09043cca65e29`.

SHA-256 после: `ace74bd899a8dd12b979d3fe78ac1223f0bc296f356a50330f63364e99527c35`.

18.1. **fixed** — P2, исходная строка 48. Формат pre-training в предыдущем разделе автоматически переносится на serving. Экономия при инференсе зависит от формата checkpoint и kernel.

Результат: Разделены формат предобучения и формат контрольной точки/ядра при выводе. Убрана безусловная экономия bandwidth, явно исключено перенесение FP4-стоимости на BF16. Пример 240 GB против 60 GB назван идеальной упаковкой весов без накладных расходов, а не измерением скорости.

Численные/исполняемые проверки:

- Executed Node arithmetic: 120e9*2=240 GB; 120e9*0.5=60 GB.

Иллюстрации: No new figure; original embeds retained

### 19. 02 Атлас моделей/Семейства/Phi.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Phi.md`.

SHA-256 до: `aadcd60b90515f5529f59470194c82011feab9ffb95450dd44db923429477667`.

SHA-256 после: `3b4e7d346e25778e30af0afc1ca7a0d036787f63c6e9157aa07d8e516d329c19`.

19.1. **fixed** — P2, исходная строка 47. Подпись выводит преимущество качества данных из сравнения моделей якобы на одной смеси; причинный вывод не следует из такого описания эксперимента.

Результат: Осмотрен полный локальный Figure 3 и исходная подпись Phi-3 v4. Точно названы логарифмические оси, серии и относящаяся к Llama 2 фраза о фиксированных данных. Убран недоказанный причинный вывод о качестве данных.

19.2. **fixed** — P2, исходная строка 37. Ключевой метод Phi-4 назван, но не объяснено, как выбираются такие позиции и строятся preference pairs.

Результат: PTS объяснён по Phi-4 §4.3: оценки успешности продолжений по oracle, рекурсивная локализация, общий префикс как запрос и однотокенные accepted/rejected ответы. Отделены вероятность успеха и вероятность токена, отмечена неполнота поиска и ссылка на DPO.

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/2404.14219v4)
- [Источник](https://arxiv.org/html/2412.08905v1#S4.SS3)

Иллюстрации: Viewed local Phi-3 Figure 3; no image changes.

### 20. 02 Атлас моделей/Семейства/Qwen.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Qwen.md`.

SHA-256 до: `2da01681c5980cfeb8f6d5f9572b275cd8e4220cb8f5775c9907a710968fc429`.

SHA-256 после: `f2d18cb7da6a131c84d4925f6f559561c78a6323b575e8a9eeff2f77203c2204`.

20.1. **fixed** — P2, исходная строка 42. Новое ядро семейства описано одним общим предложением о состоянии; нет обновления памяти, схемы hybrid-блока и перехода к главе с механизмом.

Результат: Переиспользован полный оригинальный слайд CS336 с архитектурой Qwen3-Next. Осмотрено изображение; по официальной карточке сверены 48 слоёв, 36 GDN и 12 gated attention. Добавлена размерностно согласованная рекурсия, механизм глобального забывания и направленной замены, вычисленный пример, ссылки на подробную главу.

20.2. **fixed** — P3, исходная строка 66. Схема стадий обучения объявляется доказательством утверждения об attention, которого она не исследует.

Результат: Убрано слово «доказательство»; схема теперь описывает получение checkpoint, слияние режимов и обученное переключение формата генерации без замены attention.

Проверенные первичные материалы:

- [Источник](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

Численные/исполняемые проверки:

- PyTorch CPU: k=[1,0], gamma=beta=1 replaces first row [2,3] with v=[7,8], retaining second [4,5].

Иллюстрации: Viewed full Qwen3-Next CS336 slide; reused existing registered image unchanged.

### 21. 02 Атлас моделей/Семейства/RetNet.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/RetNet.md`.

SHA-256 до: `029d3dd6ab9f506df90136a180e03db71b2876487467f898e23c9415886914fa`.

SHA-256 после: `2f7e17d735fc28ef937b725e26340994173c48834302bb35ad5fb331929acf76`.

21.1. **fixed** — P2, исходная строка 23. Повреждённая фраза и отсутствие самой рекуррентной записи оставляют читателя без проверяемой связи трёх режимов.

Результат: Исправлена повреждённая фраза chunkwise; добавлены рекурсия, размерности комплексных q/k и состояния, сопряжение ключа, развёрнутая сумма и перенос через границу блока. Все три формы сопоставлены на скалярном примере и связаны с подробным выводом.

Численные/исполняемые проверки:

- Executed scalar recurrence [2,4,8], gamma=.5 gives [2,5,10.5]; full sum and chunk boundary give 10.5.

Иллюстрации: No new figure; original embeds retained

### 22. 02 Атлас моделей/Семейства/RWKV.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/RWKV.md`.

SHA-256 до: `d2cc65384a0694d847b087826e1c0bd88c97d57e313d30743f47e65df03061e9`.

SHA-256 после: `d1a64bd5641c184a01ecaa12cf3f7dd55f496519e16d19f90ff30e9fa3c958b6`.

22.1. **fixed** — P2, исходная строка 34. Переход v5/6→v7 описан слишком общо: у v6 тоже есть data-dependent компоненты; не показан новый член обновления состояния.

Результат: По Eagle/Finch и Goose разделены фиксированное затухание v5, входозависимые коэффициенты v6 и диагонально-низкоранговое обновление v7. Даны формула, размерности, смысл нового члена и вычисленный пример переноса столбца; ссылки на первичные статьи и локальный вывод v4.

22.2. **fixed** — P3, исходная строка 15. Начало построено на споре с отсутствующим тезисом вместо описания модели.

Результат: Вводный абзац начинается с хранения истории в состоянии, без спора с отсутствующим обещанием.

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/2404.05892v2#S3)
- [Источник](https://arxiv.org/html/2503.14456v2#S3)

Численные/исполняемые проверки:

- PyTorch CPU: S=[[1,2],[3,4]], z=[1,0], b=[0,1]; S outer(z,b)=[[0,1],[0,3]], moves first column into correction of second.

Иллюстрации: No new figure; original embeds retained

### 23. 02 Атлас моделей/Семейства/T5.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/T5.md`.

SHA-256 до: `1a49877a678df776f205db561224c7ae0a5986b54259563cf593d134914a1399`.

SHA-256 после: `85ef055551fc8a65df6552e5bffa8d3492f8f5f16aa1c64437f1bd22a6d24bf0`.

23.1. **fixed** — P3, исходная строка 27. Механизм точен, но короткий пример input/target сделал бы sentinel-формат сразу понятным.

Результат: Добавлена точная input/target пара с двумя удалёнными фрагментами, двумя входными маркерами и завершающим extra_id_2. Объяснены невключение оставленной точки в цель, EOS и условность разбиения без запуска SentencePiece; дана ссылка на подробную главу.

Численные/исполняемые проверки:

- Manual reconstruction using quick brown and lazy dog recovers the original sentence with retained terminal dot; target includes exactly three ordered sentinels.

Иллюстрации: No new figure; original embeds retained

### 24. 02 Атлас моделей/Семейства/Yi.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Семейства/Yi.md`.

SHA-256 до: `b730ea012c1fafa6b9f6e5354c27f5a2e903e7f9fce4d67d6e4a3122bf3da446`.

SHA-256 после: `caa2a2df7f4437905e4728fe87e701258faeb5e3e99351e698959f57fadd5a8f`.

24.1. **fixed** — P2, исходная строка 15. Из отсутствия нового attention причинный вклад рецепта не установлен; формулировка заменяет факты оценкой.

Результат: Вступление теперь перечисляет фильтрацию, long-context этап и RoPE, мультимодальность и кодовую специализацию. Убран причинный вывод о силе recipe из отсутствия нового attention; сравнения изменений явно разделены. Непроверенная исчерпывающая latest-формулировка заменена границами охвата.

Иллюстрации: No new figure; original embeds retained

### 25. 02 Атлас моделей/Сравнения/Карта архитектурных линий.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Сравнения/Карта архитектурных линий.md`.

SHA-256 до: `975315612617cce75935c4a72b56bfcfd6c135602439785667628ac6bd7f8a25`.

SHA-256 после: `4017955fe3ddcff8febc21efbab1a214d7649f57b0d5bfb4fb58b3bdd9a507b4`.

25.1. **fixed** — P2, исходная строка 10. Карта обещает стрелки родства, но состоит из четырёх несвязанных рисунков без общего сопоставления или переходов к главам.

Результат: Убрана семантика отсутствующих стрелок. Добавлена таблица механизм → конкретные поколения → глава; четыре сохранённых рисунка соотнесены со строками. При осмотре Jamba и сверке статьи дополнительно исправлен ошибочный номер Figure 1 на Figure 6 (training-loss curves, не layout/MoE).

Проверенные первичные материалы:

- [Источник](https://arxiv.org/html/2403.19887v2#S6.F6)

Иллюстрации: Viewed Jamba loss image; retained all four original figures.

### 26. 02 Атлас моделей/Сравнения/Перенос vision encoder между VLM.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Сравнения/Перенос vision encoder между VLM.md`.

SHA-256 до: `662e32e9a13b575d8650e98f76ea7a64b4d964c06a06dbaf8a5a4aaa125e6399`.

SHA-256 после: `0e1bacdc9a5c466e73b5396db928f5e006fa3db60a3ae002bb840ec87d6e4f33`.

26.1. **fixed** — P3, исходная строка 26. Рецепт представлен как единственный обязательный путь, хотя это разумный baseline, не универсальное условие.

Результат: Заморозка обеих башен названа первым проверяемым baseline, а не законом переноса. Убран запрет на более раннее размораживание; объяснён контролируемый сравнительный эксперимент с фиксированными данными, бюджетом и оценками.

Иллюстрации: No new figure; original embeds retained

### 27. 02 Атлас моделей/Сравнения/Сравнение семейств.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Сравнения/Сравнение семейств.md`.

SHA-256 до: `9845eefaa1d458fc8a1b4c3207a8ab9b9be08c7f583c91fd1ae4acbf28e7283c`.

SHA-256 после: `12b98bbc8aa4248db540b8eaa78ba70da2689b23da50cc99fa250468ebfe3482`.

27.1. **already-fixed** — P1, исходная строка 16. Таблица устарела относительно самих карточек: у Qwen потерян DeltaNet, у Falcon H1 — Mamba, у Nemotron 3 — hybrid MoE. Пользователь получает противоречивые справочные сведения.

Результат: Исходный P1 уже был исправлен до начала этой задачи: в полном pre-task снимке присутствуют отдельные строки Qwen3-Next/3.5 с DeltaNet, Falcon-H1 с Mamba-2 и Nemotron 3 Super с LatentMoE. Дополнительно уточнены Kimi K2/Linear, Mistral/Mixtral, Falcon 7B/40B и историческая область Baichuan.

Проверенные первичные материалы:

- [Источник](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

Иллюстрации: No new figure; original embeds retained

### 28. 02 Атлас моделей/Timeline.md

Полное чтение до/после: true/true. Снимок: `.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/task-4-atlas-before/02 Атлас моделей/Timeline.md`.

SHA-256 до: `d4e9d6aef906a8eb1a1720a561a94d9a0a0133686cf0b93490b904038683a76c`.

SHA-256 после: `486e1d79c182c2679109774d4a7e8cbce38c33e776b8e725b426fc973babbed0`.

28.1. **fixed** — P2, исходная строка 68. Даты смешивают выпуск весов и публикацию paper без маркировки события: карточка Mixtral даёт декабрь 2023, Qwen2 — июнь 2024, Gemma2 — июнь, здесь другие месяцы.

Результат: В календарных таблицах явно указан тип события. Для Mixtral, Qwen2, Gemma/Gemma 2 разведены публикация статьи и выпуск весов, с прямыми первичными ссылками. По arXiv дополнительно исправлена дата подачи Gemma 2 на 2024-07-31.

28.2. **fixed** — P2, исходная строка 98. Запись сводит Nemotron3 к post-training и пропускает главное для этой хронологии — hybrid Mamba/attention MoE.

Результат: Nemotron 3 теперь описан как Mamba-2/attention MoE, с отдельными LatentMoE/MTP/NVFP4 у Super. Строка исходного семейства отнесена к декабрю 2025 года, а не только 2026 post-training.

28.3. **fixed** — P2, исходная строка 111. Абсолютные локальные URL требуют проверки base /bookvar при Pages; wiki-ссылки в остальном корпусе проходят иной converter.

Результат: 11 абсолютных agent-route ссылок заменены каноническими wiki-ссылками с точными заголовками. Исполненный тест импортировал publicationHref и convertWikiSyntax при /bookvar: raw root-relative URL остаётся без base, wiki URL получает base. Полный Pages HTML build оставлен primary; результат не выдан за полный site build.

Проверенные первичные материалы:

- [Источник](https://mistral.ai/news/mixtral-of-experts/)
- [Источник](https://qwenlm.github.io/blog/qwen2/)
- [Источник](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-open-models/)
- [Источник](https://blog.google/innovation-and-ai/technology/developers-tools/google-gemma-2/)
- [Источник](https://arxiv.org/abs/2408.00118)

Численные/исполняемые проверки:

- Executed actual adapter functions with PUBLICATION_BASE_PATH=/bookvar: raw Markdown remains /textbook/agents/formal-math/; wiki -> /bookvar/textbook/agents/formal-math/#copra. Full generated HTML validation delegated to primary.

Иллюстрации: No new figure; original embeds retained

## Передача на приёмку

Коммиты, push, merge, deploy и полная сборка не выполнялись. Общие реестры не редактировались этим агентом; запись Kimi и полную MIT notice внёс основной агент в своей области владения. Финальную проверку generated HTML под `/bookvar` и независимую содержательную приёмку выполняет основной агент. Foundations в этом блоке не изменялся.
