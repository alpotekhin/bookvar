# Berkeley deferred queue: содержательное продолжение

Дата: 2026-09-15. Рабочая область: `ml-systems-ingestion`.
Статус: **3 семантические единицы содержательно встроены; общие карты и реестр нового изображения ожидают отдельного подключения root.** Это не утверждение о завершённой сборке, публикации или приёмке всего курса.

## Граница работы и результат

Исходное задание: очередь Berkeley из `docs/audit/2026-09-15-en/course-state.md`, текущие source coverage/editorial maps и существующие тематические главы. Приоритет — объяснение механизмов с первоисточниками; не перенос всех слайдов и не отдельная глава под каждую статью.

Владение объявлено до изменений: RU/EN 64e, RU 71, этот отчёт и byte-exact snapshots. Глава 69 прочитана без изменений. По отдельному разрешению root добавлены оригинальная ESCHER Figure 2, исходный PDF и её растровая версия. Общие hubs, registries, navigation, semantic-review, editorial-map, coverage, visuals и importer не изменялись. Не выполнялись git-команды, полные сборки, commit/push/deploy, создание веток или агентов.

| Source unit | Проверенное состояние до прохода | Результат прохода |
|---|---|---|
| `meeting-07-slides-xgen-video`, PDF 84–98 | Общая механика video tokens уже объяснена в 64e; конкретного механизма xGen не было. В карте оставалась причина «Task 8 owns…». | **integrated-content / ledger-pending**: temporal compression, различие pooling и TTM, численный token budget, граница encoder/LLM cost, учебный пример в RU и существующей EN. |
| `meeting-07-slides-gens`, PDF 99–105 | Общая temporal sampling тема была, question-aware GenS mechanism отсутствовал. Название «generalist foundation agents» в semantic-review неверно. | **integrated-content / ledger-pending**: Generative Frame Sampler, question-conditioned selection, frame IDs, score/ranking, окна, бюджет и временная привязка в RU/EN 64e. |
| `meeting-11-slides-visual-concept-discovery`, PDF 82–88 | Глава 71 содержательно объясняла LaSR, но не ESCHER и смену внешнего численного evaluator на learned VLM critic. | **integrated-content / ledger-pending**: ESCHER с classifier equation, refinement loop, pseudo-confusion example, ограничениями и оригинальной Figure 2 в RU 71. |
| `meeting-07-slides-agenttrek`, PDF 33–45 | Coverage уже `integrated`; глава 69 уже содержит tutorial collection → guided replay → filtering, ограничения teacher prior и evaluation. | **already-integrated**: полный текст главы и первичный paper/project/repository проверены; дублирующий раздел не добавлен. Устарела именно причина source-only visual row, а не отсутствие механизма в главе. |
| `meeting-06-slides-plan-sequence-learn`, PDF 107–116 | Source-only; в canonical navigation нет связного маршрута embodied manipulation. | **deferred-with-evidence**: сохранить source-only. Motion planning и low-level RL требуют отдельного пояснения робототехнической среды, геометрии, контакта и termination; вставка в web/GUI planning размоет различие исполнительных контуров. |
| `meeting-06-slides-proprietary-adjacent-demo`, PDF 118–121 | Archive-only, pending explicit rights review. | **rights-blocked / source-only**: не воспроизводились, не использовались как новые учебные фигуры; отдельного подтверждения прав не получено. |
| `meeting-06-slides-nvidia-confidential-page`, PDF 117 | `excluded`, permanent do-not-reuse. | **excluded-preserved**: никаких изменений; course-level permission не отменяет page-specific запрет. |

Итого: 7 проверенных решений очереди; 3 новые содержательные интеграции, 1 уже существующая интеграция, 2 осознанно не продвинутые единицы, 1 сохранённое исключение. Это семантические единицы, **не количество интегрированных слайдов**.

## Полное чтение и сохранность

Полностью прочитаны 4 существующие главы: RU 64e (186 строк до), EN 64e (129), RU 71 (325), RU 69 (822). Все 3 изменённые главы после правок перечитаны полностью: RU 64e 1–265, EN 64e 1–202, RU 71 1–391. Для RU 64e после последней грамматической правки выполнено ещё одно полное чтение 1–265. После финального переключения ESCHER embed с preview PNG на original SVG глава 71 ещё раз прочитана полностью, 1–391. Это содержательное чтение, а не поиск только по source IDs.

Из source archive прочитаны полные text-extract spans: meeting 7 PDF 33–45 (AgentTrek), 84–98 (xGen), 99–105 (GenS); meeting 11 PDF 82–88 (ESCHER); meeting 6 PDF 107–116 (Plan–Sequence–Learn). Чтение text extracts не выдаётся за визуальную проверку каждого слайда. Новый ESCHER оригинал отдельно осмотрен по полному PDF page 3 и полному rasterized Figure 2.

Существующий authored EN mapping проверен в `publishing/navigation.yml`: только RU 64e имеет `source_en` на изменённую EN 64e. Для 69 и 71 authored EN mapping отсутствует; fallback не называется переводом, новые переводы не создавались.

Точные snapshots созданы через apply_patch **до** любых новых правок и затем сверены SHA-256 с исходными файлами. Корень снимков:

`.superpowers/sdd/2026-09-15-close-editorial-findings/task-6-berkeley-follow-through-before/`

Внутри сохранены исходные относительные пути.

### 00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md

- Before SHA-256: `ead730528af1f709ec33fee3fd495727d6d1069935739822d588a801f1ea3e11`
- After SHA-256: `5fa2da00592cdf652c45368a90bd8434778e0f1ff0fb241cd98396343fcb9b63`
- Полное повторное чтение: 1–265; snapshot matches: PASS.
- Сохранены все старые headings и source IDs; image embeds 4 → 4.

### en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models.md

- Before SHA-256: `256112e6354adf53bb521c363d44029ca125f821ca5b33c0cfc08a0f37f32a99`
- After SHA-256: `1975c1ee74a813104c013d89b9f40b53eb83772cb4ae4ffb2f59d6ed3272411d`
- Полное повторное чтение: 1–202; snapshot matches: PASS.
- Сохранены все старые headings и source IDs; image embeds 4 → 4.

### 00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery.md

- Before SHA-256: `e86b4ce1126699ed9d1b548da452a2fef79988d5c4ab9f3e32cf35d855c073aa`
- After SHA-256: `670e7741669b1da2d88ef7a763ed7877a6f43cf51645d4abbf0cf34e4b361e22`
- Полное повторное чтение: 1–391; snapshot matches: PASS.
- Сохранены все старые headings и source IDs; image embeds 8 → 9.

Read-only глава `00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents.md`: SHA-256 `bdab71660c7c4718cbf82ba36c1f008bb4fa21c819a9d97f745deac0e599abe3`, изменений этим проходом нет.

Суммарно сохранены 16 из 16 прежних figure embeds в трёх изменённых файлах; добавлена 1 оригинальная научная фигура. Не создавались Mermaid, SVG-схемы собственного сочинения или определения-заглушки.

## Существенные технические решения

### xGen и GenS

xGen вносит отдельный механизм между frame encoder и языковой моделью. Пример статьи: 8 × 729 = 5832 visual tokens, spatial reduction до 8 × 128 = 1024, затем M языковых входных visual tokens. В TTM 128 групп × 4 memory slots = 512 внутренних элементов памяти; это не «512 входных токенов LLM» и не обещание, что любое видео всегда стоит 32 токена. Указана граница конкретного 8-frame checkpoint. Compression уменьшает downstream token load, но не устраняет вычисление frame encoder.

GenS расшифрован как **Generative Frame Sampler**, не generalist foundation agent. Для pinned v1 описана Aria-based selection: вопрос + кадры → выбранные frame IDs/intervals и relevance scores. Разобраны 1 fps, окна не более 256 кадров и итоговый K = min(N_returned, N_budget). Relevance score — средство отбора, не калиброванная вероятность правильного ответа. Локальные frame IDs должны быть возвращены на общую шкалу timestamps. Учебные 600 секунд → 256 + 256 + 88 кадров и 45 кандидатов → бюджет 32 отделены от результатов paper. Учтена стоимость sampler и answer model.

Существующая OneVision source figure сохранена; новые фигуры xGen/GenS **только прямо связаны с оригиналом**, не импортированы. Права model weights/code или лицензия шаблона project website не трактуются как подтверждение прав на paper figures.

### ESCHER

Новый раздел соединён с LaSR через конкретное различие: численное выполнение формулы и trained VLM critic дают разный тип обратной связи. Текст объясняет concept descriptors, similarity vector и class-weighted classifier, zero-shot averaging против обучаемых весов, refinement по конкурентным классам и историю повторных запросов. Обновление библиотеки не тождественно обучению весов VLM.

Псевдопутаница по top-k score не названа обычной confusion matrix с истинными метками. Учебные пары A/B, B/A, A/B, C/B дают счётчики A/B = 3, B/C = 1; этого недостаточно для вывода об accuracy. Пример чашки/миски с закрытой ручкой явно учебный, не заявленный эксперимент статьи. Объяснено, почему правдоподобный verbal concept может быть плохо различим самим critic и закрепить его ошибку.

### AgentTrek и Plan–Sequence–Learn

Для AgentTrek первичный paper, official project и upstream repository доступны; существующий раздел не нуждается в повторном изложении. Подтверждение публикации не означает автоматического разрешения на все course visuals или необходимости добавления новых фигур.

Plan–Sequence–Learn действительно соединяет high-level language planning, motion planning и RL control. Это полезный материал, но для читателя нужен робототехнический контекст наблюдений, геометрической достижимости, переходов между skills и контакта с объектами. В рамках порученного video/discovery/GUI маршрута связной интеграции нет. Source-only здесь — аргументированная граница курса, а не непрочитанная задача.

## Первичные источники: live-проверка 2026-09-15

- xGen: [статья v2, §§2.1–2.3 и model details](https://arxiv.org/html/2410.16267v2); [оригинальная Figure 3](https://arxiv.org/html/2410.16267v2#S2.F3); [официальная 8-frame model card](https://huggingface.co/Salesforce/xgen-mm-vid-phi3-mini-r-v1.5-32tokens-8frames). v2 от июня 2025 — явно pinned уточняющий primary source, не утверждение, что весенний курс использовал все детали поздней ревизии.
- GenS: [статья v1, §§2.1–2.3/3.1](https://arxiv.org/html/2503.09146v1); [оригинальная Figure 1](https://arxiv.org/html/2503.09146v1#S0.F1); [официальный проект](https://generative-sampler.github.io/). Использована v1-конфигурация Aria, не смешанная с более поздними вариантами из текущего сайта.
- ESCHER: [статья v1, §§3–4, Eq.1 и Algorithm 1](https://arxiv.org/html/2504.00185v1); [официальный проект](https://trishullab.github.io/escher-web/); [upstream code](https://github.com/trishullab/escher); [CVPR 2025 primary publication](https://openaccess.thecvf.com/content/CVPR2025/html/Sehgal_Self-Evolving_Visual_Concept_Library_using_Vision-Language_Critics_CVPR_2025_paper.html). Code/README осмотрены как первичный маршрут, upstream experiments не исполнялись.
- AgentTrek: [статья v1, §§2.1–2.2](https://arxiv.org/html/2412.09605v1), [официальный проект](https://agenttrek.github.io/), [upstream repository](https://github.com/xlang-ai/AgentTrek). Доступность repository не заменяет commit-pinned reproduction; paper license не используется как разрешение на любые Berkeley page images.
- Plan–Sequence–Learn: [официальный проект](https://planseqlearn.github.io/), [первичная статья](https://arxiv.org/abs/2405.01534).

Новые существенные механические утверждения сверены live; воспроизведение benchmark результатов не заявляется.

## Новая оригинальная фигура: данные для root registry

Право: статья ESCHER v1 опубликована под [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); лицензия проверена на [arXiv HTML v1](https://arxiv.org/html/2504.00185v1). Отдельное разрешение root на новый source asset и учебное включение получено до загрузки. Это не перенос правового основания на xGen/GenS или proprietary-adjacent course pages.

Все три файла лежат под `00 Учебник/Assets/Figures/curated/course-follow-through/`:

| Файл | Bytes | SHA-256 |
|---|---:|---|
| `escher-figure2-original.svg` | 536804 | `f8cc16778fe268f6c28a4590943b62ef5e1fe5f072577860d87108395aa68c90` |
| `escher-source-2504.00185v1.pdf` | 2136518 | `341ee8bceb3b6217b0b5b53b03424a54e83ff357bae673a54c834cd835233969` |
| `escher-figure2.png` | 589257 | `5d507e55e6283f459cbd6d2937d499df71fe5fe044b0b8eebe062050d8bcf558` |

- Original SVG URL: `https://arxiv.org/html/2504.00185v1/cvpr-escher-figure-2-architecture_v3-9.svg`.
- Source PDF URL: `https://arxiv.org/pdf/2504.00185v1`; Figure 2 находится на physical PDF page 3.
- SVG geometry: 1163 × 540 pt; viewBox 0 0 1163 540. Исходный SVG не изменён.
- Preview-only PNG (не embedded): 2400 × 1114 px, RGB, белый фон. Только rasterization всей исходной фигуры (sharp, density 200, width 2400, flatten white), без crop, redraw, перевода подписей и изменения композиции.
- Оба upstream originals повторно получены и проверены byte-for-byte: PASS.
- Весь PDF page 3 и весь preview PNG визуально осмотрены. Все три панели, стрелки, подписи и заголовки читаемы. Обрезанный QuickLook thumbnail оказался ограничением preview и не используется для публикации; preview PNG проверен отдельно.
- SVG проверен на script/onload/external HTTP references: не обнаружены; встроенные data-image и path glyphs сохранены.
- Embed: **byte-exact `escher-figure2-original.svg`**, RU 71, anchor `escher-visual-concept-library`; route `textbook/agents/scientific-discovery`. Derivation embedded asset: none; copied byte-exact from Original SVG URL. Source raw для копирования SVG — сам сохранённый original SVG; PDF — отдельная неизменённая копия статьи для page/source evidence, не вход в SVG conversion.
- Caption: Sehgal et al., *Self-Evolving Visual Concept Library using Vision-Language Critics*, Figure 2, PDF page 3; CC BY 4.0; unchanged original/no-crop указаны прямо в главе.
- Предлагаемый asset ID: `course-follow-through-escher-figure2` (не записан в registry этим проходом).
- У SVG прозрачный фон и чёрные подписи. При site QA проверить светлый фон иллюстрации, особенно в dark theme; PNG сохраняет белый фон только как вспомогательное превью. Shared renderer/CSS этим проходом не изменялись.
- Не заполнять фиктивные desktop/narrow render evidence: site-render пока не выполнялся. Independent reviewer/acceptance ещё не завершены.

Эта независимая paper Figure 2 **не означает**, что весь блок Berkeley meeting 11 pages 82–88 переиспользован визуально. Его course visual row остаётся source-only.

## Точные предложения для общих карт — применяет root

Ниже **partial replacement overlay** для соответствующих записей `editorial-map.yml`; остальные ключи и записи сохранить. Для этих трёх coverage rows заменить старые source-only records целиком, чтобы удалить устаревшие Task 8 причины:

```json
{
  "coverage": {
    "meeting-07-slides-xgen-video": {
      "disposition": "integrated",
      "destination": "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md",
      "destination_anchor": "xgen-video-temporal-compression"
    },
    "meeting-07-slides-gens": {
      "disposition": "integrated",
      "destination": "00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md",
      "destination_anchor": "gens-question-aware-frame-selection"
    },
    "meeting-11-slides-visual-concept-discovery": {
      "disposition": "integrated",
      "destination": "00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery.md",
      "destination_anchor": "escher-visual-concept-library"
    }
  }
}
```

AgentTrek coverage уже integrated с `agenttrek-tutorials-guided-replay-trajectories`; его не дублировать и не переводить обратно в source-only.

Для следующих visual records заменить только устаревшую причину и evidence (disposition остаётся source-only):

```json
{
  "visuals": {
    "meeting-07-slides-visual-xgen-video": {
      "disposition": "source-only",
      "reason": "The xGen mechanism is integrated in RU/EN 64e; original paper Figure 3 is linked, not locally reproduced. No new figure-specific rights basis was confirmed. Existing OneVision imagery is not reuse of this slide sequence.",
      "evidence": "Lectures/meeting-07-slides.pages.txt#physical-pages=84-98; https://arxiv.org/html/2410.16267v2#S2.F3"
    },
    "meeting-07-slides-visual-gens": {
      "disposition": "source-only",
      "reason": "The GenS frame-selection mechanism is integrated in RU/EN 64e; original paper Figure 1 is linked, not locally reproduced. Code/model or website-template licensing is not treated as paper-figure permission.",
      "evidence": "Lectures/meeting-07-slides.pages.txt#physical-pages=99-105; https://arxiv.org/html/2503.09146v1#S0.F1"
    },
    "meeting-11-slides-visual-visual-concept-discovery": {
      "disposition": "source-only",
      "reason": "ESCHER is integrated in chapter 71 with the independent paper's original CC BY 4.0 Figure 2. The Berkeley pages 82-88 visual sequence was not reproduced and is not promoted by that separate asset.",
      "evidence": "Lectures/meeting-11-slides.pages.txt#physical-pages=82-88; https://arxiv.org/html/2504.00185v1; docs/audit/2026-09-15-completion/course-berkeley-follow-through.md"
    },
    "meeting-07-slides-visual-agenttrek": {
      "disposition": "source-only",
      "reason": "The primary paper and upstream repository are verified and the mechanism is already integrated in chapter 69. This course visual sequence has not been selected with a figure-specific rights basis and rendered-site evidence; no new image is claimed.",
      "evidence": "Lectures/meeting-07-slides.pages.txt#physical-pages=33-45; https://arxiv.org/html/2412.09605v1; https://github.com/xlang-ai/AgentTrek"
    },
    "meeting-06-slides-visual-plan-sequence-learn": {
      "disposition": "source-only",
      "reason": "Retained with its robotics mechanism outside the current canonical route; coherent teaching requires motion-planning, perception and low-level-control context not supplied by the current web/GUI agent chapters.",
      "evidence": "Lectures/meeting-06-slides.pages.txt#physical-pages=107-116; https://planseqlearn.github.io/"
    }
  }
}
```

Plan–Sequence–Learn coverage remains source-only; можно уточнить reason тем же аргументом. Both coverage/visual records p117 остаются excluded, p118–121 — source-only pending explicit rights review.

**Исправление названия GenS требует отдельного semantic-review edit.** Точный locator: `semantic-review.json → decks[7].sections[8]`, `semantic_id: "gens"`. Заменить title `GenS generalist foundation agents` на `GenS: Generative Frame Sampler for long video understanding`. Это не поле coverage overlay: разрешённые `COVERAGE_EDITORIAL_FIELDS` включают disposition/destination/destination_anchor/reason/evidence, но не title. После изменения semantic-review root должен штатно регенерировать производные source-units/coverage и выполнить валидаторы. Не добавлять неподдерживаемый title в editorial-map.

В новом source prose стоят reciprocal IDs и anchors; точные primary URLs указаны в frontmatter/разделах. Root также может исправить stale wording в course-state/hub после общего подсчёта; этот отчёт не переписывает исторический аудит или общие totals.

## Исполняемые проверки

1. Свежий Node checker: SHA snapshots, сохранность всех старых headings/source IDs/figure embeds, существование всех текущих image paths, чётность fences и display-math delimiters. Exit 0 для всех 3 страниц. Before/after SHA приведены выше.
2. Первичные links/anchors xGen `S2.F3`, GenS `S0.F1`, ESCHER Figure 2 проверены live. Ошибочный первоначальный вариант GenS `S1.F1` исправлен до итогового reread.
3. ESCHER SVG/PDF re-fetch byte equality: PASS; preview PNG визуально проверен целиком; опубликованный embed указывает на неизменённый original SVG.
4. Следующий CPU-only toy check выполнен через `/private/tmp/bookvar-editorial-venv/bin/python` (NumPy 2.2.6), exit 0:

```python
import numpy as np
from collections import Counter
assert 8*729 == 5832 and 8*128 == 1024 and 128*4 == 512
frames = np.arange(600)  # t=0,...,599 seconds; teaching convention
windows = [frames[i:i+256] for i in range(0,len(frames),256)]
assert [len(w) for w in windows] == [256,256,88]
assert sum(map(len,windows)) == 600 and windows[1][0] == 256
assert min(45,32) == 32
s = np.array([[.8,.79,.2],[.74,.76,.1],[.72,.71,.3],[.1,.2,.85]])
top2 = np.argsort(-s,axis=1)[:,:2]
counts = Counter(tuple(sorted(pair)) for pair in top2)
assert counts[(0,1)] == 3 and counts[(1,2)] == 1
pred = s.argmax(axis=1)
assert np.mean(pred==pred) == 1 and np.mean(pred==(pred+1)%3) == 0
concept_scores = np.array([.8,.6,.3,.5])
weights = np.array([[.5,.5,0,0],[0,0,.5,.5]])
assert np.allclose(weights@concept_scores,[.7,.4])
print("PASS: xGen 5832 -> 1024; grouped memory 512 (not 32)")
print("PASS: GenS toy windows [256, 256, 88], second-window time 256s, K=32")
print("PASS: concept-weighted scores [.7,.4], pseudo-confusion A/B=3, B/C=1")
print("PASS: unchanged unlabeled scores admit accuracy 1.0 or 0.0 under different labels")
print("CPU arithmetic/invariants only; no upstream model training/inference executed")
```

Вывод:

```text
PASS: xGen 5832 -> 1024; grouped memory 512 (not 32)
PASS: GenS toy windows [256, 256, 88], second-window time 256s, K=32
PASS: concept-weighted scores [.7,.4], pseudo-confusion A/B=3, B/C=1
PASS: unchanged unlabeled scores admit accuracy 1.0 or 0.0 under different labels
CPU arithmetic/invariants only; no upstream model training/inference executed
```

## Ограничения и передача

- Ни model inference/training, ни upstream benchmark reproduction не исполнялись; малые проверки подтверждают только арифметику и логические границы учебных примеров.
- Site build, full course-ingestion validators, desktop/narrow browser rendering и независимая приёмка не выполнялись по границе этого bounded задания. Оригинальный SVG подключён к главе, но shared figure registry ещё должен обновить root.
- Планируемые overlay changes выше не применены. Поэтому нельзя на основании одного этого отчёта увеличивать опубликованные integrated counts или утверждать, что общий ledger уже прошёл проверку.
- Существующая EN 64e исправлена симметрично; никаких отсутствующих переводов не создано.
- Права на новые локальные изображения xGen/GenS не подтверждены: прямые primary links оставлены без local reuse. Для ESCHER есть отдельная CC BY 4.0 основа.
- Meeting 6 p117 не использовалась; pages 118–121 не продвинуты; Plan–Sequence–Learn остаётся явно отложенной по учебному маршруту.
- Этот отчёт не изменяет before/after hashes ранее переданных Advanced и Reference reports. Другие агенты могли внести свои позднейшие изменения в общие файлы; здесь зафиксирован только текущий owned delta.
