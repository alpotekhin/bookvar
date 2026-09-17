# Advanced: завершение редакционного прохода

Статус: complete, 2026-09-15. Полностью прочитаны **40/40 RU** страниц исходного аудита и **7/7** существующих авторских EN counterparts. Изменены и полностью перечитаны **31 RU + 4 EN**. Из 55 исходных замечаний: **48 fixed, 7 already-fixed, 0 open**.

Это результат содержательного чтения, а не grep-pass. Byte-exact baseline — состояние уже dirty worktree перед новыми изменениями этого агента; прежние правки не приписаны этому проходу. У некоторых fixed-замечаний часть исходной ошибки уже была устранена (restore внутри цикла, approval перед execute), а новая работа закрыла оставшиеся требования контракта и примера. Это явно указано в resolution.

[Полный машиночитаемый отчёт](advanced.json) сохраняет все исходные поля каждого finding, status/resolution, SHA-256 до/после, evidence полного чтения и перечитывания, источники, проверки и ограничения. Снимки находятся в `.superpowers/sdd/2026-09-15-close-editorial-findings/task-3-before/` с исходными путями.

## Границы и сохранность

Точный набор — 40 RU путей из `docs/audit/2026-09-15/advanced.json`. Семь EN mappings взяты из `publishing/navigation.yml`; языковые fallback не дублировались, отсутствующие переводы не создавались. Существенно расширенные EN VLM главы сохранены; изменены только четыре с остаточными проблемами.

Все 47 исходных файлов имеют проверенный snapshot SHA. Порядок и количество исходных figure references сохранены в 47/47 файлах; explicit anchors не потеряны. В главе 70 сохранены тексты исходных заголовков внутри четырёх крупных частей; в главе 67 уточнён один заголовок с сохранением старого явного якоря. Новых Mermaid/SVG или собственных инфографик нет. Shared navigation, registries, schema, rendering, AGENTS и CI не изменялись этим агентом.

## Выполненная проверка

- Свежие CPU-тесты: RLVR verifier — 15 assertions PASS; provenance пар — все 6 и соседние 3 пары PASS; GRPO — centering, zero variance, clipping signs и padding PASS.
- Branch isolation — альтернативы дают 11 и 20; старый parser воспроизводит ожидаемый ValueError, исправленный проходит 4 случая.
- SFT template — проверены Jinja-текст и generation span. Это **не** запуск tokenizer-mask API и не обучение.
- Арифметика CE, hard/multi-positive, BM25, RRF, MaxSim, DDIM, pass@k и symbolic-regression validation пересчитана на CPU: PASS. Точный код и stdout сохранены в JSON.
- 14 JSON code fences синтаксически разобраны без ошибок. Это не полная JSON Schema validation.
- Полное чтение изменённых страниц проведено после правок; последующие локальные корректировки перечитаны ещё раз. Проверки final report и `git diff --check` указаны в JSON после финальной сверки.

По навыку verification-before-completion заявления об исправлениях ограничены выполненными проверками. Полная сборка и browser/render review не выполнялись; обучение, Lean/Coq, tokenizer-mask API, live tool/security integration не запускались. Нет утверждения, что перепроверены все исторические benchmark-числа или все ссылки.

## Источники и ограничения доступа

Ключевые уточнения сверялись с первоисточниками: [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html), [DPO](https://arxiv.org/html/2305.18290v2), [Dr. GRPO](https://arxiv.org/html/2503.20783v1), [s1](https://arxiv.org/html/2501.19393), [TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/sft_trainer), [BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html), [ColBERT](https://arxiv.org/abs/2004.12832), [Moshi](https://arxiv.org/html/2410.00037v2), [Qwen3-VL](https://arxiv.org/html/2511.21631v1), [DDIM](https://arxiv.org/abs/2010.02502), [SWE-agent](https://arxiv.org/html/2405.15793v2) и [официальной проверкой Lean proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/). Полный выбранный набор привязан к страницам в JSON; отсутствие URL означает локальную редакционную/контрактную проверку, а не проверку всех внешних утверждений.

Для Berkeley доступны прямые [Yu Su PDF](https://rdi.berkeley.edu/adv-llm-agents/slides/language_agents_YuSu_Berkeley.pdf) и [Swarat Chaudhuri PDF](https://rdi.berkeley.edu/adv-llm-agents/slides/swarat.pdf). Xinyun Chen PDF превысил размерный лимит web-инструмента — это не 404. Локальная debate-фигура осмотрена, числа сверены с [Huang et al., Table 4](https://arxiv.org/html/2310.01798v1#S3.SS3). URL курса с завершающим `/sp25/` вернул 404; глобальный link-repair не заявлен.

Точные pinned URL двух LLaVA transfer-фигур и VQ-VAE учебной иллюстрации совпали с существующим `05 Источники/asset-registry.yml`. Их HTTP-доступность не подтверждена: web fetch ошибки, shell proxy/DNS ошибки. Идентичность исходного объекта подтверждена отдельно от доступности URL. Регистры и бинарные фигуры не изменялись.

## Все страницы

Для неизменённых страниц повторное чтение не требовалось: полный первый проход относится к тем же байтам. SHA каждой страницы и EN counterpart приведены в JSON.

| № | RU страница | Findings | Новое изменение | Полное чтение / reread |
|---:|---|---|---|---|
| 1 | 00 Карта модуля и источники.md | нет | нет | full / not-required-unchanged |
| 2 | 01 SFT, RLHF и DPO.md | нет | нет | full / not-required-unchanged |
| 3 | 02 RLVR, reasoning и distillation.md | нет | нет | full / not-required-unchanged |
| 4 | 00 Карта модуля и источники.md | 1 fixed; 0 already-fixed | да | full / full |
| 5 | 01 Vision-language и omni models.md | нет | нет | full / not-required-unchanged |
| 6 | 00 Agent Harness и Context Engineering — карта модуля.md | нет | нет | full / not-required-unchanged |
| 7 | 01 SFT и instruction data.md | 2 fixed; 0 already-fixed | да | full / full |
| 8 | 02 Preference data.md | 1 fixed; 0 already-fixed | да | full / full |
| 9 | 03 Reward modeling.md | 2 fixed; 0 already-fixed | да | full / full |
| 10 | 04 Policy gradient и PPO для LLM.md | 1 fixed; 0 already-fixed | да | full / full |
| 11 | 05 DPO.md | 1 fixed; 0 already-fixed | да | full / full |
| 12 | 06 RLVR и verifiers.md | 0 fixed; 2 already-fixed | нет | full / not-required-unchanged |
| 13 | 07 GRPO и DeepSeek-R1.md | 3 fixed; 0 already-fixed | да | full / full |
| 14 | 08 Reasoning distillation.md | 1 fixed; 1 already-fixed | да | full / full |
| 15 | 01 Test-time compute.md | 2 fixed; 0 already-fixed | да | full / full |
| 16 | 01 Embeddings и retrieval.md | 2 fixed; 0 already-fixed | да | full / full |
| 17 | 02 RAG как система.md | 2 fixed; 0 already-fixed | да | full / full |
| 18 | 60 Embeddings и metric learning.md | 3 fixed; 0 already-fixed | да | full / full |
| 19 | 61 Retrieval — от BM25 до dense и hybrid.md | 1 fixed; 0 already-fixed | да | full / full |
| 20 | 62 Reranking — cross-encoder и late interaction.md | 3 fixed; 0 already-fixed | да | full / full |
| 21 | 63 RAG — полный конвейер.md | 1 fixed; 0 already-fixed | да | full / full |
| 22 | 64 Мультимодальные модели.md | нет | нет | full / not-required-unchanged |
| 23 | 64a Connectors и fusion.md | 0 fixed; 1 already-fixed | да | full / full |
| 24 | 64b Разрешение, tiling и пространственные позиции.md | 2 fixed; 0 already-fixed | да | full / full |
| 25 | 64c Обучение VLM — alignment, instruction tuning и данные.md | 2 fixed; 0 already-fixed | да | full / full |
| 26 | 64d Документы, OCR и visual grounding.md | 1 fixed; 0 already-fixed | да | full / full |
| 27 | 64e Видео, аудио и omni-модели.md | 0 fixed; 2 already-fixed | да | full / full |
| 28 | 64f Оценивание, отказы и serving VLM.md | нет | да | full / full |
| 29 | 64g Диффузионные и flow-модели изображений.md | 2 fixed; 0 already-fixed | да | full / full |
| 30 | 64h Единая мультимодальная последовательность и Chameleon.md | 2 fixed; 0 already-fixed | да | full / full |
| 31 | 01 Tool use и agents.md | нет | нет | full / not-required-unchanged |
| 32 | 02 Evaluation и воспроизводимость.md | нет | нет | full / not-required-unchanged |
| 33 | 65 Tool use — от вызова функции к действию.md | 1 fixed; 0 already-fixed | да | full / full |
| 34 | 66 Agent harness и context engineering.md | 1 fixed; 0 already-fixed | да | full / full |
| 35 | 67 Память, планирование и оркестрация агентов.md | 2 fixed; 0 already-fixed | да | full / full |
| 36 | 68 Оценивание агентных систем.md | 2 fixed; 0 already-fixed | да | full / full |
| 37 | 69 Coding, web и computer-use agents.md | 2 fixed; 1 already-fixed | да | full / full |
| 38 | 70 Формальные доказательства и математические агенты.md | 2 fixed; 0 already-fixed | да | full / full |
| 39 | 71 Агенты научного поиска и discovery.md | 2 fixed; 0 already-fixed | да | full / full |
| 40 | 72 Безопасность агентных систем.md | 1 fixed; 0 already-fixed | да | full / full |

## Авторские EN counterparts

| Страница | Изменена | Чтение / reread |
|---|---|---|
| en/00 Textbook/16 Multimodal Models/64 Multimodal models.md | да | full / full |
| en/00 Textbook/16 Multimodal Models/64a Connectors and fusion.md | нет | full / not-required-unchanged |
| en/00 Textbook/16 Multimodal Models/64b Resolution, tiling, and spatial positions.md | да | full / full |
| en/00 Textbook/16 Multimodal Models/64c Training VLMs — alignment, instruction tuning, and data.md | да | full / full |
| en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding.md | да | full / full |
| en/00 Textbook/16 Multimodal Models/64e Video, audio, and omni models.md | нет | full / not-required-unchanged |
| en/00 Textbook/16 Multimodal Models/64f Evaluation, failure modes, and VLM serving.md | нет | full / not-required-unchanged |

## Все 55 исходных замечаний

Строка в ID относится к порядку исходной страницы, а `line` оригинального аудита сохраняется в JSON и не выдаётся за текущую строку после вставок.

| ID | Статус | Resolution |
|---|---|---|
| advanced-04-1 | fixed | Заголовок уточнён: это указатель актуального маршрута, а не несуществующая карта источников. |
| advanced-07-1 | fixed | Разведены производная CE по target-logit и параметрический градиент с Jacobian; приведены точные значения. |
| advanced-07-2 | fixed | Игрушечная последовательность теперь говорит о сохранении порядка TCP и не утверждает, что UDP быстрее. |
| advanced-08-1 | fixed | Provenance сохраняется отдельно как chosen_policy/rejected_policy по ID выбранных ответов. |
| advanced-09-1 | fixed | RLVR отделён от оценки и SFT-фильтрации тем же верификатором. |
| advanced-09-2 | fixed | Добавлен явный маршрут расширения после DPO без удаления исследовательских механизмов. |
| advanced-10-1 | fixed | Устаревание данных объяснено mismatch старой/текущей policy; ошибка logprobs другого генератора выделена отдельно. |
| advanced-11-1 | fixed | Частные производные DPO и реальные изменения вероятностей общих параметров разведены; подпись рисунка согласована. |
| advanced-12-1 | already-fixed | До прохода parser уже проверял последнюю строку через fullmatch и имел отрицательные регрессионные тесты. |
| advanced-12-2 | already-fixed | До прохода уже указаны корректные награды (2,0,2,1) и объяснён shaping shortcut. |
| advanced-13-1 | fixed | Показаны разные веса ответов 2/20 и численный результат 1.5 против 1.909. |
| advanced-13-2 | fixed | Доказательство (G-1)/G выписано отдельно без std/length/clipping и с iid предпосылками. |
| advanced-13-3 | fixed | Удалён повтор В статье. |
| advanced-14-1 | already-fixed | До прохода уже исправлены s1K-1.1, DeepSeek-R1 и смысл grader correctness; сверено с s1. |
| advanced-14-2 | fixed | Закреплён TRL 0.23.1, EOS, template generation mask и fail-fast probe до trainer; полный training не выполнялся. |
| advanced-15-1 | fixed | Ограничение thinking отделено от final answer, указаны раздельные бюджеты. |
| advanced-15-2 | fixed | Retrieval не обещает проверенности: доказательность зависит от корпуса и проверки применимости. |
| advanced-16-1 | fixed | Hit и recall разведены формулами и примером с тремя gold-документами. |
| advanced-16-2 | fixed | В начале explicit карта переноса; все уникальные вычисления contrastive/BM25/RRF/MaxSim перенесены в60–62, старый текст и фигуры сохранены. |
| advanced-17-1 | fixed | Добавлен последовательный trace исходник/chunks/ранги/context/claim с причиной потери исключения. |
| advanced-17-2 | fixed | Гарантия заменена гипотезой с oracle-context контрольным опытом. |
| advanced-18-1 | fixed | Добавлен численный batch CE и false-negative multi-positive сценарий. |
| advanced-18-2 | fixed | Recall и hit даны отдельно с явными знаменателями. |
| advanced-18-3 | fixed | Документные изменения отделены от query-only изменений и переоценки качества. |
| advanced-19-1 | fixed | Определены BM25 параметры/IDF; добавлены ручной расчёт и RRF двух списков. |
| advanced-20-1 | fixed | Position bias отнесён к видимому списку/паре; pointwise rank не видит. |
| advanced-20-2 | fixed | Candidate recall и oracle/model nDCG сравниваются в согласованных единицах на одном примере. |
| advanced-20-3 | fixed | Добавлены матрицы2×3 двух документов и изменение порядка. |
| advanced-21-1 | fixed | Убран обязательный характер sparse channel в подписи. |
| advanced-23-1 | already-fixed | До прохода в RU и EN уже корректно H_v=Z_vW; размерности совпадают. |
| advanced-24-1 | fixed | Геометрическое обратное отображение отделено от невосстановимых пикселей; EN уточнён. |
| advanced-24-2 | fixed | Шаг сетки patch tokens отделён от внутрипатчевой различимости. |
| advanced-25-1 | fixed | Добавлены прямые pinned URL двух transfer-иллюстраций и Li et al. рядом с объектами RU/EN; точные URL совпадают с существующим asset-registry. Доступность по HTTP не подтверждена из-за ошибок fetch, это не скрыто. |
| advanced-25-2 | fixed | Оформлено внешнее расширение: report не содержит точного нормировочного контракта, формула не выдумана; RU/EN согласованы. |
| advanced-26-1 | fixed | В RU и EN различены unit_price/quantity/line_total; число 900 больше не объявляется суммой двух единиц. |
| advanced-27-1 | already-fixed | До прохода RU/EN уже описывали text→semantic→acoustic и Temporal/Depth с точной ссылкой eq.2/6; сверено с Moshi. |
| advanced-27-2 | already-fixed | До прохода RU/EN явно отличали position-wise projector от сокращающего длину pooling/resampler. |
| advanced-29-1 | fixed | Определены beta/alpha/bar-alpha; добавлен скалярный DDIM eta=0 переход 0.8→0.85218 с формулой и границей интерпретации. |
| advanced-29-2 | fixed | Исправлена LaTeX-команда epsilon и display delimiters; текстовый синтаксис проверен, browser/render не запускался. |
| advanced-30-1 | fixed | Lossy tokenizer с decoder отделён от обратимого кодирования. |
| advanced-30-2 | fixed | Названы авторы метода VQ-VAE и добавлен pinned URL учебной иллюстрации Stanford CS336 рядом с рисунком; точный объект сверен с существующим asset-registry, HTTP availability не подтверждена. |
| advanced-33-1 | fixed | Добавлен полный schema→ошибка→исправление→observation→независимая проверка trace. |
| advanced-34-1 | fixed | Добавлен unknown-write handoff с key, task state, плохим/исправленным summary и двумя инвариантами. |
| advanced-35-1 | fixed | Внешняя память отделена от параметрической; старый heading anchor сохранён. |
| advanced-35-2 | fixed | Каждый Berkeley figure получил прямой PDF-page URL и автора; debate привязан к проверенной Table4 с моделью и бюджетом. |
| advanced-36-1 | fixed | pass@k сохранён как исследовательская coverage-метрика; delivered selected-answer success отделён. Добавлен toy n10/c2/k3. |
| advanced-36-2 | fixed | Выписаны 1−56/120=.5333 и plugin .488 с объяснением конечной выборки. |
| advanced-37-1 | fixed | Restore уже был внутри for до прохода. Добавлены explicit parent/action/observation child и исполнимый тест альтернатив 11/20. |
| advanced-37-2 | already-fixed | До прохода подпись уже явно называла ACI дополнением Bash и отделяла sandbox; сверено с SWE-agent Table4. |
| advanced-37-3 | fixed | Добавлен конкретный split parser, failing input/ValueError, две гипотезы, минимальный diff и 4 проверки. |
| advanced-38-1 | fixed | Добавлены sorry/custom-axiom отрицательные примеры, transitively checked allowlist, trusted challenge и sandbox/comparator boundary. |
| advanced-38-2 | fixed | Существующие механизмы сгруппированы в 4 части; исходные заголовки и explicit anchors сохранены с понижением уровня, изображения не удалены. |
| advanced-39-1 | fixed | Добавлен toy r/y train/validation, fitted 3.6/r и 4/r², MSE+operator penalty, selection и единственная final extrapolation test. |
| advanced-39-2 | fixed | LaSR схемы получили точные direct PDF-page ссылки и авторов; URL deck доступен через web. |
| advanced-40-1 | fixed | Approval уже стоял до execute. Добавлены привязка identity/resource/args/version, invalidation и atomic compare-and-set boundary. |

## Передача

Следующий шаг — независимая приёмка parent/reviewer и общие build/render проверки под владением primary. Этот агент не начинал новый scope, не делал commit/push/deploy/new branch и не выполнял полную сборку.
