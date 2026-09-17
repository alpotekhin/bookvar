# Справочник и обзоры направлений — завершение редакционного прохода

Статус: готово к независимой приёмке, не является результатом общей сборки.

Полностью прочитаны все **45 RU-страниц** точного поднабора `reference-practice.json` с префиксами `01 Справочник/` и `03 Исследовательские линии/`. Разобраны **57 исходных замечаний: 48 fixed, 9 already-fixed, 0 rejected-with-evidence, 0 open**. Изменены и полностью перечитаны 38 страниц; семь оставлены без изменений. Во всех 45 записях manifest отсутствуют authored EN mappings: fallback не выдавался за перевод.

Перед правками сняты 45 byte-exact snapshots. Финальная сверка SHA подтверждает их соответствие исходному состоянию этого прохода. Полные SHA before/after, исходные формулировки всех findings, решения, источники и свидетельства чтения находятся в JSON.

## Закрытые замечания по страницам

| № | Страница | Fixed / Already-fixed | Результат |
|---|---|---:|---|
| 1 | 01 Справочник/_index.md | 0 / 0 | прочитана, без новых правок |
| 2 | 01 Справочник/Архитектурные паттерны/Decoder-only.md | 1 / 0 | исправлена и перечитана |
| 3 | 01 Справочник/Архитектурные паттерны/Encoder-Decoder.md | 1 / 0 | исправлена и перечитана |
| 4 | 01 Справочник/Архитектурные паттерны/Encoder-only.md | 1 / 0 | исправлена и перечитана |
| 5 | 01 Справочник/Математика и DL/Нормализация.md | 1 / 0 | исправлена и перечитана |
| 6 | 01 Справочник/Позиционные представления/RoPE.md | 1 / 0 | исправлена и перечитана |
| 7 | 01 Справочник/Текст и токенизация/Токенизация.md | 0 / 1 | прочитана, без новых правок |
| 8 | 01 Справочник/Agents/Agent mechanisms.md | 1 / 1 | исправлена и перечитана |
| 9 | 01 Справочник/Attention/Cross-Attention.md | 1 / 0 | исправлена и перечитана |
| 10 | 01 Справочник/Attention/GQA.md | 0 / 0 | прочитана, без новых правок |
| 11 | 01 Справочник/Attention/MHA, MQA и GQA.md | 0 / 0 | прочитана, без новых правок |
| 12 | 01 Справочник/Attention/MHA.md | 0 / 0 | прочитана, без новых правок |
| 13 | 01 Справочник/Attention/MLA.md | 1 / 0 | исправлена и перечитана |
| 14 | 01 Справочник/Attention/MQA.md | 0 / 0 | прочитана, без новых правок |
| 15 | 01 Справочник/Attention/Self-Attention.md | 2 / 0 | исправлена и перечитана |
| 16 | 01 Справочник/Evaluation/Evaluation.md | 1 / 0 | исправлена и перечитана |
| 17 | 01 Справочник/FFN и MoE/Dense FFN и gated activations.md | 1 / 0 | исправлена и перечитана |
| 18 | 01 Справочник/FFN и MoE/Mixture of Experts.md | 1 / 1 | исправлена и перечитана |
| 19 | 01 Справочник/FFN и MoE/SwiGLU.md | 0 / 0 | прочитана, без новых правок |
| 20 | 01 Справочник/Inference/FlashAttention.md | 1 / 0 | исправлена и перечитана |
| 21 | 01 Справочник/Inference/KV-cache.md | 1 / 0 | исправлена и перечитана |
| 22 | 01 Справочник/Inference/PagedAttention.md | 1 / 0 | исправлена и перечитана |
| 23 | 01 Справочник/Inference/SGLang и RadixAttention.md | 1 / 1 | исправлена и перечитана |
| 24 | 01 Справочник/Inference/Triton и GPU kernels.md | 1 / 1 | исправлена и перечитана |
| 25 | 01 Справочник/Inference/vLLM — анатомия inference engine.md | 2 / 1 | исправлена и перечитана |
| 26 | 01 Справочник/Post-training/DPO.md | 1 / 0 | исправлена и перечитана |
| 27 | 01 Справочник/Post-training/GRPO.md | 1 / 0 | исправлена и перечитана |
| 28 | 01 Справочник/Post-training/Reward Model.md | 1 / 0 | исправлена и перечитана |
| 29 | 01 Справочник/Post-training/RLHF.md | 1 / 0 | исправлена и перечитана |
| 30 | 01 Справочник/Post-training/RLVR.md | 1 / 0 | исправлена и перечитана |
| 31 | 01 Справочник/Post-training/SFT.md | 2 / 0 | исправлена и перечитана |
| 32 | 01 Справочник/Post-training/Verifiable Reward.md | 1 / 0 | исправлена и перечитана |
| 33 | 01 Справочник/Retrieval/Embeddings.md | 1 / 0 | исправлена и перечитана |
| 34 | 01 Справочник/Retrieval/RAG.md | 1 / 0 | исправлена и перечитана |
| 35 | 01 Справочник/Retrieval/Retrieval.md | 2 / 0 | исправлена и перечитана |
| 36 | 01 Справочник/Security и Robustness/01 Security и privacy.md | 1 / 2 | исправлена и перечитана |
| 37 | 01 Справочник/Security и Robustness/02 Robustness.md | 3 / 0 | исправлена и перечитана |
| 38 | 03 Исследовательские линии/Agentic learning.md | 2 / 0 | исправлена и перечитана |
| 39 | 03 Исследовательские линии/RLHF → DPO → RLVR.md | 2 / 0 | исправлена и перечитана |
| 40 | 03 Исследовательские линии/Reasoning и test-time compute.md | 2 / 0 | исправлена и перечитана |
| 41 | 03 Исследовательские линии/Sparse и MoE-модели.md | 2 / 0 | исправлена и перечитана |
| 42 | 03 Исследовательские линии/_index.md | 1 / 0 | исправлена и перечитана |
| 43 | 03 Исследовательские линии/Длинный контекст.md | 2 / 0 | исправлена и перечитана |
| 44 | 03 Исследовательские линии/Синтетические данные и дистилляция.md | 1 / 0 | исправлена и перечитана |
| 45 | 03 Исследовательские линии/Эффективный attention.md | 0 / 1 | исправлена и перечитана |

## Что проверено

- Четыре группы CPU-проверок повторно выполнены: нормализации/RoPE/MLA/Wilson; KV и блочное расписание/Triton; DPO/GRPO/RM/RLHF/SFT/метрики поиска; Laplace/PSI/нормы атак/Huber/линейное attention. Все завершились с кодом 0. Код и полный вывод сохранены в JSON.
- Все 68 исходных вставок рисунков, их порядок, source-unit IDs и ссылки атрибуции сохранены. Рисунки MoE и Longformer перенесены к соответствующему объяснению без перерисовки; Lost in the Middle — к оцениванию.
- Удалённые или переименованные заголовки сохраняют явные якоря: «Лестница обучения», «Тезис», «Споры», «Как изменилось оценивание».
- Новые wiki-ссылки разрешаются в существующие файлы. Все 45 routes соответствуют manifest. Парность fenced-блоков и display-формул проверена; scoped `git diff --check` — PASS.

## Уже исправлено до этого прохода

- 01 Справочник/Текст и токенизация/Токенизация.md — Before this pass the definition already avoided reversibility claims, gave Hello/hello and byte-level/no-loss conditions; verified against HF normalizers.
- 01 Справочник/Agents/Agent mechanisms.md — Before this pass the full Formal verifier section already separated Lean kernel checks, compilation and finite tests, with sorting counterexample and official Lean documentation.
- 01 Справочник/FFN и MoE/Mixture of Experts.md — Pre-edit page already explained V3 bias-based routing and weak sequence-wise auxiliary term; source confirmed exact distinction.
- 01 Справочник/Inference/SGLang и RadixAttention.md — Pre-edit page already required fixed weights, LoRA, positions/masks and multimodal inputs and described extra_key; live RadixCache source confirms namespace behavior.
- 01 Справочник/Inference/Triton и GPU kernels.md — Pre-edit caption already correctly said column index changes first under row-major, with GPU scheduling caveat. Added explicit 2x3 coordinate sequences as extra explanation.
- 01 Справочник/Inference/vLLM — анатомия inference engine.md — Pre-edit page already separated 90% SLO attainment from 9 req/s goodput and gave DistServe's max-arrival-per-GPU convention; live paper checked.
- 01 Справочник/Security и Robustness/01 Security и privacy.md — Before snapshot already distinguishes record/user adjacency and per-example versus user clipping; verified Dwork–Roth Definition 2.4 and group privacy.
- 01 Справочник/Security и Robustness/01 Security и privacy.md — Before snapshot already calls delta additive slack in defining inequality, not probability of catastrophic disclosure; source definition checked.
- 03 Исследовательские линии/Эффективный attention.md — Pre-edit page already separates sparse edge removal from linear kernel/aggregation with all prior positions. Added source-linked cumulative S/z mechanism and CPU equivalence check to strengthen existing correct account, not to count the already-resolved P1 as a new fix.

## Источники и границы проверки

Механизмы проверялись по первичным статьям, официальной документации PyTorch/TRL/vLLM/SGLang/Triton, учебнику Dwork–Roth, Stanford IR и исходным страницам курсов. Точные URL привязаны к страницам в JSON. Для новых числовых кейсов отдельно указаны соглашения: population std в GRPO, однотокенный сдвиг SFT, логические и выделенные KV-блоки, единица защиты DP, сглаживание PSI. Эксперименты Aguvis и InSTA изложены в пределах их протоколов; они не воспроизведены нами.

- Completion means adjudication of this exact audit subset, not independent replication of all cited model results or a new audit of all historical claims.
- No full publishing build, browser/Obsidian rendering, GPU kernels, serving engine, large training or adversarial attack evaluation was run.
- Preserved 68 existing figure embeds, source-unit IDs and attribution. No newly drawn SVG/Mermaid or replacement illustrations. Existing licensing was not re-audited; no new source image was copied.
- Authored EN mappings: 0. Locale fallback is not counted as a translation; none was created.
- Dwork–Roth author PDF failed to fetch; the same primary textbook was verified via the course-hosted mirror listed on Security page evidence. Original canonical source URL was retained.
- Primary source checks targeted findings and material additions; scores such as Aguvis/InSTA are author-reported, not independently reproduced.
- Repository was dirty before assignment; snapshots and before/after hashes distinguish only this task's changes, not HEAD diff. Shared navigation/registry/build files were read where relevant but not edited.
- Some retained historical prose uses English technical labels; this pass corrected all original language findings and new explanatory prose, not every pre-existing English term across the entire reference library.

Навыки executing-plans и llm-wiki определили порционный проход и сохранение связей; verification-before-completion — повторные проверки перед итогом. Никаких commit/push/build, правок общих регистров или новых агентов не было.
