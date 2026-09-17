# Точечные исправления после advanced-аудита

Дата: 2026-09-15. Исправлены семь подтверждённых P2 в пяти разрешённых главах. Исходные оценки и read_sha256 в advanced.json сохраняют прочитанную до исправлений версию; эта запись связывает findings с последующими изменениями. Полный аудит охватывает 40 advanced-страниц; отдельный questions-отчёт — 9 страниц и все 100 основных архивных пунктов. Архивные ответы, их английские фрагменты и рисунки не изменены.

## Что исправлено

| Глава | Изменение | Проверка |
|---|---|---|
| 06 RLVR и verifiers | FINAL проверяется только в последней строке, единственный маркер; ASCII decimal grammar задан явно | 15 исполняемых assertions в самой главе, включая trailing text, повторный/нечисловой маркер, пустой ввод, NaN, Infinity, Unicode minus, exponent и reward |
| 06 RLVR и verifiers | При весе формата 1 награды равны (2,0,2,1), неверный ответ не приравнивается к верному | Ручная подстановка в опубликованную reward formula |
| 08 Reasoning distillation | 63,0% относятся к s1K-1.1 с заново сгенерированными DeepSeek-R1 traces, а не ручной правке; доля — grader verdict, не пошаговая верификация | [s1 §2.2 / Appendix A / C.3](https://arxiv.org/html/2501.19393#A1) |
| 64a Connectors и fusion | H_v=Z_vW при Z_v:N_v×d_v и W:d_v×d_l | Совместимость внутренних размерностей и output:N_v×d_l |
| 64e Видео, аудио и omni-модели | Projector меняет ширину features; pooling/resampler сокращают число позиций | Согласовано с формами главы64a |
| 64e Видео, аудио и omni-модели | Temporal head выдаёт text; Depth Transformer — semantic audio и acoustic codes, отдельно от audio-only RQ-вводной | [Moshi §3.4.1 / §3.4.4, eq.2/6](https://arxiv.org/html/2410.00037v2#S3.SS4.SSS4) |
| 69 Coding, web и computer-use agents | ACI дополняет Bash; это интерфейс, не sandbox/permissions boundary | [SWE-agent Table4](https://arxiv.org/html/2405.15793v2#A1) |

Правки restore внутри search-loop главы69 и approval binding главы72 выполнены корневым агентом раньше и здесь не редактировались. Также не расширял SFT config distillation, не переписывал финальный coding-case и не исправлял другие оставшиеся findings.

## Тест parser: red → green

Сначала в главу добавлены assertions, затем из Markdown извлечены настоящий code block parser/reward и блок tests и исполнены Python. Исходная реализация завершилась AssertionError на `parse_final("FINAL: 4\\nтекст после результата") is None`. После минимальной правки те же 15 assertions прошли с exit code0. Повторный запуск после всех пяти изменений: 15 assertions, exit0, пустой stderr. Первый запуск test harness имел ошибку shell quoting; он не считался фазой red, после исправления harness получено именно ожидаемое AssertionError.

Для воспроизведения: исполнить подряд Python-блок с `def parse_final` и следующий блок assertions из главы06. Зависимости только стандартная библиотека. Это проверка учебного контракта, не security certification verifier и не запуск RL training.

Изменённые фрагменты перечитаны. Веб-проверка выполнена только для указанных трёх primary papers; арифметика и shapes проверены локально. Изображения не рендерились, построение сайта и built-links повторно не запускались — publisher одновременно диагностирует корневой агент. Семь anchors вопросов оставлены без изменений, поскольку их target HTML содержал пустые тела при существующих Markdown headings.

## SHA256 до и после моих точечных правок

Хеш «до» снят после уже выполненных корневых изменений, непосредственно перед моими правками. Это не замена read_sha256 исходного аудита.

### 00 Учебник/12 Post-training и Alignment/06 RLVR и verifiers.md

До: `71a2f70fa9171580d9ad6b05244f90e05f7edb34c9b05f25610281cf2afc13b2`.

После: `9b75b0cb4c95ee0ad38d296d235d2da590637cbeb8fe5847e2eb3eb45d41f094`.

### 00 Учебник/12 Post-training и Alignment/08 Reasoning distillation.md

До: `4a68babf763bdb362e0d7564e29c50b01d539875a8ea85dfd19272b271e91ae1`.

После: `e14cf6b8b724cf33f7d6ed112b45283134339b32cf0f120dee570d9388602cdc`.

### 00 Учебник/16 Multimodal Models/64a Connectors и fusion.md

До: `30bf5150dbfbd09e068a0585e4c9d3178330365f019fa9fc834dc3eac0796698`.

После: `006b7f02cc4432b430264463760f32f914e35b5fe4e5cab1c815bc4c78a7b6a8`.

### 00 Учебник/16 Multimodal Models/64e Видео, аудио и omni-модели.md

До: `1697139bc9417065561cb8fed218f193ef74b46561413bcbc73acd4d42964781`.

После: `2c263585334a2830c921c5fdbd13794c7b29669f20d09832a51301cbd74d376a`.

### 00 Учебник/17 Tools и Agents/69 Coding, web и computer-use agents.md

До: `408117f8963e700efae4cc6dcbf90cd9538732e7f77a7fc87c60137b4a054fcc`.

После: `acc7697bcbfaf84738f093a6ab3c165a9c18769979dded4a5c482d5f2145619d`.


