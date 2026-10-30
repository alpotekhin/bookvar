---
title: Аудит 44c — tensor, sequence и context parallelism
type: editorial
status: editorial
last_verified: 2026-10-23
---

# Проверенная редакция

Каноническая глава:
`00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md`.

- До правок: 612 строк, SHA-256 `2a7cd5f3cdc437d2d176593ceac6266f81ce05da9c01296d3bea410af7e667b2`.
- После правок: 1012 строк, SHA-256 `19a681a8a334905e3c2b65b66757b4dcff58086e7c2e4c48a3717882767da30a`.
- [Исходная редакция](snapshots/2026-10-23-tensor-sequence-before.md) сохранена до редактирования; совпадение проверено через cmp.
- Route: `/textbook/pretraining/tensor-sequence-parallelism/`.

Исходная глава прочитана целиком. Итоговый текст перечитан по порядку;
добавленные после этого уточнения сверены отдельно. Сохранены пять
source_unit_id, прежние явные anchors, ссылки на соседние главы и подробные
матричные примеры. Новые маркеры не добавлялись для имитации полного
переноса курса. Это новый проход одной главы, а не завершение всего учебника.

# Сопоставление с источниками

Для прежних объяснений также действует [предыдущий разбор](2026-10-13-tensor-sequence.md).
В текущем проходе заново прочитаны конкретные разделы:

- Megatron-LM, [arXiv:1909.08053v4](https://arxiv.org/abs/1909.08053v4),
  §3: парные разрезы MLP, головы внимания, embedding и распределённая
  cross-entropy без сбора полного массива logits.
- Stanford CS336, lecture 8, физические страницы 39–54, закреплённая
  ревизия `8b59b50730766695c2ffedd1a79c50cd09b9eb91`:
  разрез по ширине, границы SP/TP и память активаций.
- [CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf),
  §8.4–8.5, включая tp_calcs и fsdp_tp_calcs. Предпосылки C/W сверены
  с §8.2; разбор использует три проекции SwiGLU, а не два умножения ReLU MLP.
- Korthikanti et al., [arXiv:2205.05198v1](https://arxiv.org/abs/2205.05198v1),
  §4.1–4.2.2 и §5: разложение 34sbh, квадратичные сохранения внимания,
  replicated/partitioned activations и selective recomputation.
  Соответствующие рисунки self-attention, MLP и TP+SP лично осмотрены;
  для сайта использованы отдельно атрибутированные изображения Playbook.
- Jacobs et al., [DeepSpeed Ulysses v2](https://arxiv.org/html/2309.14509v2),
  §3.1–3.4: две перестановки all-to-all, полный контекст внутри группы
  голов и условия на число голов. Figure 2 импортирована целиком.
- Liu et al., [Ring Attention v4](https://arxiv.org/html/2310.01889v4),
  §3 и Algorithm 1: циркуляция K/V, online softmax, оценка перекрытия
  вычислений передачей. Проверка исполняет c блоков и c−1 пересылок,
  включая c=1; текст не копирует границы цикла псевдокода без проверки.
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook),
  тематические разделы TP/SP/CP из commit
  `c93890dd4b22356ee18ed9b863e6f2d0909729bc`: схема голов, области
  разрезов, Ring и zigzag. Ring P2P не назван Ulysses all-to-all.
- Официальные PyTorch 2.14 API ColwiseParallel, RowwiseParallel,
  SequenceParallel и loss_parallel; docstring и ограничения loss.py
  отдельно сверены с [v2.14.1](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/distributed/tensor/parallel/loss.py).
  Это проверка контракта документации, не запуск GPU DTensor.
- Оригинальный EDLS practice_part2: первые 180 строк, включая
  ComputeWithAllReduce. Его пересчёт в backward не перенесён как готовая
  универсальная реализация без оговорок о входах и RNG; архив не изменялся.

# Дополнения и исправления

1. Обратный проход SwiGLU доведён до градиентов всех трёх матриц;
   объяснено, почему разные TP-shards весов не надо суммировать.
   В attention сначала складываются локальные ветви dQ/dK/dV, затем
   выполняется один общий обмен dX. Число редукций указано для обычного
   блока, без молчаливого включения embedding/loss.
2. Добавлены vocabulary-parallel embedding и cross-entropy: владелец
   строки, глобальный MAX, сумма экспонент, target logit, локальный
   градиент logits и сумма dX. Пример [0,1]/[2,3], target=3 разобран
   до loss 0.440190 и четырёх компонент градиента. Включены tied weights,
   padded vocabulary, valid-token normalization и ограничения loss_parallel.
3. Оценка времени TP выведена для трёх проекций. При заданных C/W
   получаются границы 25/49, а TP=16/32 сравниваются численно.
   Граница относится к указанной ring-модели обмена, не обещает
   полного перекрытия зависимых операций и не является GPU benchmark.
4. Память активаций выведена из MLP 19sbh, attention 11sbh+5as²b
   и двух нормализаций 4sbh. Объяснены 10sbh реплик, отличие TP от SP
   и четыре результата 3648/736/456/136 MiB. Модель исходной статьи
   ограничена classic MHA/MLP; её коэффициенты не выданы за универсальный
   калькулятор SwiGLU/FlashAttention. sbh — число элементов, не байт.
5. Под рисунком Ulysses расшифрованы исходные N/d/P и частный случай
   четырёх процессов/голов. Математическая N×N матрица не объявлена
   обязательным материализованным буфером при FlashAttention.
6. Ring Attention дополнен пошаговым online-softmax примером с итогом
   7.5, расчётом 112/448 MiB для Ulysses/Ring и условием на размер
   MHA-блока u ≥ b_e C/(2W). При гипотетических C/W порог равен 1792
   токенам. Объёмы не превращены в заявление о четырёхкратном speedup;
   causal skips, latency и GQA требуют отдельного учёта.
7. Объяснена агрегация градиентов параметров по CP-копиям: передача
   dK/dV её не заменяет. Пример 0.5+1.5=2 показывает ошибку повторного
   усреднения. Совмещённые группы ZeRO не должны суммировать одно дважды.
8. Для FSDP×TP 2×2 перечислены stored/gathered shapes и разные
   назначения двух групп. All-gather FSDP восстанавливает локальную
   половину TP, не полную матрицу исходной модели.
9. В видимом Python-примере теперь есть backward, а не только forward.
   Восемь широких формул разбиты на связанные строки; таблица SP
   перекомпонована без удаления операций и чисел.

# Оригинальные изображения

В главе шесть изображений. Два новых файла:

- `tensor-sequence-2026-10-23/ulysses-design.png`, 1408×649,
  полный Figure 2, CC BY 4.0. SHA-256
  `0c41d8e6105bfda47d57280eb992d450a5ed9dcbdcd820d3a9663b8f9346903d`.
- `tensor-sequence-2026-10-23/hf-ring-attention.gif`, 3000×1688,
  16 кадров по 750 ms, полный цикл 12 s, бесконечный повтор.
  SHA-256 `ac30e82373020c1248fed596c85a86ccc2934c1920e8b9b82b9a8d42d0a055a1`.

Оба объекта скопированы побайтно без crop, перерисовки, изменения
английских подписей или перекодирования. Сохранены статья Ulysses
и version-specific декларация CC BY 4.0; для HF — HTML и README
закреплённого commit с Apache-2.0. Полный текст Apache уже опубликован
в `/licenses/ultrascale-Apache-2.0.txt`. Лицензия Playbook не приписана
иллюстрациям из Ring Attention paper.

Происхождение записано в соседних assets.json/NOTICE и центральном
`05 Источники/asset-registry.yml`. Исходные файлы и license evidence:
`05 Источники/Originals/tensor-sequence-2026-10-23/`.

Все четыре прежних изображения заново получены с закреплённых upstream
и побайтно совпали: Harvard tensor-parallel-split.svg, HF tp_full_diagram.png,
tp_sp_diagram.png и cp_zigzagmask.svg. Harvard pin:
`45ecc8d82fcae70c149cdce550d3b3d3411df913`; HF pin указан выше.
Все шесть лично осмотрены целиком в desktop-просмотрщике.
Для GIF также проверено видимое движение K/V при неподвижных Q.

# Исполнение и сборка

`publishing/tools/check_tensor_sequence_examples.py` исполняет три
actual Python-блока Markdown. Итоговый JSON содержит тот же SHA-256,
что и глава выше. NumPy 2.3.5, один CPU-процесс, 14 PASS-групп:

- Парные MLP: TP=1/2/4, forward/backward, finite differences по x/A/B.
- SwiGLU: обе ветви dX, градиенты и соответствующие колонки.
- SP: сопряжённость all-gather/reduce-scatter, расположение строк,
  RMSNorm и x/gamma finite differences, сумма вместо среднего.
- MHA/GQA: разрез голов и правильная сумма градиентов реплицированных KV.
- Ulysses: перестановки sequence↔heads, forward и положение Q/K/V gradients.
- Ring: causal/noncausal, c=1/2/4, пустые блоки, большие logits, gradients.
- Vocabulary CE: каждый target, TP=1/2/4, сдвиги logits 0/±1000,
  неодинаковые shards, недопустимые target и embedding с повторными строками.
- Напечатанные времена TP=16/32, границы 25/49, 112/448 MiB,
  порог 1792 токена, CP normalization и все четыре строки памяти.
- FSDP×TP 2×2: shapes, восстановленный forward и все градиенты весов.
- Прежние anchors/IDs и шесть оригинальных figure paths.

Проверки проекта:

- 394 unit-теста: PASS, 39 файлов, 18:49:25. После этого менялись только
  уточнения главы и её численного checker; asset/test-код не менялся.
- Финальная сборка: PASS, 1961 страница, завершена 18:56:52.
- По этой сборке: 92 output-теста PASS; 0 broken internal routes,
  fragments или files; source-coverage check PASS.
- Логи находятся в `/var/tmp/bookvar-tp-sp-2026-10-23.rHPthd/`:
  checker-final.log, unit-ready.log, build-delivered.log,
  output-delivered.log, links-delivered.log, coverage-delivered.log.

На этой macOS-машине не исполнялись distributed backend, CUDA, NCCL,
DTensor loss_parallel и многопроцессный TP/CP. Численная модель
не доказывает работу сети, overlap, GPU memory peak или ускорение.

# Проверка отображения

После финальной сборки страница явно перезагружена в IAB.
При 360×800:

- 315 выражений KaTeX, 0 katex-error, 0 переполняющихся display-блоков.
- Все четыре таблицы: clientWidth=scrollWidth=313 px.
- Все шесть основных изображений загружены; их границы внутри x=16..329.
- Ulysses проверен на телефоне при 2.5× с перемещением по рисунку;
  правые обозначения не отрезаны исходным файлом.

Скриншоты в `docs/audit/screenshots/`:

- `2026-10-23-tp-ulysses-desktop.png` — полный рисунок с полной подписью,
  переснят после окончательной сборки.
- `2026-10-23-tp-ulysses-mobile.png` и `…-mobile-right.png` — увеличение/перемещение.
- `2026-10-23-tp-ring-frame-a.png` и `…-frame-b.png` — разные состояния GIF.
- `2026-10-23-tp-memory-mobile.png` — четыре строки памяти на телефоне.

Временный viewport сброшен. CSS и image-viewer не менялись.
EN, raw-архивы и чужие изменения не редактировались; commit/push/deploy
не выполнялись. Далее — 44d Pipeline parallelism. Полный проход книги,
общая интеграция курсов и прежние незакрытые проверки остаются открытыми.
