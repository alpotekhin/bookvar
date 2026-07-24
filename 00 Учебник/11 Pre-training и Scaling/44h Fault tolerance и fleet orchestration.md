---
title: Fault tolerance и fleet orchestration
type: textbook-chapter
status: canonical
last_updated: 2026-07-24
---

# 44h. Fault tolerance и fleet orchestration

При синхронном обучении отказ одного rank останавливает world. Если независимый MTBF устройства равен $M$, то грубый MTBF job на $N$ устройствах — $M/N$. Для $M=5$ лет и $N=1024$ это около 42.8 часа: многонедельный запуск обязан проектировать recovery как штатный путь.

## Домены отказа и silent corruption

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/failure-domains.svg]]

*Источник: Harvard Edge ML Systems Book, [Fault Tolerance, figure `fig-failure-domains`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd), CC BY-NC-SA 4.0.*

GPU, node, top-of-rack switch, power domain, storage rack и control plane — разные failure domains. Реплика checkpoint в том же rack не защищает от rack failure. Размещение должно разносить recovery copies, но training ranks — собирать рядом для высокой bandwidth: надёжность и locality тянут topology в разные стороны.

Fail-stop обнаруживается timeout/ECC; silent data corruption (SDC) продолжает выдавать правдоподобные числа. Защита включает ECC, checksums при передаче и хранении, периодические deterministic canaries, проверку finite/range, loss/gradient anomaly detection и сравнение replicated computations для критических участков. Checkpoint после SDC бесполезен, если неизвестно, когда corruption началась; поэтому сохраняют несколько поколений и валидируют до продвижения «good» marker.

## Интервал Young–Daly

Если checkpoint занимает $C$, а MTBF job равен $M$, ожидаемый overhead при интервале полезного compute $T$:

$$W(T)\approx \frac{C}{T}+\frac{T}{2M}.$$

Минимум Young:

$$T^*=\sqrt{2CM}.$$

Daly учитывает checkpoint duration и даёт практическую поправку порядка $T_{\mathrm{Daly}}\approx\sqrt{2CM}-C$ при обычных предпосылках.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/young-daly-optimization.svg]]

*Источник: Harvard Edge ML Systems Book, [Fault Tolerance, section `sec-fault-tolerance-checkpoint-optimization`, figure `fig-young-daly-optimization`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd), CC BY-NC-SA 4.0.*

Пример: $C=120$ s, $M=12$ h $=43200$ s. $T^*=\sqrt{10\,368\,000}=3219$ s ≈53.6 min; Daly ≈51.6 min. Checkpoint каждые 10 min платит 20% только записью; каждые 3 h теряет в среднем 1.5 h при failure. Формула предполагает стационарные независимые отказы и синхронный checkpoint; реальные preemption windows и correlated failures требуют simulation/trace.

Recovery time включает detection $D$, scheduler/rendezvous $Q$, restore $R$ и replay примерно $T/2$. Availability полезной работы оценивают с этими членами, не только с $C/T$.

## Elastic recovery

Elastic restart может заменить rank или изменить world size. Но изменение DP degree меняет global batch, scheduler semantics, shard placement и порядок data sampler. Состояние должно быть topology-independent; rendezvous формирует новый world, distributed checkpoint reshard’ится, а data position не повторяет и не пропускает примеры. Для TP/PP изменение степени часто невозможно без смены layout, поэтому elasticity обычно ограничивают DP-осью.

## Fleet orchestration

Gang scheduling запускает job только если одновременно доступны все требуемые ranks. Частичный старт 60 из 64 GPU расходует ресурсы, но collective не начнётся. Backfilling заполняет окна короткими jobs, не задерживая зарезервированный крупный запуск; preemption полезна только при checkpoint-aware grace period.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/ml-systems/harvard/distributed/topology-placement.svg]]

*Источник: Harvard Edge ML Systems Book, [Fleet Orchestration, figure `fig-topology-placement`](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd), CC BY-NC-SA 4.0.*

Topology-aware placement минимизирует дорогие edges: TP ranks в одном NVLink node, EP/DP groups по полным rails, PP stages рядом и в порядке пути. Рассыпанные свободные GPU не эквивалентны компактному блоку.

Slurm естественно даёт atomic allocation, topology constraints, fair-share и backfill для batch HPC. Kubernetes по умолчанию планирует pods независимо; для distributed jobs нужны admission/gang abstractions (Volcano/Coscheduling/Kueue), Training Operator, RDMA device plugins, host networking и storage QoS. Выбор не религиозный: dedicated training fleet часто проще на Slurm, смешанный training/serving control plane — на Kubernetes; возможен гибрид.

Autoscaling training ограничен gang size, временем provisioning и checkpoint/reconfiguration. Масштабировать по мгновенной загрузке GPU бессмысленно: полезный сигнал — queue wait, ожидаемая длительность job, compact topology capacity и цена изменения world. Inference autoscaling реагирует быстрее; training fleet планируют минутами/часами.

### Расчёт решения

Job на 256 GPU ждёт compact block 2 h или может стартовать сейчас на фрагментированной topology, где step на 18% медленнее. Для ожидаемого 20-h baseline фрагментированный запуск займёт 23.6 h и завершится через 23.6 h; ожидание + compact — 22 h. Ждать выгоднее по completion time, хотя immediate utilization выглядит лучше.

## Runbook проверки восстановления

1. Сохранить checkpoint и durable manifest.
2. Убить rank, node и отдельно сетевой path в контролируемых тестах.
3. Измерить detection, rescheduling, restore и replay.
4. Сравнить IDs следующего batch, loss, gradients, optimizer/scaler и RNG с uninterrupted control.
5. Повторить restore при допустимом изменении DP topology.
6. Провести SDC drill: checksum/canary должен остановить продвижение corrupted checkpoint.

## Источники

- Harvard Edge ML Systems Book, [Fault Tolerance](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd), sections `sec-fault-tolerance-failure-models`, `sec-fault-tolerance-checkpoint-optimization`, `sec-fault-tolerance-silent-data-corruption`.
- Harvard Edge ML Systems Book, [Fleet Orchestration](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/fleet_orchestration/fleet_orchestration.qmd), sections on gang scheduling, topology placement, Slurm and Kubernetes.
- Daly, [A Higher Order Estimate of the Optimum Checkpoint Interval](https://doi.org/10.1016/j.future.2004.11.016), 2006.

← [[44g Network, storage и distributed checkpoints]] · Далее: [[01 SFT и instruction data]]
