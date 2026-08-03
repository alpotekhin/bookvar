---
title: Mixture of Experts — routing, capacity и serving
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://arxiv.org/abs/1701.06538
  - https://arxiv.org/abs/2101.03961
  - https://arxiv.org/abs/2401.04088
  - https://arxiv.org/abs/2401.06066
  - https://arxiv.org/abs/2412.19437
---

# Mixture of Experts: условное вычисление как распределённая система

В обычном Transformer каждый токен проходит через одну и ту же полносвязную
сеть. В разреженном Mixture of Experts вместо неё хранится несколько таких
сетей-экспертов, а маршрутизатор выбирает для каждого токена лишь одну или две.
Модель получает значительно больше параметров, не выполняя все их при каждом
шаге. Экономия вычислений, однако, не означает бесплатного увеличения модели:
веса всех экспертов нужно разместить в памяти, представления токенов — доставить
к выбранным экспертам, а неравномерную нагрузку — удержать под контролем.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-moe/mixtral-smoe-layer.png]]

*Обобщённая схема sparse MoE layer: router выбирает несколько expert FFN и
смешивает их выходы. Авторы иллюстрации и статьи: Omar Sanseviero, Lewis Tunstall,
Philipp Schmid, Sourab Mangrulkar, Younes Belkada, Pedro Cuenca; страница
[Mixture of Experts Explained](https://huggingface.co/blog/moe), прямой файл
[01_moe_layer.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png).
Отдельная лицензия изображения в dataset card не указана.*

## Router и top-k

Для состояния токена $x_t\in\mathbb R^d$ маршрутизатор выдаёт $N$ логитов:

$$
z_t=x_tW_r,\qquad p_t=\operatorname{softmax}(z_t),\qquad
S_t=\operatorname{TopK}(p_t,k).
$$

Выход:

$$
y_t=\sum_{i\in S_t}\alpha_{t,i}E_i(x_t),
\qquad
\alpha_{t,i}=\frac{p_{t,i}}{\sum_{j\in S_t}p_{t,j}}.
$$

При top-1, как в Switch, пересылка и смешивание проще. Top-2, используемый в
GShard и Mixtral, добавляет второй маршрут и взвешенную комбинацию, но удваивает
число назначений. Невыбранные эксперты не получают градиент от основной функции
потерь для этого токена. Операция top-k дискретна; маршрутизатор обучается через
непрерывные веса выбранных путей и вспомогательные функции потерь.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-moe/sparse-routing.png]]

*Путь через экспертов может меняться от слоя к слою. Автор исходной схемы:
Maarten Grootendorst; повторно опубликована Hugging Face в статье авторов Aritra
Roy Gosthipaty, Pedro Cuenca, Merve Noyan, Ilyas Moutawwakil, Arthur Zucker,
Sergio Paniego и Pablo Montalvo. Страница [Mixture of Experts in Transformers](https://huggingface.co/blog/moe-transformers),
прямой файл [moe_routing.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe-transformers/moe_routing.png).
Отдельная лицензия изображения не указана.*

Маршрут выбирается заново в каждом MoE-слое. Экспертам не назначают профессии:
«кодер» или «математик» — эмпирическая гипотеза, которую нужно проверять по
маршрутам на независимом наборе.

## Общие и активные параметры

Если dense-часть содержит $P_D$, каждый эксперт $P_E$, экспертов $N$, top-$k$:

$$
P_{total}\approx P_D+NP_E,
\qquad
P_{active/token}\approx P_D+kP_E.
$$

Но число активных параметров не равно точному числу FLOP: attention и вложения
общие, а маршрутизация и обмен данными добавляют работу. Кроме того, веса всех
экспертов должны быть доступны в
совокупной памяти. Mixtral 8×7B — не восемь независимых 7B моделей.

## Почему коллапс маршрутизации усиливает сам себя

Случайно полезный эксперт получает больше токенов, быстрее улучшается и становится
ещё привлекательнее. Редкие эксперты недообучаются, а GPU популярного эксперта
задаёт step time. Балансировка нужна одновременно статистике обучения и системе.

Switch использует:

$$
f_i=\frac1M\sum_t\mathbf1[\operatorname{route}(t)=i],\qquad
P_i=\frac1M\sum_t p_{t,i},
$$

$$
\mathcal L_{balance}=\alpha N\sum_i f_iP_i.
$$

$f_i$ дискретна, $P_i$ дифференцируема. Минимум поощряет равномерные назначения,
но слишком большой $\alpha$ заставляет маршрутизатор балансировать нагрузку в ущерб основной задаче.
Router z-loss

$$
\mathcal L_z=\beta\left(\log\sum_i e^{z_i}\right)^2
$$

сдерживает величину логитов и помогает численной устойчивости. На практике
softmax маршрутизатора нередко считают в FP32, даже если эксперты работают в BF16.

DeepSeek-V3 применяет динамические bias для выбора: перегруженному эксперту bias
понижают, недогруженному повышают. Bias влияет на top-k, но не на вес смешивания;
небольшая sequence-level auxiliary loss сохраняется. Это точнее, чем фраза
«балансировка без loss вообще».

## Capacity и dropped tokens

Для $M$ токенов, $N$ экспертов и top-$k$ получается $Mk$ назначений. Средняя нагрузка:

$$
\bar C=\frac{Mk}{N},\qquad
C=\left\lceil c\frac{Mk}{N}\right\rceil,
$$

где $c$ — capacity factor. При $M=4096,N=8,k=2,c=1.25$ получаем $C=1280$
мест на эксперта. Всего выделяется 10240 мест для 8192 назначений: запас 25%.

Статические буферы удобны для XLA и ускорителей, но при переполнении нужно либо
отбросить ветвь эксперта и оставить остаточный путь, либо выбрать другого
эксперта, либо использовать динамическую маршрутизацию. Большая ёмкость снижает
число отброшенных назначений, но увеличивает заполнение пустыми значениями,
память и обмен данными. MoE без отбрасывания токенов всё равно страдает от
дисбаланса: ядра для групп разного размера и самый загруженный процесс влияют
на задержку всего слоя.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-37-40-audit/megablocks-expert-matmuls.png]]

*Почему dropless MoE требует специального kernel. Обычный batched matmul
выравнивает каждый expert batch до общей capacity; block-diagonal запись
устраняет часть лишней работы, но предполагает одинаковые блоки; MegaBlocks
упаковывает только занятые блоки и выполняет block-sparse matmul при разном числе
токенов у экспертов. Оригинальная Figure из
[MegaBlocks](https://arxiv.org/abs/2211.15841), опубликованная в учебной статье
[Hugging Face MoE Explained](https://huggingface.co/blog/moe); [прямой файл](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/11_expert_matmuls.png).*

## Expert parallelism: два all-to-all

Экспертов распределяют по процессам. После top-k представления токенов находятся
у процессов параллелизма данных, а нужные веса — у процессов параллелизма экспертов:

1. Router вычисляет назначения локальных токенов и группирует строки по
   `destination rank, expert id`.
2. Первый **all-to-all (dispatch)** отправляет каждую строку hidden state на rank,
   где лежит выбранный эксперт. При top-2 одна строка может уйти на два rank.
3. Получившиеся группы сортируют по expert id и подают в grouped GEMM. Размеры
   групп различаются, поэтому padding или block-sparse kernel непосредственно
   влияют на полезную загрузку ускорителя.
4. Второй **all-to-all (combine)** возвращает выходы на исходные data-parallel
   ranks. Там строки восстанавливают в исходном порядке и складывают с весами
   router $\alpha_{t,i}$.

Таким образом, пересылается не готовый ответ слоя, а представление токена до
эксперта и результат после него. Если на один процесс пришло существенно больше
строк, чем на остальные, вся коллективная операция ждёт этот процесс:
равномерное среднее число
назначений ещё не гарантирует короткий step time.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-37-40-audit/switch-parallelism.png]]

*Figure из Switch Transformer сопоставляет разбиение weights и данных при data,
model и expert parallelism. В колонке Expert and Data Parallelism разные цвета
в верхней строке — разные experts на cores, а нижняя строка показывает, что
каждый core начинает со своей части batch. Поэтому в MoE-слое токены необходимо
переслать к цвету выбранного эксперта. Источник: William Fedus, Barret Zoph,
Noam Shazeer, [Switch Transformers](https://arxiv.org/abs/2101.03961), повторная
публикация в [Hugging Face MoE Explained](https://huggingface.co/blog/moe),
[прямой файл](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/10_parallelism.png).*

Top-2 может переслать состояние токена дважды. Объём обмена на слой приблизительно
растёт с $Mk d$ элементов **в каждом направлении**; фактический объём также
зависит от dtype, padding и того, остаётся ли часть назначений локальной.
Expert parallelism комбинируют с tensor, pipeline и data parallelism; группы
collective нельзя выбирать независимо.
Удачное размещение удерживает обмен внутри быстрых доменов NVLink, а репликация
популярных экспертов иногда выгоднее строгого разделения.

## Стабильность обучения

Практический контрольный список:

- считать маршрутизатор в повышенной точности и отслеживать максимальный логит и z-loss;
- записывать долю назначений, массу вероятности, энтропию и переполнение **по слоям**;
- проверять баланс по глобальному batch, а не одному microbatch;
- давать каждому эксперту достаточно токенов для эффективного GEMM;
- аккуратно инициализировать маршрутизатор, использовать случайное возмущение там, где это
  подтверждено экспериментом;
- различать коллапс экспертов и их содержательную специализацию;
- отслеживать максимальную нагрузку процесса и время all-to-all вместе с основной функцией потерь.

Общие эксперты DeepSeekMoE выполняются для каждого токена и усваивают общие
закономерности; мелкозернистые маршрутизируемые эксперты дают больше комбинаций
при близком числе активных параметров. Это уменьшает дублирование, но общая
ветвь снова делает часть вычислений плотной.

## Serving: FLOPs экономятся, latency — не автоматически

Предзаполнение даёт много строк токенов и хорошо загружает сгруппированные GEMM.
При декодировании каждая последовательность приносит по одному новому токену:
эксперты получают маленькие и неравномерные группы, поэтому становятся заметны
затраты на запуск ядер и задержка сети. Непрерывное пакетирование улучшает
заполнение, но маршруты заранее неизвестны.

Веса всех экспертов должны находиться в памяти GPU или CPU либо подгружаться по
требованию. Выгрузка редких экспертов уменьшает VRAM, но промах в кэше резко
увеличивает хвостовую задержку. Квантование снижает расход памяти, однако
маршрутизация и коллективные операции остаются. Для малой задержки на одном
устройстве плотная модель часто проще; преимущество MoE заметнее при высокой
пропускной способности на крупном кластере.

| Метрика | Что обязательно измерять |
|---|---|
| качество | loss/perplexity при том же active compute |
| баланс | max/mean load, entropy, dropped-token rate |
| обучение | tokens/s, all-to-all доля, worst-rank time |
| serving | TTFT, inter-token latency, p95/p99, memory всех weights |

> [!danger] Разреженное вычисление не означает разреженное хранение
> Неактивный expert не выполняет matmul для данного токена, но его weights должны
> быть быстро доступны, а его rank участвует в синхронизации слоя.

## После главы нужно уметь

- вывести top-k output, balance loss и capacity;
- отличить total, active parameters и реальный FLOP/communication budget;
- по готовой схеме из первоисточника проследить оба all-to-all и объяснить, какие данные передаются;
- объяснить router collapse, z-loss и loss-free bias balancing;
- сравнить training prefill-like throughput с decode serving.

## Источники

- [Stanford CS336](https://cs336.stanford.edu/) — conditional computation и
  distributed cost; [CS25](https://web.stanford.edu/class/cs25/) — доклады о
  масштабировании Transformer.
- [Hugging Face: Mixture of Experts Explained](https://huggingface.co/blog/moe)
  — наиболее цельное объяснение top-k, capacity, z-loss, parallelism и serving.
- [D2L Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)
  задаёт dense FFN baseline; MoE видно как условную замену именно этой ветви.
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/llm-bootcamp/) и
  [Chip Huyen](https://huyenchip.com/2023/04/11/llm-engineering.html) — production
  evaluation, batching, memory и latency.
- Papers: [Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538),
  [Switch](https://arxiv.org/abs/2101.03961), [Mixtral](https://arxiv.org/abs/2401.04088),
  [DeepSeekMoE](https://arxiv.org/abs/2401.06066),
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437).
