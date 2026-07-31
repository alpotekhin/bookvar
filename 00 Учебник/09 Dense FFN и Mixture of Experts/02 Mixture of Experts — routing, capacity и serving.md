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

Для token state $x_t\in\mathbb R^d$ router выдаёт $N$ logits:

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

Top-1 (Switch) проще dispatch и смешивание. Top-2 (GShard, Mixtral) даёт второй
маршрут и взвешенную комбинацию, но удваивает assignments. Не выбранные эксперты
не получают gradient от LM loss этого токена. Top-k дискретен; router обучается
через непрерывные веса выбранных путей и вспомогательные losses.

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

Но active parameters — не точные FLOPs: attention и embeddings общие, router и
communication добавляют работу, а веса всех экспертов должны быть доступны в
совокупной памяти. Mixtral 8×7B — не восемь независимых 7B моделей.

## Почему routing collapse сам себя усиливает

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
но слишком большой $\alpha$ заставляет router балансировать вопреки LM quality.
Router z-loss

$$
\mathcal L_z=\beta\left(\log\sum_i e^{z_i}\right)^2
$$

сдерживает величину logits и помогает численной устойчивости. На практике router
softmax нередко считают в FP32, даже если experts работают в BF16.

DeepSeek-V3 применяет динамические bias для выбора: перегруженному эксперту bias
понижают, недогруженному повышают. Bias влияет на top-k, но не на вес смешивания;
небольшая sequence-level auxiliary loss сохраняется. Это точнее, чем фраза
«балансировка без loss вообще».

## Capacity и dropped tokens

Для $M$ токенов, $N$ experts и top-$k$ всего $Mk$ назначений. Средняя нагрузка:

$$
\bar C=\frac{Mk}{N},\qquad
C=\left\lceil c\frac{Mk}{N}\right\rceil,
$$

где $c$ — capacity factor. При $M=4096,N=8,k=2,c=1.25$ получаем $C=1280$
мест на эксперта. Всего выделяется 10240 slots для 8192 assignments: запас 25%.

Статические buffers удобны XLA/accelerators, но overflow требует решения:
отбросить expert branch и оставить residual, отправить к следующему эксперту или
использовать dynamic routing. Большая capacity снижает drops, но увеличивает
padding, memory и communication. Современный dropless MoE не устраняет дисбаланс:
ragged kernels и самый загруженный rank всё равно влияют на latency.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/source-first-37-40-audit/megablocks-expert-matmuls.png]]

*Почему dropless MoE требует специального kernel. Обычный batched matmul
выравнивает каждый expert batch до общей capacity; block-diagonal запись
устраняет часть лишней работы, но предполагает одинаковые блоки; MegaBlocks
упаковывает только занятые блоки и выполняет block-sparse matmul при разном числе
токенов у экспертов. Оригинальная Figure из
[MegaBlocks](https://arxiv.org/abs/2211.15841), опубликованная в учебной статье
[Hugging Face MoE Explained](https://huggingface.co/blog/moe); [прямой файл](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/11_expert_matmuls.png).*

## Expert parallelism: два all-to-all

Экспертов раскладывают по ranks. После top-k токены находятся на data-parallel
ranks, а нужные weights — на expert ranks:

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

Таким образом, пересылается не готовый ответ слоя, а token representation до
эксперта и результат после него. Если на rank пришло существенно больше строк,
чем на остальные, весь collective ждёт этот rank: равномерное среднее число
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

Top-2 может переслать token state дважды. Communication на слой приблизительно
растёт с $Mk d$ элементов **в каждом направлении**; фактический объём также
зависит от dtype, padding и того, остаётся ли часть назначений локальной.
Expert parallelism комбинируют с tensor, pipeline и data parallelism; группы
collective нельзя выбирать независимо.
Хороший layout держит communication внутри быстрых NVLink domains, а replication
популярных experts иногда выгоднее строгого sharding.

## Стабильность обучения

Практический checklist:

- считать router в повышенной точности и мониторить max logits/z-loss;
- логировать долю assignments, probability mass, entropy и overflow **по слоям**;
- проверять баланс по глобальному batch, а не одному microbatch;
- давать каждому expert достаточно tokens для эффективного GEMM;
- аккуратно инициализировать router, использовать jitter/noise там, где это
  подтверждено экспериментом;
- различать expert collapse и semantic specialization;
- отслеживать worst-rank load и all-to-all time вместе с LM loss.

Shared experts DeepSeekMoE выполняются для каждого токена и учат общие знания;
мелкозернистые routed experts дают больше комбинаций при близком active budget.
Это уменьшает дублирование, но shared branch снова делает часть compute dense.

## Serving: FLOPs экономятся, latency — не автоматически

Prefill даёт много token rows и хорошо загружает grouped GEMM. Decode приносит по
одному новому токену на sequence: experts получают маленькие и неравномерные
группы, kernel launch и network latency становятся заметны. Continuous batching
улучшает заполнение, но routes заранее неизвестны.

Weights всех experts должны находиться в GPU/CPU memory или подгружаться. Offload
редких experts может уменьшить VRAM, но tail latency взрывается при cache miss.
Quantization снижает память, однако router и collective остаются. Для
single-device low-latency dense model часто проще; MoE особенно силён для
throughput на крупном кластере.

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

## Сопоставление учебных объяснений и источники

| Подраздел | Выбранная основа объяснения | Почему она сильнее сравненных альтернатив | Готовый визуал |
|---|---|---|---|
| router/top-k | [Hugging Face MoE Explained](https://huggingface.co/blog/moe) | пошагово выводит noisy top-k; Switch paper сразу переходит к top-1 system design | HF MoE-layer и routing figures выше |
| total/active budget | [Mixtral](https://arxiv.org/abs/2401.04088) | даёт проверяемый 8×7B case; общие FSDL/Chip Huyen материалы не раскладывают shared weights | готовая HF MoE-layer figure |
| load balance | [Switch Transformer](https://arxiv.org/abs/2101.03961) | определяет $f_i$, $P_i$ и auxiliary loss; visual guide даёт интуицию без полного вывода | формула из Switch, HF routing visual |
| z-loss/stability | [ST-MoE](https://arxiv.org/abs/2202.08906) | связывает magnitude logits, precision и stability экспериментально; HF пересказывает результат | paper formula и monitoring checklist |
| capacity/overflow | [GShard](https://arxiv.org/abs/2006.16668) + [HF MoE Explained](https://huggingface.co/blog/moe) | GShard задаёт статические buffers, HF яснее объясняет trade-off capacity factor | численный расчёт 4096 tokens |
| expert parallelism | [GShard](https://arxiv.org/abs/2006.16668) + [Switch](https://arxiv.org/abs/2101.03961) | GShard описывает dispatch/collectives как sharding problem; Switch наглядно сопоставляет раскладку weights и data; CS336 даёт общий distributed baseline | оригинальная Figure Switch выше |
| loss-free balancing | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | первично различает selection bias и mixing weights; блоги часто ошибочно говорят «без losses» | алгоритм bias update адаптирован словами |
| serving | [HF MoE Explained](https://huggingface.co/blog/moe) + [MegaBlocks](https://arxiv.org/abs/2211.15841) | соединяет VRAM/offload с dropless block-sparse kernels; FSDL даёт общий serving framework | оригинальная Figure MegaBlocks + таблица метрик |

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
