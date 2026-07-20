---
title: Mamba, RWKV, RetNet и гибридные архитектуры
type: textbook-chapter
status: canonical
last_updated: 2026-07-20
primary_sources:
  - https://arxiv.org/abs/2312.00752
  - https://arxiv.org/abs/2305.13048
  - https://arxiv.org/abs/2307.08621
  - https://arxiv.org/abs/2403.19887
---

# Mamba, RWKV, RetNet и гибриды: последовательность без полного attention

Transformer обучает все позиции параллельно, но dense attention строит
$T\times T$ взаимодействий и хранит растущий KV-cache. RNN декодирует с
фиксированным состоянием, но наивная recurrence последовательна и плохо обучается
на длинных зависимостях. Mamba, RWKV и RetNet ищут разные точки между этими
полюсами: **параллельное обучение, recurrent inference и линейный рост по длине**.

## State space model: динамика скрытого состояния

Непрерывная линейная система:

$$
\dot h(t)=Ah(t)+Bx(t),\qquad y(t)=Ch(t)+Dx(t).
$$

После дискретизации:

$$
h_t=\bar Ah_{t-1}+\bar Bx_t,\qquad y_t=Ch_t+Dx_t.
$$

Её можно вычислять recurrently за $O(1)$ state на шаг или раскрыть как свёртку:

$$
y_t=\sum_{i=0}^{t}C\bar A^{t-i}\bar Bx_i+Dx_t.
$$

Так возникает duality: recurrence удобна на decode, convolution/parallel scan —
на training. В отличие от attention, прошлое хранится не набором всех K/V, а
сжатым состоянием. Это экономия и одновременно bottleneck.

## Mamba: selective state space

У линейной time-invariant SSM параметры перехода одинаковы для любого токена;
трудно избирательно запомнить важное и забыть шум. Mamba делает $B_t,C_t,\Delta_t$
зависящими от входа:

$$
h_t=\bar A(\Delta_t)h_{t-1}+\bar B(x_t,\Delta_t)x_t,
\qquad y_t=C(x_t)h_t.
$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-alternatives/mamba-selective-ssm.png]]

*Selective SSM: input-dependent параметры позволяют выборочно переносить или
сбрасывать информацию. Авторы: Albert Gu, Tri Dao, страница [Mamba](https://arxiv.org/abs/2312.00752),
прямой [PDF](https://arxiv.org/pdf/2312.00752), Figure 1. Официальный код:
[state-spaces/mamba](https://github.com/state-spaces/mamba), лицензия Apache 2.0;
отдельная лицензия figure в paper не указана.*

Input dependence разрушает простую глобальную convolution, поэтому авторы
разработали hardware-aware selective scan: ассоциативно группируют переходы,
держат состояние в быстрой памяти и recompute промежуточные значения вместо
записи огромных tensors в HBM. «Линейная сложность» сама по себе не гарантирует
скорость — важны fusion, scan kernel и длина.

На autoregressive inference Mamba хранит небольшой state каждого слоя и локальный
convolution buffer; cache не растёт с $T$. Но состояние — lossy summary: прямой
адресации к токену 50 тысяч шагов назад нет.

## RWKV: attention-подобное смешивание как recurrence

RWKV чередует time-mixing и channel-mixing. Weighted key-value accumulation
можно обновлять recurrently, поэтому decode имеет фиксированное state, а обучение
вычисляет последовательность параллельными kernels. Экспоненциальные time-decay
задают, как быстро прошлое теряет вес; receptance играет роль gate.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-alternatives/rwkv-architecture.png]]

*Одна и та же RWKV-сеть на трёх последовательных шагах: time-mix передаёт
состояние вперёд, а token shift связывает соседние входы. Авторы: Bo Peng et al., страница
[RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048),
прямой [PDF](https://arxiv.org/pdf/2305.13048), Figure 3. Официальный
repo [BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM), лицензия Apache 2.0;
отдельная лицензия figure не указана.*

RWKV не строит softmax по явному списку прошлых позиций. Это даёт стабильную
memory footprint, но усложняет интерпретацию и точное извлечение произвольного
старого фрагмента. Состояние нужно переносить между запросами осторожно: оно
зависит от всей истории и является частью cache semantics.

## RetNet: три эквивалентных формы retention

RetNet вводит retention — decay-weighted взаимодействие query/key/value. Один и
тот же оператор заявлен в трёх формах: parallel для training, recurrent для
token-by-token inference и chunkwise recurrent для длинных chunks.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-alternatives/retnet-dual-form.png]]

*Parallel и recurrent representations одного retention-оператора. Chunkwise
форма сочетает эти два режима и разбирается в тексте ниже. Авторы:
Yutao Sun et al., страница [Retentive Network](https://arxiv.org/abs/2307.08621),
прямой [PDF](https://arxiv.org/pdf/2307.08621), Figure 3. Официальный repo
[microsoft/unilm](https://github.com/microsoft/unilm/tree/master/retnet), лицензия
MIT для кода; отдельная лицензия paper figure не указана.*

Упрощённо recurrent state — матрица накопленных outer products:

$$
S_t=\gamma S_{t-1}+k_t^\top v_t,\qquad y_t=q_tS_t.
$$

Разные heads используют разные decay $\gamma$, создавая несколько временных
масштабов. Chunkwise форма обрабатывает взаимодействия внутри chunk параллельно,
а между chunks передаёт state. Это полезный компромисс throughput/latency.

## Parallel scan: почему recurrence можно обучать параллельно

Переход можно представить парой $(A_t,b_t)$, задающей affine map
$h\mapsto A_th+b_t$. Композиция ассоциативна:

$$
(A_2,b_2)\circ(A_1,b_1)=(A_2A_1,A_2b_1+b_2).
$$

Ассоциативный prefix scan строит все prefixes деревом за логарифмическую depth,
хотя общая работа остаётся линейной. Это и есть мост между recurrent semantics
и parallel training. Ограничение: не всякая нелинейная RNN допускает удобный
ассоциативный оператор.

## Почему гибриды возвращают attention

State-space/recurrent layers хороши для дешёвого переноса локального и
накопленного состояния; attention хороша для content-addressed lookup — прямого
сравнения query со всеми сохранёнными keys. Jamba чередует Mamba и attention и
добавляет MoE: большинство слоёв линейны по длине, редкие attention layers дают
точный глобальный маршрут, MoE увеличивает capacity.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/modern-37-40-alternatives/jamba-hybrid.png]]

*Jamba сочетает Mamba, attention и MoE blocks. Авторы: Opher Lieber et al.,
страница [Jamba](https://arxiv.org/abs/2403.19887), прямой [PDF](https://arxiv.org/pdf/2403.19887),
Figure 1. Официальный repo [ai21labs/Jamba](https://github.com/ai21labs/Jamba),
лицензия Apache 2.0 для кода; отдельная лицензия figure не указана.*

Гибрид требует двух cache: фиксированное recurrent state для Mamba и растущий
KV-cache только для attention layers. Если attention стоит раз в $r$ слоёв,
KV-cache приблизительно уменьшается в $r$ раз относительно fully-attentional
модели той же глубины, но не исчезает.

## Сравнение без маркетинговых сокращений

| Семейство | Training | Decode state | Сильная сторона | Риск |
|---|---|---|---|---|
| Transformer | parallel attention | растущий KV | точная адресация прошлого | $T^2$, cache |
| Mamba | selective scan | фиксированный | content-dependent memory update | сжатие истории |
| RWKV | parallel WKV kernel | фиксированный | простая recurrent generation | decay/recall |
| RetNet | parallel/chunkwise | фиксированный | явная dual form | экспоненциальное забывание |
| Hybrid | смешанный | state + частичный KV | баланс recall/throughput | сложнее kernels и serving |

Сравнение качества должно удерживать параметры, training tokens, tokenizer и
compute. Benchmark на коротких sequences не показывает long-memory; synthetic
recall не заменяет language modeling. Для production измеряют prefill throughput,
decode latency, batch scaling, state/cache bytes на sequence и качество при
разной позиции доказательства.

## Train–inference mismatch и состояние

Parallel и recurrent формы должны быть численно эквивалентны в пределах
погрешности. Полезный тест official implementation: прогнать одну
последовательность целиком и по токену, сравнить logits/state. Chunk boundaries,
precision и reset state могут нарушить эквивалентность. State нельзя переносить
между независимыми пользователями; при prefix caching он должен версионироваться
вместе с model weights и dtype.

> [!warning] O(T) не означает «лучше Transformer»
> Линейный оператор может уступить по качеству точного recall, а короткий prompt
> не окупает сложный scan kernel. Победа определяется Pareto-поверхностью
> качество × latency × memory × throughput.

## После главы нужно уметь

- вывести recurrent и convolution/scan view SSM;
- объяснить selectivity Mamba и fixed-state decode;
- различить механизм RWKV и retention RetNet;
- показать ассоциативность affine scan;
- обосновать, зачем гибрид сохраняет редкие attention layers.

## Сопоставление учебных объяснений и источники

| Подраздел | Выбранная основа объяснения | Почему она сильнее сравненных альтернатив | Готовый визуал |
|---|---|---|---|
| SSM three views | [Mamba paper](https://arxiv.org/abs/2312.00752), §2 | в одном выводе связывает continuous, recurrent и convolution forms; D2L подробно объясняет RNN, но не SSM duality | Mamba Figure 1 |
| selective scan | [Mamba paper](https://arxiv.org/abs/2312.00752), §3 | связывает input-dependent parameters с SRAM/HBM algorithm; HF docs сильнее только для API/cache | Mamba Figure 1 выше |
| RWKV | [RWKV paper](https://arxiv.org/abs/2305.13048) + [official repo](https://github.com/BlinkDL/RWKV-LM) | paper выводит WKV recurrence, repo показывает state semantics; CS25 даёт обзор без implementation detail | RWKV architecture figure выше |
| RetNet | [RetNet paper](https://arxiv.org/abs/2307.08621) | единственный первичный источник, выводящий parallel/recurrent/chunkwise эквивалентность | RetNet dual-form figure выше |
| associative scan | [Stanford CS336](https://cs336.stanford.edu/) | affine composition делает parallel prefix понятным на уровне оператора; общие HF docs scan не выводят | формула композиции, без новой схемы |
| hybrid | [Jamba report](https://arxiv.org/abs/2403.19887) | даёт реальный layer schedule Mamba+attention+MoE и cache trade-off; обзор Chip Huyen полезен для system metrics, но не архитектуры | Jamba Figure 1 выше |
| serving/eval | [HF Mamba docs](https://huggingface.co/docs/transformers/model_doc/mamba) + [FSDL](https://fullstackdeeplearning.com/llm-bootcamp/) | HF фиксирует recurrent cache interface, FSDL — end-to-end latency/eval; papers в основном сообщают model benchmarks | сравнительная таблица свойств из первичных reports |

- [Stanford CS336](https://cs336.stanford.edu/) — attention alternatives,
  associative scan и hardware efficiency; [CS25](https://web.stanford.edu/class/cs25/)
  — seminar perspective на SSM/recurrent architectures.
- [Hugging Face Mamba docs](https://huggingface.co/docs/transformers/model_doc/mamba)
  и [Jamba docs](https://huggingface.co/docs/transformers/model_doc/jamba) — cache
  и реальные model interfaces.
- [D2L RNN](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) — baseline
  recurrence/BPTT; новые модели сохраняют recurrent inference, меняя оператор.
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/llm-bootcamp/) и
  [Chip Huyen](https://huyenchip.com/2023/04/11/llm-engineering.html) — проверка
  end-to-end latency, memory и deployment constraints.
- Papers/repos: [Mamba](https://arxiv.org/abs/2312.00752),
  [RWKV](https://arxiv.org/abs/2305.13048), [RetNet](https://arxiv.org/abs/2307.08621),
  [Jamba](https://arxiv.org/abs/2403.19887).
