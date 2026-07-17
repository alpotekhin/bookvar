---
title: Атлас современных LLM
type: textbook-chapter
status: living
last_updated: 2026-07-17
as_of: 2026-07-17
previous: "[[02 Areas/ML & DL/00 Учебник/09 Dense FFN и Mixture of Experts/01 От SwiGLU к MoE]]"
next: "[[02 Areas/ML & DL/00 Учебник/10 Атлас современных архитектур/02 Альтернативы Transformer и multimodality]]"
---

# Атлас: Llama, Qwen, DeepSeek, GLM и Kimi

> [!abstract] Задача главы
> Название семейства почти ничего не говорит о механике конкретного checkpoint.
> Атлас учит разбирать release по независимым слоям: backbone, attention, FFN,
> position/context, training objective, post-training и serving. Это живая
> страница: новые releases добавляются строкой в timeline и отдельной
> версионированной карточкой, а старые утверждения не переписываются задним
> числом.

> [!warning] Дата среза
> Проверено по доступным первичным источникам на **17 июля 2026 года**. Поля,
> которых разработчик не раскрыл, обозначаются `не раскрыто`, а не
> восстанавливаются из слухов.

## Почему обычный список моделей быстро становится неверным

Фразы вроде:

```text
Llama использует GQA.
Qwen — dense model.
DeepSeek — reasoning architecture.
```

не выдерживают проверки.

- LLaMA 1 была MHA, GQA появилась не во всех Llama 2 sizes.
- Qwen3 включает dense и MoE checkpoints.
- DeepSeek-R1 reasoning behavior получен post-training поверх V3-family
  architecture; «reasoning» не является новым attention layer.

Правильная единица атласа — **release + checkpoint**, а не бренд.

## Семь слоёв описания

Для каждой модели заполняем один template:

```yaml
release:
checkpoint:
date:

backbone:
  topology: decoder-only | encoder-decoder | hybrid
  dense_or_moe:

attention:
  type: MHA | MQA | GQA | MLA | hybrid | sparse
  local_or_global:
  kv_cache:

ffn:
  activation:
  routed_experts:
  shared_experts:
  experts_per_token:

position_and_context:
  encoding:
  trained_context:
  extension_method:

training:
  tokenizer:
  data_tokens:
  auxiliary_objectives:
  precision_and_parallelism:

post_training:
  sft:
  preference_or_rl:
  reasoning_mode:
  tool_use:

evidence:
  primary_source:
  config_or_code:
  confidence:
```

Architecture не смешивается с recipe:

| Утверждение | Слой |
|---|---|
| используется MLA | attention |
| 671B total, 37B active | FFN/MoE |
| обучали в FP8 | training system |
| Multi-Token Prediction | training objective |
| научили длинному CoT через RL | post-training |
| runtime использует FlashMLA | inference system |

## Базовая система координат

Возьмём LLaMA-style block:

```text
RMSNorm → causal attention with RoPE → residual
RMSNorm → SwiGLU FFN              → residual
```

Дальнейшие releases удобно описывать как изменения отдельных осей:

![[02 Areas/ML & DL/00 Учебник/Assets/Model innovation matrix.md]]

Если embedded canvas неудобен в текущей теме Obsidian, используйте таблицу ниже.

| Ось | Варианты |
|---|---|
| attention memory | MHA → GQA/MQA → MLA |
| attention span | full → local/global hybrid → sparse/linear hybrid |
| FFN | dense SwiGLU → top-k MoE → fine-grained/shared experts |
| modalities | text only → external encoder/projector → early fusion |
| generation | direct → thinking/non-thinking → tool-integrated reasoning |

## Llama: от открытого dense baseline к multimodal MoE

### LLaMA 1

Опорная точка предыдущей главы:

- decoder-only;
- MHA;
- RMSNorm;
- RoPE;
- SwiGLU;
- dense FFN.

[LLaMA paper](https://arxiv.org/abs/2302.13971) важен не только результатами:
он явно перечисляет происхождение architectural choices.

### Llama 2

Recipe блока в основном сохранён. Важные различия:

- context увеличен относительно LLaMA 1;
- GQA использована в 70B, но не должна приписываться всем размерам;
- отдельный Llama 2-Chat post-training включает SFT и RLHF;
- architecture base и alignment recipe описываются раздельно.

Источник: [Llama 2 paper](https://arxiv.org/abs/2307.09288).

### Llama 3 / 3.1

Llama 3 продолжает ту же линию, но:

- использует GQA в опубликованных основных sizes;
- vocabulary увеличена до 128K;
- существенно увеличены training data и масштабы post-training;
- Llama 3.1 расширяет context и model lineup.

Новый tokenizer меняет embeddings/output shapes и token efficiency, но не
формулу Transformer block.

### Llama 4 Scout и Maverick

Здесь семейство меняется сильнее:

- autoregressive MoE;
- early-fusion multimodal input;
- alternating dense и MoE layers;
- shared expert + routed experts;
- 17B active parameters для опубликованных Scout/Maverick variants.

Meta сообщает для Maverick 128 routed experts: token проходит через shared
expert и один routed expert. Scout использует другую конфигурацию, поэтому
«Llama 4 = 128 experts» тоже нельзя превращать в семейное правило.

Для Llama 4 authoritative visual source — официальный
[release page](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) и model
card. Не следует иллюстрировать этот раздел схемой Llama 3: она относится к
предыдущей generation и создаёт ложное впечатление, что block не изменился.

### Timeline Llama

| Release | Backbone | Attention/FFN | Главное изменение другого слоя |
|---|---|---|---|
| LLaMA 1 | dense decoder | MHA + SwiGLU | открытый efficient baseline |
| Llama 2 | dense decoder | MHA; GQA в 70B | longer context, chat RLHF |
| Llama 3/3.1 | dense decoder | GQA + SwiGLU | 128K vocab, больше data/context |
| Llama 4 | multimodal MoE decoder | routed/shared experts | early fusion vision |

## Qwen: одна линия, много размеров и режимов

### Qwen2 и Qwen2.5

Qwen2-family близка к LLaMA-like decoder:

- GQA;
- RoPE;
- SwiGLU;
- RMSNorm;
- QKV bias в Qwen2 architecture;
- dense и специализированные derivatives.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/qwen25-figure1-hq.png]]

*Qwen2.5 report показывает lineup разных размеров и специализаций. Figure
описывает семейство releases, а не один универсальный block.*

Qwen2.5 improvement нельзя объяснить одним новым layer: data quality, количество
tokens, long-context training и post-training являются существенной частью.

### Qwen3

Qwen3 выпускается в dense и MoE variants. Technical report указывает:

- QK-Norm;
- удаление QKV bias относительно Qwen2;
- dense и MoE backbones;
- для MoE — много fine-grained experts без shared experts в описанной Qwen3
  configuration;
- unified thinking и non-thinking modes.

Последний пункт — post-training/interface, не новый вид attention.

```text
same base model
  ├→ non-thinking response
  └→ thinking trajectory → final response
```

Thinking budget определяет test-time compute. Он не изменяет количество layers
checkpoint.

### Qwen3-Next и последующая hybrid line

В Qwen3-Next architecture появляются:

- hybrid attention: gated linear attention layers перемежаются full attention;
- ultra-sparse MoE;
- Multi-Token Prediction;
- stability changes для масштабирования.

Это уже не просто «ещё один размер Qwen3»: меняется sequence mixer. Конкретные
Qwen3.5 checkpoints, основанные на этой линии, нужно описывать собственной
карточкой и official config, потому что product/API name не гарантирует один
открытый backbone.

### Timeline Qwen

| Release | Backbone | Attention/FFN | Post-training |
|---|---|---|---|
| Qwen2 | dense decoder | GQA, QKV bias, SwiGLU | instruct variants отдельно |
| Qwen2.5 | dense family | LLaMA-like GQA recipe | усилены code/math/long context |
| Qwen3 dense | dense decoder | QK-Norm, no QKV bias | unified thinking modes |
| Qwen3 MoE | sparse decoder | fine-grained top-k MoE | те же два режима |
| Qwen3-Next line | hybrid sparse | linear/full attention + ultra-sparse MoE | MTP и reasoning variants |

Первичный маршрут:
[Qwen3 report](https://arxiv.org/abs/2505.09388),
[official Qwen3 post](https://qwenlm.github.io/blog/qwen3/) и config конкретного
checkpoint.

## DeepSeek: architecture и reasoning особенно легко перепутать

### DeepSeek-V2

Два основных architectural changes:

- MLA вместо MHA/GQA;
- DeepSeekMoE вместо обычного dense FFN.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/deepseek-v2/x1.png]]

*Overview DeepSeek-V2: MLA и DeepSeekMoE находятся внутри model architecture.*

### DeepSeek-V3

V3 сохраняет MLA и DeepSeekMoE и добавляет/развивает:

- auxiliary-loss-free load balancing;
- Multi-Token Prediction auxiliary objective;
- FP8 training;
- DualPipe pipeline schedule.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/deepseek-v3-figure2-architecture-hq.png]]

*Figure 2 из DeepSeek-V3 report: слева полный Transformer block, справа отдельно
показаны DeepSeekMoE и MLA. Training objectives, FP8 и DualPipe в эту схему не
входят и рассматриваются как другие слои системы.*

Классификация:

| Innovation | Слой |
|---|---|
| MLA | architecture / attention |
| fine-grained + shared experts | architecture / FFN |
| bias-based balancing | routing/training |
| MTP | auxiliary objective |
| FP8 | numerical training system |
| DualPipe | distributed schedule |

### DeepSeek-R1

R1-Zero/R1 — прежде всего post-training research line:

- base берётся из DeepSeek-V3 family;
- RLVR-style rewards усиливают reasoning behavior;
- R1 добавляет cold-start data и многостадийный pipeline;
- distillation переносит reasoning patterns в меньшие dense models.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/deepseek-r1-figure1-hq.png]]

*Pipeline из DeepSeek-R1 paper. На этой схеме меняется путь обучения, а не
формула MLA или MoE block.*

Поэтому корректно:

```text
DeepSeek-V3 architecture + R1 post-training
```

а не:

```text
R1 attention architecture
```

### Как добавить новый DeepSeek release

Если появляется «DeepSeek-X with RLVR»:

1. определить base checkpoint;
2. проверить, менялись ли config/code architecture;
3. отдельно записать reward, data и RL algorithm;
4. отдельно записать inference/test-time policy;
5. не создавать новый architectural family без изменения forward graph.

## GLM: нельзя переносить ранний blank infilling на всё семейство

Исторический GLM использовал autoregressive blank infilling и 2D positional
encoding. Это важная ветвь развития, но современные GLM releases должны читаться
по собственным reports.

### GLM-4

[ChatGLM family report](https://arxiv.org/abs/2406.12793) описывает путь от
GLM-130B к GLM-4 и tool-oriented post-training. Для открытого GLM-4-9B нужно
смотреть model config: свойства закрытого flagship нельзя автоматически
переносить на open checkpoint.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/glm4-figure1-hq.png]]

*Локальная figure из GLM-4 report должна читаться вместе с caption оригинала:
она может описывать training/evaluation family, а не точный block diagram.*

### GLM-4.5 → 4.7

Официальный repository описывает GLM-4.5 как MoE family:

- GLM-4.5: 355B total / 32B active;
- GLM-4.5-Air: 106B total / 12B active;
- thinking и non-thinking modes;
- focus на reasoning, coding и agents.

GLM-4.6 расширяет context и улучшает post-training. GLM-4.7 добавляет
agent-oriented режимы вроде preserved thinking и turn-level thinking.
Эти product capabilities нельзя автоматически называть новым backbone:
точные architectural changes считаются подтверждёнными только при наличии
report/config.

Источник текущего статуса:
[official GLM repository](https://github.com/zai-org/GLM-4.5).

## Kimi: long context, RL и крупный MoE — разные линии

### Kimi k1.5

k1.5 paper важен как работа о multimodal long-context reinforcement learning и
long chain-of-thought. Это прежде всего post-training/reasoning contribution, а
не новая базовая attention formula.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/kimi-k15/x1.png]]

*Figure из Kimi k1.5: используйте её для training/reasoning pipeline, не как
универсальную архитектуру всех Kimi.*

### Kimi K2

Official Kimi K2 report/config:

- decoder MoE;
- около 1T total / 32B active parameters;
- 61 layers;
- 384 experts, top-8, один shared expert;
- MLA;
- SwiGLU;
- 128K context в исходном K2 release;
- MuonClip и large-scale training stability work;
- agentic post-training.

Это сходство с DeepSeek на axes MLA/MoE, но не идентичность всей system.

Источник:
[MoonshotAI/Kimi-K2](https://github.com/MoonshotAI/Kimi-K2).

### Поздние K2 variants

У variants вроде K2.x/Code могут меняться context, multimodality и
post-training. Например официальный Kimi K2.7 Code page указывает 256K context
и MoonViT vision encoder. Такие изменения добавляются отдельной строкой:
исходный K2 не переписывается задним числом с 128K на 256K.

## Сводная матрица

| Release | Attention | FFN | Modalities | Reasoning mode |
|---|---|---|---|---|
| LLaMA 1 | MHA | dense SwiGLU | text | нет отдельного |
| Llama 3 | GQA | dense SwiGLU | text | post-trained instruct |
| Llama 4 | GQA-family release details | MoE, shared+routed | early-fusion vision+text | post-training |
| Qwen2.5 | GQA | dense SwiGLU | text line; VL отдельно | instruct variants |
| Qwen3 | full attention + QK-Norm | dense или MoE | text base line | unified thinking/non-thinking |
| Qwen3-Next line | linear/full hybrid | ultra-sparse MoE | release-specific | thinking/instruct variants |
| DeepSeek-V2/V3 | MLA | fine-grained MoE + shared | text | V3 chat separately |
| DeepSeek-R1 | MLA inherited | MoE inherited | text | RL reasoning pipeline |
| GLM-4.5 | report/config-specific | MoE | text; V line отдельно | hybrid modes |
| Kimi K2 | MLA | 384 experts, top-8 + shared | text | agentic post-training |

Пустое или осторожное поле лучше уверенной ошибки. Для production решения
матрицу дополняют exact config конкретного checkpoint.

## Как читать model config

Минимальный script:

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

fields = [
    "model_type",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "max_position_embeddings",
    "rope_theta",
]

for name in fields:
    print(f"{name:28}", getattr(cfg, name, "not disclosed here"))
```

Названия различаются по codebase. Config подтверждает shapes, но не раскрывает
data mixture или post-training. Для этого нужен report/model card.

## Как добавлять новую работу: пример RLVR

Пусть выходит `DeepSeek-X RLVR`.

### Шаг 1. Создать release card

```yaml
release: DeepSeek-X-RLVR
base_checkpoint: DeepSeek-X-Base
architecture_delta: none reported
post_training_delta:
  - verifiable reward
  - RL algorithm
  - training tasks
inference_delta:
  - reasoning budget
```

### Шаг 2. Встроить по концепции

- paper идёт в [[02 Areas/ML & DL/Papers]];
- RLVR mechanism — в главу post-training;
- изменение reasoning policy — в test-time compute;
- строка release — в семейство DeepSeek;
- новая architecture page создаётся только при новом forward graph.

### Шаг 3. Связать вопрос

В список [[02 Areas/ML & DL/Вопросы по NLP]] добавляется вопрос:

```text
Что именно изменилось между base model и RLVR checkpoint?
```

Ссылка ведёт не на дублированный ответ, а на секцию post-training chapter и
release card.

Так новые papers расширяют граф знаний, не ломая оглавление.

## Красные флаги при сравнении моделей

> [!danger] Сравнивать total parameters dense и MoE
> Нужны одновременно total, active, FLOPs/token и memory footprint.

> [!danger] Называть thinking архитектурой
> Чаще это training/inference policy поверх того же backbone.

> [!danger] Переносить context API на trained context
> Serving provider может ограничить или расширить window. Нужны model card и
> validation, а не только API field.

> [!danger] Переносить свойство flagship на малый checkpoint
> Heads, MoE, context и tokenizer могут отличаться внутри release.

> [!danger] Считать семейство линейной эволюцией
> Qwen, GLM, Kimi и DeepSeek взаимно заимствуют идеи и развивают параллельные
> research lines; LLaMA — удобная координата, не единственный предок.

## Практика

### 1. Архитектурный diff

Выберите Llama 3, Qwen3 и DeepSeek-V3. Для каждого заполните YAML template только
по primary sources/config. Затем сделайте diff:

```text
Llama 3 → Qwen3: QK-Norm, dense/MoE lineup, post-training modes
Llama 3 → DeepSeek-V3: GQA→MLA, dense→fine-grained MoE, MTP/training system
```

### 2. Проверка утверждений

Для каждой строки таблицы поставьте evidence grade:

- A — paper + code/config;
- B — official model card/blog;
- C — inference из runtime code;
- D — community report.

В canonical text допускаются A/B; C помечается как вывод; D хранится как lead,
но не как подтверждённый факт.

### 3. Обновление без перезаписи истории

Добавьте fictitious Qwen4:

1. новая карточка с датой;
2. новая строка timeline;
3. links на новые concepts;
4. старые Qwen3 sections остаются неизменными;
5. спорные поля — `unknown`.

## После главы нужно уметь

- отделять block architecture от training и post-training;
- объяснить evolution каждой семьи без рекламного списка;
- читать exact checkpoint config;
- не смешивать R1/k1.5 reasoning с новым attention;
- добавить новый RLVR paper в несколько узлов базы без дублирования;
- указывать дату среза и степень доказательности.

## Primary source index

- [LLaMA](https://arxiv.org/abs/2302.13971),
  [Llama 2](https://arxiv.org/abs/2307.09288),
  [Llama 4 official release](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).
- [Qwen3 report](https://arxiv.org/abs/2505.09388),
  [Qwen official blog](https://qwenlm.github.io/blog/qwen3/).
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434),
  [DeepSeek-V3](https://arxiv.org/abs/2412.19437),
  [DeepSeek-R1](https://arxiv.org/abs/2501.12948).
- [ChatGLM family report](https://arxiv.org/abs/2406.12793),
  [GLM current official repository](https://github.com/zai-org/GLM-4.5).
- [Kimi k1.5](https://arxiv.org/abs/2501.12599),
  [Kimi K2 official repository](https://github.com/MoonshotAI/Kimi-K2).

## Связанные узлы

- [[02 Areas/ML & DL/02 Атлас моделей/Семейства/Llama]]
- [[02 Areas/ML & DL/02 Атлас моделей/Семейства/Qwen]]
- [[02 Areas/ML & DL/02 Атлас моделей/Семейства/DeepSeek]]
- [[02 Areas/ML & DL/02 Атлас моделей/Сравнения/Сравнение семейств]]

**Дальше:** [[02 Areas/ML & DL/00 Учебник/10 Атлас современных архитектур/02 Альтернативы Transformer и multimodality|что меняется, если отказаться от полного attention: Mamba, RWKV, RetNet и гибриды.]]
