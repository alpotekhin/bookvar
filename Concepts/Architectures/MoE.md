---
title: "MoE"
aliases: [MoE, Mixture of Experts, Sparse Mixture of Experts]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/Mixtral of Experts|Mixtral]]"
  - "[[02 Areas/ML & DL/Papers/DeepSeek-V2|DeepSeek-V2]]"
  - "[[02 Areas/ML & DL/Papers/DeepSeek-V3 Technical Report|DeepSeek-V3]]"
  - "[[02 Areas/ML & DL/Papers/Jamba SSM-Transformer Hybrid|Jamba]]"
courses: []
sources:
  - "[Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer (2017)](https://arxiv.org/abs/1701.06538)"
  - "[Hugging Face — Mixture of Experts Explained](https://huggingface.co/blog/moe)"
---

# MoE — Mixture of Experts

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/mixtral-of-experts/smoe-layer.png]]
*Mixture of Experts Layer: router распределяет токены по экспертам, выходы взвешиваются gating weights (Mixtral paper, 2401.04088)*

## Зачем это нужно: scaling dilemma

[[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] показывают: увеличение числа параметров → лучшее качество. Но увеличение параметров **пропорционально** увеличивает compute на каждый токен. Для 70B модели каждый токен проходит через все 70B параметров.

**MoE разрывает эту связь:** модель может иметь 47B total параметров, но на каждый токен активирует только 13B — получая quality большой модели при compute маленькой.

Ключевое разделение:
- **Total parameters** (определяют capacity/knowledge модели) — сколько параметров хранится
- **Active parameters** (определяют compute на токен) — сколько параметров используется для одного forward pass

## Архитектура MoE Layer

### Sparse vs Dense

В **dense** [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] каждый токен проходит через один [[02 Areas/ML & DL/Concepts/Architectures/Feed-Forward Network|FFN]] блок в каждом слое. В **sparse MoE** FFN заменяется на набор из $N$ экспертов (каждый — отдельный FFN) + router, выбирающий подмножество экспертов для каждого токена.

```
Dense Transformer layer:
  Input → Attention → FFN → Output
  (все параметры FFN задействованы)

MoE Transformer layer:
  Input → Attention → Router → [Expert₁, Expert₂, ..., Expertₙ] → Weighted Sum → Output
  (только top-k экспертов активированы)
```

Attention layers остаются **dense** — все параметры attention используются для каждого токена. MoE заменяет только FFN.

### Router / Gating Function

Router (gating network) — линейный слой, превращающий hidden state токена $x$ в scores для каждого эксперта:

$$g(x) = \text{Softmax}(\text{TopK}(W_g \cdot x, k))$$

**Шаг 1.** Линейная проекция: $s = W_g \cdot x \in \mathbb{R}^N$, где $N$ — число экспертов.

**Шаг 2.** Top-K selection: выбираются $k$ экспертов с наибольшими scores. Остальные зануляются.

**Шаг 3.** Softmax по оставшимся $k$ scores — получаются веса (gates) $g_1, g_2, \ldots, g_k$, $\sum g_i = 1$.

**Шаг 4.** Финальный output — взвешенная сумма:

$$y = \sum_{i \in \text{TopK}} g_i \cdot E_i(x)$$

где $E_i(x)$ — выход $i$-го эксперта для токена $x$.

### Конкретный пример: Mixtral 8x7B

[[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral]] 8x7B:
- 8 экспертов ($N = 8$) в каждом MoE-слое
- Top-2 routing ($k = 2$) — каждый токен активирует 2 из 8 экспертов
- Каждый эксперт ≈ FFN из Mistral 7B
- **Total params: ~47B** (8 копий FFN + shared attention + embeddings)
- **Active params per token: ~13B** (2 эксперта + shared layers)
- Quality сравнима с LLaMA 2 70B, но inference в ~6x быстрее

## Expert Balancing: проблема и решения

### Проблема: expert collapse

Без ограничений router быстро обучается отправлять большинство токенов в 1-2 «любимых» эксперта. Остальные эксперты недообучаются → модель деградирует до dense модели с размером 1-2 экспертов.

Это **positive feedback loop**: эксперт получает больше токенов → обучается лучше → router отправляет ещё больше токенов.

### Auxiliary Load Balancing Loss

Стандартное решение — дополнительная loss, штрафующая неравномерную загрузку экспертов:

$$\mathcal{L}_{balance} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

где:
- $f_i$ — доля токенов в batch, отправленных в эксперт $i$
- $P_i$ — средняя вероятность (gate score) для эксперта $i$ по batch
- $\alpha$ — коэффициент (обычно 0.01)
- $N$ — число экспертов

Произведение $f_i \cdot P_i$ минимизируется, когда $f_i = P_i = 1/N$ (равномерное распределение). Loss пропорциональна **корреляции** между popularity (сколько токенов получает эксперт) и confidence (насколько router уверен в эксперте).

### Capacity Factor

**Capacity factor** $C$ ограничивает максимальное число токенов, которое может обработать один эксперт:

$$\text{Expert buffer size} = C \cdot \frac{k \cdot T}{N}$$

где $T$ — число токенов в batch, $k$ — top-k, $N$ — число экспертов. При $C = 1.0$ каждый эксперт получает ровно свою «справедливую долю». При $C > 1$ допускается небольшой дисбаланс (обычно $C = 1.1$-$1.25$).

Токены, превышающие capacity буфера, **дропаются** (пропускают MoE слой через residual connection). Это гарантирует bounded compute, но может ухудшить качество при сильном дисбалансе.

### DeepSeek-V3: вспомогательный bias вместо loss

[[02 Areas/ML & DL/Papers/DeepSeek-V3 Technical Report|DeepSeek-V3]] предложил **auxiliary-loss-free balancing**: вместо дополнительной loss используется learnable bias term для каждого эксперта, который корректируется в зависимости от текущей загрузки. Это позволяет точнее контролировать баланс без искажения основного training signal.

## Expert Parallelism: распределённое обучение

MoE создаёт уникальные возможности и вызовы для distributed training:

### All-to-All Communication

При tensor parallelism каждый GPU хранит часть каждого эксперта. При expert parallelism каждый GPU хранит **целые эксперты** (например, 2 из 8). Routing требует **all-to-all** коммуникации: каждый GPU должен отправить токены другим GPU, где находятся нужные эксперты, и получить результаты обратно.

```
GPU 0: Expert₁, Expert₂     GPU 1: Expert₃, Expert₄
GPU 2: Expert₅, Expert₆     GPU 3: Expert₇, Expert₈

Token X → Router → needs Expert₃, Expert₇
  → Send X to GPU 1 (Expert₃) and GPU 3 (Expert₇)
  → Receive results, weighted sum
```

All-to-all — дорогая коммуникация. Именно поэтому expert parallelism часто комбинируется с другими стратегиями (data parallelism, pipeline parallelism).

### Memory trade-off

MoE модели занимают **больше GPU memory** при inference, чем dense модели с тем же числом active parameters: все эксперты должны быть загружены, хотя для каждого токена используется только подмножество. Mixtral 8x7B требует ~90 GB для весов (FP16), хотя active params — 13B (~26 GB).

## Fine-tuning MoE: вызовы

### Expert collapse при fine-tuning

При fine-tuning на небольшом датасете routing часто деградирует: модель «забывает» использовать часть экспертов, и few experts начинают доминировать. Это особенно проблематично для instruction tuning, где distribution данных сильно отличается от pre-training.

### Routing degradation

Router обучен на pre-training distribution. При fine-tuning на другой distribution (например, code → medical) routing patterns могут стать неоптимальными. Frozen router иногда работает лучше, чем fine-tuned — парадоксальный результат, объясняемый overfitting router на малом датасете.

### LoRA placement

Для dense моделей LoRA однозначно применяется к attention + FFN проекциям. Для MoE модели возникает вопрос: **куда ставить LoRA?**

- **Только shared layers (attention)** — самый простой вариант, не трогает экспертов
- **Все эксперты** — максимальная expressiveness, но $N\times$ больше параметров LoRA
- **Router + shared** — может изменить routing patterns
- **Только активные эксперты** — зависит от данных, нестабильно

На практике «attention-only LoRA» часто даёт лучший trade-off для MoE fine-tuning. Но однозначного консенсуса нет.

## Использование в современных моделях

### Mixtral 8x7B (Mistral AI, 2023)

Первая массово доступная open-source MoE LLM. 8 экспертов, top-2, 47B total / 13B active. Показала, что MoE может быть практичным — inference на 2x A100 вместо 8x для сравнимого по качеству dense LLaMA 2 70B.

### Switch Transformer (Google, 2021)

Пионер modern MoE: top-1 routing (только 1 эксперт на токен), упрощённый gating. Показал scaling до 1.6T параметров. Ввёл capacity factor и simplified load balancing.

### DeepSeek-V2 / V3 (2024)

DeepSeek-V2 использует **fine-grained experts**: 160 экспертов, top-6 routing. Каждый эксперт маленький, но routing более гранулярный. DeepSeek-V3 — 256 экспертов (8 shared + 256 routed), top-8. Auxiliary-loss-free balancing (bias вместо loss).

### Jamba (AI21, 2024)

[[02 Areas/ML & DL/Papers/Jamba SSM-Transformer Hybrid|Jamba]] — гибрид Mamba (SSM) + Transformer + MoE. MoE применяется к Transformer слоям (не к Mamba). 16 экспертов, top-2. Демонстрирует, что MoE совместим с не-transformer архитектурами.

## Масштабирование и Scaling Laws для MoE

MoE подчиняются **модифицированным [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]]**: качество модели больше коррелирует с total parameters, чем с active parameters, но **не линейно**. Модель с 47B total / 13B active не эквивалентна 47B dense — она ближе к ~30-35B dense по quality, при ~13B dense compute cost.

Эмпирическое правило: MoE модель с $N$ экспертами и top-$k$ routing ведёт себя как dense модель размером примерно $2-3\times$ от active parameters (не $N/k\times$). Дополнительные эксперты дают diminishing returns.

## Сравнение Dense vs MoE

| Свойство | Dense | MoE |
|----------|-------|-----|
| Params on disk | = active params | >> active params |
| Compute per token | Пропорционально total params | Пропорционально active params |
| GPU memory (inference) | Пропорционально total params | Пропорционально total params (!) |
| Training comm | Standard AllReduce | AllReduce + All-to-All |
| Fine-tuning | Straightforward | Expert collapse, routing issues |
| Inference hardware | Предсказуемо | Нужно больше GPU memory |
| Quality vs compute | Baseline | Лучше (больше knowledge per FLOP) |

**Ключевой trade-off MoE:** выигрыш в compute per token, но проигрыш в memory (все эксперты должны быть в памяти). MoE выгоден, когда bottleneck — compute (training, high-throughput serving), и менее выгоден, когда bottleneck — memory (single-user inference на consumer GPU).

## Related concepts

- [[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral]] — первая массовая open-source MoE LLM
- [[02 Areas/ML & DL/Concepts/Architectures/Feed-Forward Network|Feed-Forward Network]] — компонент Transformer, который MoE заменяет набором экспертов
- [[02 Areas/ML & DL/Concepts/Training/Scaling Laws|Scaling Laws]] — MoE изменяет trade-off между параметрами и compute
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, в которой применяется MoE
- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-V3|DeepSeek-V3]] — fine-grained MoE с auxiliary-loss-free balancing
- [[02 Areas/ML & DL/Concepts/Architectures/Jamba|Jamba]] — гибрид SSM + Transformer + MoE
