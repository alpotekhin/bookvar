---
title: "DeepSeek-V3"
aliases: [DeepSeek-V3, DeepSeek V3, DS-V3]
type: concept
status: legacy
category: Architectures
papers:
  - "[[02 Areas/ML & DL/Papers/DeepSeek-V3 Technical Report|DeepSeek-V3]]"
courses: []
sources:
  - "[DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437)"
  - "[Towards Data Science — DeepSeek-V3 Explained: Multi-head Latent Attention](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/)"
  - "[DeepWiki — DeepSeek-V3 Architecture Overview](https://deepwiki.com/deepseek-ai/DeepSeek-V3/1.2-model-architecture-overview)"
  - "[VitalAB — DeepSeek-V3 Technical Report Review](https://vitalab.github.io/article/2025/02/11/DeepSeekV3.html)"
  - "[Lior Sinai — DeepSeek's Multi-Head Latent Attention](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html)"
---

# DeepSeek-V3

## Зачем эта модель появилась

К концу 2024 года обучение frontier-моделей стоило сотни миллионов долларов (GPT-4 — ~$100M, Gemini Ultra — предположительно больше). DeepSeek (Китай) задал вопрос: **можно ли обучить модель уровня GPT-4o за $5.5M?** Ответ — DeepSeek-V3: MoE-модель с 671B параметров (37B активных на токен), обученная на 14.8T токенов за 2.788M GPU-часов на H800.

Результат: DeepSeek-V3 сравнялась с GPT-4o и Claude 3.5 Sonnet на бенчмарках MMLU, MATH, HumanEval — при стоимости обучения в **20-50 раз ниже** конкурентов.

## Три столпа архитектуры

DeepSeek-V3 стоит на трёх инновациях:
1. **Multi-head Latent Attention (MLA)** — эффективный inference через сжатие KV-cache
2. **DeepSeekMoE** — экономичный training через fine-grained Mixture-of-Experts
3. **FP8 Mixed Precision Training** — первое успешное FP8-обучение на модели такого масштаба

## Multi-head Latent Attention (MLA): глубокий разбор

### Проблема: KV-cache как bottleneck

Стандартный Multi-Head Attention хранит в KV-cache полные Key и Value для каждой головы: $2 \times n_h \times d_h$ на токен. При 128 головах и длинном контексте это десятки гигабайт. Существующие решения (GQA, MQA) уменьшают число KV-голов, но **теряют представительную мощность** — все query-головы вынуждены работать с одними и теми же ключами/значениями.

### Идея MLA: сжатие через латентное пространство

Вместо уменьшения числа голов, MLA сжимает **само представление** KV в низкоразмерный латентный вектор $c^{KV}$, а при вычислении attention — восстанавливает Key и Value:

$$c_t^{KV} = W^{DKV} h_t$$

$$k_t = W^{UK} c_t^{KV}, \quad v_t = W^{UV} c_t^{KV}$$

где $W^{DKV} \in \mathbb{R}^{d_c \times d}$ — матрица сжатия (down-projection), $W^{UK}, W^{UV}$ — матрицы восстановления (up-projection), $d_c \ll n_h \cdot d_h$.

### Почему это работает

Информация в Key и Value **сильно избыточна** между головами — разные головы часто кодируют похожие паттерны. Латентный вектор $c^{KV}$ захватывает «суть» токена в компактном пространстве. В кэше хранится только $c^{KV}$, что сокращает KV-cache на **93.3%** по сравнению с MHA (результат из DeepSeek-V2).

### Математическая деталь: поглощение up-projection

Ключевой трюк MLA — матрицы up-projection $W^{UK}$ и $W^{UV}$ можно **поглотить в матрицы проекций** Query и Output при inference. Это означает, что мы **никогда не вычисляем** полноразмерные K и V явно, а работаем напрямую с $c^{KV}$ через модифицированные проекции. Вычислительная стоимость attention остаётся такой же, но KV-cache уменьшается на порядок.

### Decoupled RoPE для позиционной информации

Стандартный RoPE (Rotary Position Embeddings) применяется к Key, что **конфликтует** со сжатием: после RoPE Key зависит от позиции, и его нельзя восстановить из $c^{KV}$, хранящего position-independent информацию.

Решение MLA — **decoupled RoPE**: отдельный компактный вектор $k_t^{PE}$ для позиционной информации, конкатенированный с content-based ключом:

$$k_t = [W^{UK} c_t^{KV}; \; \text{RoPE}(W^{KR} c_t^{KV})]$$

Этот трюк позволяет хранить в кэше и $c^{KV}$, и маленький $k_t^{PE}$, сохраняя и сжатие, и позиционную информацию.

### MLA vs GQA vs MQA: сравнение

| Метод | KV-cache на токен | Качество | Представительная мощность |
|-------|-------------------|----------|--------------------------|
| MHA | $2 n_h d_h$ | Baseline | Максимум |
| MQA | $2 d_h$ | Хуже | Все головы делят 1 KV |
| GQA-8 | $2 \times 8 d_h$ | Близко к MHA | 8 групп KV |
| **MLA** | $d_c + d_h^{PE}$ | **Лучше GQA** | Каждая голова уникальна |

Подробнее: [[02 Areas/ML & DL/Concepts/Training/Multi-head Latent Attention|Multi-head Latent Attention]]

## DeepSeekMoE: fine-grained экспертная маршрутизация

Стандартный MoE (как в Mixtral) использует 8 крупных экспертов, активируя 2. DeepSeekMoE идёт дальше:

- **256 мелких экспертов** вместо 8 крупных — каждый эксперт специализируется на более узком знании
- **8 активных экспертов** на токен + **1 shared expert** (всегда активен, содержит общие знания)
- **Hidden dimension** каждого эксперта уменьшена в $m$ раз, но экспертов в $m$ раз больше

**Зачем fine-grained:** крупный эксперт неизбежно содержит смесь разных знаний (knowledge hybridity). Мелкие эксперты обеспечивают более чистую декомпозицию знаний, что улучшает маршрутизацию и качество.

### Auxiliary-loss-free балансировка нагрузки

Классическая проблема MoE: одни эксперты получают все токены, другие простаивают. Обычно решается auxiliary loss (дополнительная функция потерь, штрафующая дисбаланс). Но auxiliary loss **вредит** основной задаче — он отвлекает оптимизацию от language modeling.

DeepSeek-V3 вводит **bias-based балансировку** без auxiliary loss: к логитам маршрутизатора добавляется обучаемый bias $b_i$. Если эксперт перегружен — bias уменьшается, если недогружен — увеличивается. Это мягко выравнивает нагрузку без деградации качества.

### Multi-Token Prediction (MTP)

Дополнительный training objective: модель предсказывает не один, а **несколько следующих токенов** одновременно. Каждый дополнительный токен предсказывается через отдельный prediction head, каждый из которых получает на вход causal-concatenation представления текущей позиции и предсказанного предыдущего токена.

**Зачем MTP:** это обеспечивает более dense gradient signal (больше supervision на каждый forward pass) и улучшает representation learning. MTP может использоваться для speculative decoding при inference — ускорение генерации за счёт параллельного предсказания.

## FP8 Mixed Precision: обучение за полцены

DeepSeek-V3 — первая модель масштаба 671B, обученная с FP8 (8-bit floating point) для основных вычислений:

- **Forward/backward GEMM** (матричные умножения) — FP8
- **Накопление, нормализация, attention** — BF16/FP32
- **Master weights** — FP32

### Почему FP8 критично

FP8 GEMM на H800/H100 даёт ~2x throughput по сравнению с BF16. При 2.788M GPU-часов обучения это экономит миллионы долларов.

### Решение проблемы точности

**Проблема:** FP8 (формат E4M3) имеет узкий динамический диапазон: [-448, 448] с только 3 битами мантиссы. Наивная квантизация приводит к divergence при обучении.

**Tile-wise quantization:** вместо одного scaling factor на весь тензор, используются отдельные факторы для каждого блока (tile) размером $128 \times 128$. Это позволяет точнее отслеживать локальные распределения значений.

**Online quantization:** scaling factor вычисляется из **текущего** тензора, а не из статистики предыдущего шага (как в delayed quantization). Это критически важно для стабильности — распределение активаций меняется от шага к шагу.

**High-precision accumulation:** промежуточные результаты GEMM накапливаются в FP32, чтобы избежать потери точности при суммировании многих малых слагаемых.

## Инфраструктура обучения: DualPipe

### Проблема: communication overhead в MoE

Для MoE-модели главный bottleneck — **all-to-all коммуникация** между узлами. Каждый токен отправляется на GPU с нужным экспертом и возвращается обратно. При 256 экспертах и тысячах GPU это колоссальный объём сетевого трафика.

### Решение: перекрытие computation и communication

DeepSeek разработал **DualPipe** — схему pipeline parallelism с двунаправленным планированием:

1. Pipeline разбивается на micro-batches
2. Два потока micro-batches идут в **противоположных направлениях** по pipeline
3. Forward computation одного micro-batch перекрывается с all-to-all communication другого
4. Backward pass аналогично перекрывается с forward pass следующего micro-batch

**Результат:** достигнуто **почти полное перекрытие** — communication практически не добавляет latency. Это снижает pipeline bubble (простой GPU) по сравнению со стандартными схемами (1F1B, interleaved 1F1B).

### Кластер обучения

- **2048 GPU NVIDIA H800** (кластер DeepSeek)
- **InfiniBand IB** между узлами внутри ноды, **NVLink** между GPU внутри ноды
- Каждая нода: 8 GPU с 80GB HBM3

## Результаты

| Бенчмарк | DeepSeek-V3 | GPT-4o | Claude 3.5 Sonnet | LLaMA 3.1 405B |
|-----------|-------------|--------|--------------------|----------------|
| MMLU | 88.5 | 87.2 | 88.3 | 88.6 |
| MATH-500 | 90.2 | 74.6 | 78.3 | 73.8 |
| HumanEval | 82.6 | 90.2 | 93.7 | 61.0 |
| Codeforces | 51.6 | 23.6 | 20.3 | 25.3 |
| GPQA Diamond | 59.1 | 53.6 | 65.0 | 51.1 |
| MMLU-Pro | 75.9 | 72.6 | 78.0 | 73.3 |

Обучение на **2048 GPU H800** за **~2 месяца**. Стоимость: **$5.576M** (только GPU-часы, без учёта R&D).

## Разбивка стоимости обучения

| Этап | GPU-часы | % от total |
|------|----------|------------|
| Pre-training (14.8T токенов) | 2,664,000 | 95.6% |
| Context extension (32K → 128K) | 119,000 | 4.3% |
| Post-training (SFT + RL) | 5,000 | 0.2% |
| **Total** | **2,788,000** | **100%** |

При цене H800 ~$2/GPU-час = **$5,576,000**. Для сравнения:
- GPT-4: ~$100M (оценки)
- Gemini Ultra: ~$191M (оценки)
- LLaMA 3 405B: ~$30M (оценки на 30.8M GPU-часов H100)

## Почему $5.5M — это revolution

DeepSeek-V3 показала, что **архитектурные инновации** (MLA + fine-grained MoE + FP8) могут компенсировать разницу в compute budget на порядок. Это сдвинуло дискуссию от «у кого больше GPU» к «у кого лучше архитектура».

Три конкретных фактора экономии:
1. **MoE:** 671B knowledge в 37B compute — ~18x leverage
2. **FP8:** ~2x throughput per GPU vs BF16
3. **DualPipe:** ~100% GPU utilization, минимальные pipeline bubbles

Итоговый effective compute: **~36x** ниже, чем у dense BF16 модели эквивалентного качества.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/Multi-head Latent Attention|Multi-head Latent Attention]] — механизм сжатия KV-cache
- [[02 Areas/ML & DL/Concepts/Architectures/DeepSeek-R1|DeepSeek-R1]] — reasoning-модель, построенная поверх V3
- [[02 Areas/ML & DL/Concepts/Architectures/Mixtral of Experts|Mixtral of Experts]] — другой подход к MoE
- [[02 Areas/ML & DL/Concepts/Inference/KV-Cache|KV-Cache]] — проблема, которую решает MLA
- [[02 Areas/ML & DL/Concepts/NLP/Attention Mechanism|Attention Mechanism]] — базовый механизм

## Дополнительные ресурсы

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — полный отчёт (68 страниц)
- [TDS — Multi-head Latent Attention Explained](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4/) — визуальное объяснение MLA
- [Lior Sinai — DeepSeek's MLA](https://liorsinai.github.io/machine-learning/2025/02/22/mla.html) — математический разбор с выводами
- [DeepWiki — Architecture Overview](https://deepwiki.com/deepseek-ai/DeepSeek-V3/1.2-model-architecture-overview) — интерактивная визуализация
