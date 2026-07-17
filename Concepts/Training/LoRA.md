---
title: "LoRA"
aliases: [LoRA, Low-Rank Adaptation, QLoRA]
type: concept
category: Training
papers:
  - "[[02 Areas/ML & DL/Papers/LoRA|LoRA]]"
  - "[[02 Areas/ML & DL/Papers/QLoRA|QLoRA]]"
courses: []
sources:
  - "[Sebastian Raschka — Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)"
  - "[Lightning AI — LoRA LLM](https://lightning.ai/pages/community/article/lora-llm/)"
  - "[Hugging Face — LoRA Course](https://huggingface.co/learn/llm-course/en/chapter11/4)"
---

# LoRA — Low-Rank Adaptation

## Зачем это нужно: проблема масштаба fine-tuning

Full fine-tuning GPT-3 175B означает обновление **175 миллиардов** параметров. Каждый параметр требует хранения в памяти: сам вес (fp16 = 2 байта), градиент (2 байта), два состояния оптимизатора Adam (по 4 байта) = **12 байт на параметр**. Для 175B это ~2 TB VRAM — недоступно для большинства организаций.

Даже если найти ресурсы, full fine-tuning создаёт отдельную 350 GB копию модели для каждой задачи. Хранить и переключаться между десятками таких копий непрактично.

LoRA (Hu et al., 2021, Microsoft) решает обе проблемы: **10,000x меньше обучаемых параметров**, **3x меньше VRAM**, и **нулевая дополнительная задержка** при инференсе. Де-факто стандарт parameter-efficient fine-tuning для LLM.

## Ключевая гипотеза: low intrinsic rank

Центральная идея LoRA строится на наблюдении Aghajanyan et al. (2020): **изменение весов при адаптации $\Delta W$ имеет низкий intrinsic rank**. То есть, хотя матрица весов $W$ может быть размером $d \times d$ (тысячи на тысячи), реальное «полезное» изменение при fine-tuning лежит в низкоранговом подпространстве.

Интуиция: при адаптации к конкретной задаче модели не нужно переучивать *все* представления — достаточно «повернуть» существующие в правильном направлении. Это «вращение» описывается малым числом степеней свободы.

## Как работает LoRA: пошаговый разбор

### Reparametrization

Вместо обновления полной матрицы весов $W_0 \in \mathbb{R}^{d \times k}$, LoRA параметризует изменение через **произведение двух маленьких матриц**:

$$h = W_0 x + \Delta W \cdot x = W_0 x + BA \cdot x$$

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig1.png]]
*LoRA reparametrization: замороженные веса $W$ (синий) + обучаемые низкоранговые матрицы $B$ и $A$ (оранжевый). При инференсе $BA$ сливается с $W$ — нулевой overhead (источник: Hu et al., 2021)*

Где:
- $W_0 \in \mathbb{R}^{d \times k}$ — замороженные pre-trained веса
- $B \in \mathbb{R}^{d \times r}$ — инициализируется **нулями**
- $A \in \mathbb{R}^{r \times k}$ — инициализируется **random Gaussian** $\mathcal{N}(0, \sigma^2)$
- $r \ll \min(d, k)$ — rank (типично 1-8)
- Scaling: $\Delta W \cdot x$ умножается на $\alpha / r$

**Почему именно такая инициализация?** $B = 0$ означает, что в начале обучения $\Delta W = BA = 0$ — модель стартует точно из pre-trained точки. Это critical: случайная инициализация обоих матриц дала бы случайный «толчок» всем весам, разрушив pre-trained представления.

### Какие слои адаптировать?

В оригинальной статье LoRA применяется к **attention матрицам** $W_q$ и $W_v$. Авторы провели ablation:

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig2.png]]
*Ablation: адаптация разных подмножеств attention матриц в GPT-3. Лучше адаптировать больше матриц с меньшим r, чем одну с большим r (источник: Hu et al., 2021)*

| Конфигурация | Параметры | Результат |
|-------------|-----------|-----------|
| $W_q$ only | 4.7M | Хуже baseline |
| $W_q, W_v$ | 4.7M | Лучше full FT |
| $W_q, W_k, W_v, W_o$ | 4.7M | Ещё лучше |
| Один слой с $r=64$ | 4.7M | Хуже, чем распределить r по слоям |

**Практическое правило:** распределять бюджет параметров по **большему числу матриц с меньшим r**, а не концентрировать в одном месте. $r = 4$ уже конкурентоспособен, $r = 8$ — sweet spot для большинства задач.

### Inference: нулевой overhead

Главное преимущество LoRA перед Adapters: при деплое $BA$ **сливается** с $W_0$:

$$W = W_0 + BA$$

Это вычисляется один раз. Результат — обычная матрица тех же размеров, что $W_0$. Никакой дополнительной латентности при инференсе.

**Переключение задач:** вычесть одну $BA$, прибавить другую $B'A'$ — мгновенно. Можно хранить десятки адаптеров по 35 MB каждый (vs 350 GB полный checkpoint).

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig5.png]]
*Сравнение inference latency: Adapter layers (AdapterL/AdapterH) добавляют 5-30% latency при batch\_size=1, LoRA — нулевой overhead (источник: Hu et al., 2021)*

## Результаты: LoRA vs Full Fine-Tuning

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig4.png]]
*Сравнение LoRA с другими методами PEFT на GPT-3 175B (источник: Hu et al., 2021)*

| Метрика | Full FT (GPT-3 175B) | LoRA (0.003% params) |
|---------|---------------------|---------------------|
| Обучаемые параметры | 175B | 4.7M |
| Checkpoint size | 350 GB | 35 MB |
| VRAM | ~1.2 TB (with Adam) | ~350 GB |
| Performance | Baseline | **Превосходит** на GPT-3 |

На RoBERTa-base: LoRA с 0.3M параметров превосходит full FT с 125M параметров. На GPT-3 175B: LoRA с 4.7M (0.003%) параметров **превосходит** full FT. Это контринтуитивно — меньше параметров, но лучше результат. Объяснение: low-rank constraint действует как регуляризатор, предотвращая overfitting.

## Выбор rank: сколько достаточно?

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/lora/fig3.png]]
*Влияние rank r на quality: даже r=1 конкурентоспособен, r=4-8 — оптимальный диапазон (источник: Hu et al., 2021)*

| Rank $r$ | Параметры (на слой) | Когда использовать |
|----------|--------------------|--------------------|
| 1 | Минимум | Простые задачи, classification |
| 4 | Стандарт | Большинство NLU задач |
| 8 | Sweet spot | Instruction tuning, сложные задачи |
| 16-64 | Много | Сложные domain-specific задачи |
| 256+ | Почти full FT | Только если LoRA с малым r не работает |

**Практический совет от Raschka:** начинать с $r = 8$, $\alpha = 16$. Если модель недообучается — увеличить $r$. Если переобучается — уменьшить $r$ или увеличить dropout.

## QLoRA: 4-bit квантизация + LoRA

QLoRA (Dettmers et al., 2023, UW) — следующий шаг: **квантизовать замороженные веса до 4 бит**, обучать только LoRA адаптеры в bf16. Результат: LLaMA 65B fine-tuning на **одной 48 GB GPU** (вместо >780 GB).

### Три инновации QLoRA

**1. 4-bit NormalFloat (NF4):**

Веса LLM приблизительно нормально распределены ($\sim \mathcal{N}(0, \sigma^2)$). NF4 — квантильное квантизование: уровни квантизации расставляются так, чтобы каждому «бину» соответствовало равное количество значений из $\mathcal{N}(0, 1)$. Это **информационно-оптимальный** тип данных для нормально распределённых весов. NF4 значительно превосходит FP4 и Int4 по quality/bit.

**2. Double Quantization (DQ):**

Стандартная квантизация хранит scaling constants для каждого блока — это 0.5 бит/параметр overhead. DQ квантизует **сами constants** (fp32 → fp8): overhead снижается до 0.127 бит/параметр. На 65B модели это ~3 GB экономии.

**3. Paged Optimizers:**

Optimizer states (Adam momentum и variance) выгружаются из GPU в CPU RAM через NVIDIA unified memory при нехватке VRAM. Автоматический page-in/page-out.

### QLoRA: формула forward pass

$$Y = X \cdot \text{doubleDequant}(W^{NF4}) + X \cdot L_1 \cdot L_2$$

Хранение в NF4 (4 бита), вычисления в BFloat16.

**Критически важно:** QLoRA требует LoRA на **всех** linear layers (не только $W_q, W_v$), чтобы match 16-bit full fine-tuning performance.

### Результат: Guanaco

- **Guanaco-65B:** 99.3% от ChatGPT на Vicuna benchmark при обучении на **одной 48 GB GPU** за 24 часа
- **Guanaco-7B:** вписывается в 5 GB VRAM — работает на смартфонах

## Когда использовать LoRA vs Full Fine-Tuning

| Критерий | Full FT | LoRA | QLoRA |
|---------|---------|------|-------|
| **Данных** | Много (>100K) | Мало-средне | Мало-средне |
| **GPU** | Multi-node кластер | 1-2 A100 | 1 consumer GPU |
| **Задача** | Максимальное quality | 95-100% от full FT | 90-99% от full FT |
| **Multi-task** | Отдельная копия на задачу | Лёгкие адаптеры | Лёгкие адаптеры |
| **Inference overhead** | Нет | Нет (merge) | Нет (merge) |
| **Training time** | Долго | Быстрее | Быстрее |

**Когда full FT лучше:** (1) задача сильно отличается от pre-training distribution, (2) огромный dataset, (3) нужен абсолютный максимум quality.

**Когда LoRA лучше:** (1) ограниченные ресурсы, (2) много задач/клиентов, (3) быстрые эксперименты, (4) production deployment с multiple adapters.

## Экосистема и adoption

LoRA стал де-факто стандартом fine-tuning в open-source LLM экосистеме:

- **HuggingFace PEFT** — реализация LoRA + QLoRA для всех моделей
- **Unsloth** — 2x ускорение LoRA обучения через kernel fusion
- **Alpaca-LoRA** — LLaMA 7B + LoRA за несколько часов на RTX 4090
- **PEFT adapters на HF Hub** — тысячи готовых адаптеров

QLoRA демократизировал fine-tuning 65B+ моделей на потребительских GPU. Вместе LoRA + QLoRA — основа экосистемы open-source LLM fine-tuning в 2023-2025.

## Related concepts

- [[02 Areas/ML & DL/Concepts/Training/PEFT|PEFT]] — семейство parameter-efficient методов
- [[02 Areas/ML & DL/Concepts/Training/Fine-tuning|Fine-tuning]] — umbrella статья о fine-tuning
- [[02 Areas/ML & DL/Concepts/Training/DPO|DPO]] — часто обучается через LoRA
- [[02 Areas/ML & DL/Concepts/Architectures/Transformer|Transformer]] — архитектура, к которой применяется LoRA
- [[02 Areas/ML & DL/Concepts/Architectures/LLaMA|LLaMA]] — главная модель для LoRA fine-tuning

## Дополнительные ресурсы

- [Sebastian Raschka — Practical Tips for LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) — лучшие практические советы
- [Lightning AI — LoRA LLM](https://lightning.ai/pages/community/article/lora-llm/) — пошаговый tutorial с кодом
- [Hugging Face — LoRA Course](https://huggingface.co/learn/llm-course/en/chapter11/4) — курс по PEFT
- [Unsloth — LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) — гайд по гиперпараметрам
