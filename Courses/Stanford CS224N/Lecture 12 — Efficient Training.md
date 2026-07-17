---
title: "CS224N — Lecture 12: Efficient Neural Network Training"
course: "Stanford CS224N"
lecture: 12
type: course-note
raw: "[[02 Areas/ML & DL/raw/courses/Stanford CS224N/slides/cs224n-spr2024-lecture12-training-shikhar]]"
concepts: ["[[02 Areas/ML & DL/Concepts/Training/Mixed Precision Training|Mixed Precision]]", "[[02 Areas/ML & DL/Concepts/Training/Distributed Training|Distributed Training]]", "[[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]]", "[[02 Areas/ML & DL/Concepts/Training/Gradient Checkpointing|Gradient Checkpointing]]"]
---

# Lecture 12: Efficient Neural Network Training

> *"Hopefully useful for final projects!"* -- Shikhar Murty

Лектор: Shikhar Murty.

## Floating Point форматы

### Устройство чисел с плавающей точкой

Число = sign (1 бит) + exponent ($e$ бит) + mantissa ($m$ бит). Exponent определяет **range** (диапазон), mantissa -- **precision** (точность).

$$\text{value} = (-1)^s \times 2^E \times (1 + M)$$

### Три ключевых формата

| Формат | Exponent | Mantissa | Байты | Range | Precision |
|--------|----------|----------|-------|-------|-----------|
| **FP32** | 8 бит | 23 бита | 4 | $\sim 10^{-38}$ до $10^{38}$ | Высокая |
| **FP16** | 5 бит | 10 бит | 2 | $\sim 10^{-14}$ до $10^{15}$ | Средняя |
| **BF16** | 8 бит | 7 бит | 2 | $\sim 10^{-38}$ до $10^{38}$ | Низкая |

**FP16**: маленький range -> gradient underflow (малые градиенты становятся ровно 0). Маленькая precision -> rounding errors (1.0001 округляется до 1.0).

**BF16**: тот же range, что FP32 (8 бит exponent), но меньше precision. **Не нужен** gradient scaling!

## [[02 Areas/ML & DL/Concepts/Training/Mixed Precision Training|Mixed Precision Training]]

### Проблема

FP32 -- стандарт для обучения, но 4 байта на параметр. Большие модели не помещаются в VRAM.

Наивное решение -- FP16 -- имеет два проблемы:
1. **Gradient underflow**: малые градиенты обнуляются
2. **Imprecise weight updates**: $w + \Delta w \approx w$ при малом $\Delta w$

### Рецепт Mixed Precision (Micikevicius et al., 2018)

1. **Master weights** в FP32 (основная копия параметров)
2. **Forward pass** в FP16/BF16
3. **Scale loss** на большое значение (artificially увеличить градиенты)
4. **Backward pass** в FP16/BF16 (градиенты не underflow благодаря scaling)
5. **Copy gradient** в FP32, **поделить на scale factor**
6. **Update master weights** в FP32 (точное обновление!)
7. **Copy** обратно в FP16/BF16

### Результат

- ~2x ускорение (FP16 Tensor Cores)
- ~2x экономия памяти на forward/backward
- **Минимальная потеря качества** (доказано эмпирически)

### BF16 vs FP16

С BF16 **не нужен GradScaler** -- тот же range, что FP32, поэтому gradient underflow не происходит. На практике:

```python
# BF16 -- проще
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
loss.backward()
optimizer.step()
```

**Рекомендация из лекции**: "Always use BFloat16 if `torch.cuda.is_bf16_supported()`"

## [[02 Areas/ML & DL/Concepts/Training/Distributed Training|Multi-GPU Training]]

### Что хранится на GPU VRAM?

- Model parameters (FP16): 2 байта/параметр
- Backward pass gradients (FP16): 2 байта/параметр
- Master weights (FP32): 4 байта/параметр
- Adam momentum (FP32): 4 байта/параметр
- Adam variance (FP32): 4 байта/параметр
- **Model activations** (масштабируется с batch size!)

**Итого**: ~16 байт на параметр + activations. Для 7B модели: ~112 GB только для параметров + optimizer.

### DDP (Distributed Data Parallel)

Каждый GPU имеет **полную копию** модели.

1. **Forward**: каждый GPU обрабатывает свой mini-batch параллельно
2. **Backward**: вычисление градиентов + **All-Reduce** коммуникация
3. **All-Reduce**: каждый GPU получает **усреднённый** градиент со всех GPU
4. **Update**: одинаковое обновление на всех GPU → модели остаются синхронизированными

**Communication overhead**: 2 байта на параметр (градиенты в FP16).

**Проблема DDP**: каждый GPU хранит **полную** копию модели + optimizer states. Память **не масштабируется** с числом GPU.

### ZeRO: шардирование для экономии памяти

**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al. 2020) -- три стадии шардирования:

#### ZeRO Stage-1 (Optimizer State Sharding)

Каждый GPU хранит **шард** optimizer states (master weights + momentum + variance). Forward/backward -- с полными параметрами. После backward: **reduce-scatter** градиентов, каждый GPU обновляет свой шард, **all-gather** для синхронизации.

**Communication**: 2 байта/параметр (как DDP). **Память сэкономлена бесплатно!**

#### ZeRO Stage-2 (+ Gradient Sharding)

Дополнительно шардируем **градиенты**: никогда не материализуем полный gradient vector. При backward: как только gradient для шарда вычислен, **отправляем** на ответственный GPU и **освобождаем** память.

**Communication**: 2 байта/параметр. **Ещё больше памяти сэкономлено -- бесплатно!**

#### ZeRO Stage-3 (Full FSDP) -- Fully Sharded Data Parallel

Шардируем **всё**: parameters + gradients + optimizer states.

1. Делим параметры модели на FSDP units
2. Шардируем каждый unit по GPU
3. **Forward**: all-gather параметров текущего слоя, forward pass, **discard** non-local шарды
4. **Backward**: all-gather, backward, reduce-scatter градиентов
5. Каждый GPU обновляет свой шард

**Communication overhead**:
- DDP: All-reduce
- ZeRO 1/2: Reduce-scatter + All-gather (бесплатно!)
- ZeRO 3: All-gather + Reduce-scatter + **All-gather** (дополнительный overhead!)

### Decision tree из лекции

```
Always: Mixed Precision + BF16 if supported
              │
     Does batch_size=1 fit on 1 GPU?
         ├── Yes: Try larger batch / ZeRO Stage-2
         └── No: Does ZeRO Stage-3 solve OOM?
                  ├── Yes: Use it
                  └── No: Parameter-Efficient Finetuning!
```

## [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] (Low-Rank Adaptation, Hu et al. 2022)

### Мотивация

Full finetuning всех параметров:
- $|\Delta\Phi| = |\Phi_0|$ -- для GPT-3 это 175 **миллиардов** параметров
- Дорого хранить и деплоить отдельную копию для каждой задачи
- Ключевое наблюдение: updates к весам имеют **low intrinsic rank** (Aghajanyan et al. 2020)

### Идея

Вместо обновления полной матрицы $W_0 \in \mathbb{R}^{d \times k}$, параметризуем update через **low-rank** разложение:

$$W = W_0 + \alpha \cdot BA$$

где $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

### Свойства

- **Обучаемые параметры**: только $A$ и $B$ (~0.1-1% от полного числа)
- **Инициализация**: $A$ -- random Gaussian, $B$ -- zeros (при старте $BA = 0$)
- **$\alpha$**: tradeoff между pretrained knowledge и task-specific adaptation
- **No inference latency**: при деплое $BA$ сливается в $W_0$. Переключение между задачами: вычесть $BA$, прибавить $B'A'$
- **Сходимость**: при увеличении $r$ → приближение к full finetuning

### Где применять LoRA

Обычно к weight matrices в [[02 Areas/ML & DL/Concepts/NLP/Self-Attention|self-attention]] ($W_Q$, $W_K$, $W_V$, $W_O$). Иногда и к FFN. Типичные значения $r$: 4-64.

### Экологический аспект

Рост compute для крупнейших AI-моделей: ~2x каждые 3.4 месяца (vs глобальный compute: ~2x каждые 1.5 года). **Неустойчивый** темп. Parameter-efficient finetuning снижает barrikade для входа и углеродный след.

Пример из лекции: в курсе CS234 (Stanford, RL), 200+ студентов выполняли homework. Если бы все использовали более эффективный алгоритм, коллективное энергопотребление снизилось бы на **880 kWh** -- месячное потребление типичного американского домохозяйства.

## Concepts covered

- [[02 Areas/ML & DL/Concepts/Training/Mixed Precision Training|Mixed Precision]] -- FP32 master weights + FP16/BF16 forward/backward, loss scaling
- [[02 Areas/ML & DL/Concepts/Training/Distributed Training|Distributed Training]] -- DDP, ZeRO Stages 1/2/3 (FSDP), шардирование параметров/градиентов/optimizer states
- [[02 Areas/ML & DL/Concepts/Training/LoRA|LoRA]] -- Low-Rank Adaptation, $W + \alpha BA$ с $r \ll \min(d, k)$
- [[02 Areas/ML & DL/Concepts/Training/Gradient Checkpointing|Gradient Checkpointing]] -- tradeoff memory vs compute для activations
