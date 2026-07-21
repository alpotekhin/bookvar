---
title: "Mixed Precision Training"
aliases: [Mixed Precision, AMP, FP16 training, BF16 training]
type: concept
status: legacy
category: Training
papers: []
courses: []
sources:
  - "[Micikevicius et al. — Mixed Precision Training (2017)](https://arxiv.org/abs/1710.03740)"
  - "[NVIDIA — Train With Mixed Precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)"
  - "[PyTorch AMP docs](https://pytorch.org/docs/stable/amp.html)"
---

# Mixed Precision Training — обучение в смешанной точности

## Зачем это нужно

Стандартное обучение в FP32 (32-бит float) избыточно точно для нейросетей: gradient signal устойчив к шуму, а веса редко требуют больше 3-4 значащих цифр. Переход на FP16/BF16 даёт:

- **2x экономия памяти** — критично для больших моделей и длинного контекста.
- **2-3x ускорение** — за счёт Tensor Cores (NVIDIA Volta+, AMD CDNA, TPU).
- **2x пропускная способность памяти** — узкое место многих операций.

**Mixed precision** означает: большинство операций (matmul, conv) — в FP16/BF16, но некоторые критичные (accumulator в matmul, master weights, loss) — в FP32.

## Форматы: FP32 vs FP16 vs BF16

| Формат | Бит | Exponent | Mantissa | Диапазон | Точность |
|--------|-----|----------|----------|----------|----------|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \cdot 10^{38}$ | ~7 знаков |
| FP16 | 16 | 5 | 10 | $\pm 6.5 \cdot 10^{4}$ | ~3 знака |
| BF16 | 16 | 8 | 7 | $\pm 3.4 \cdot 10^{38}$ | ~2-3 знака |
| TF32 | 19 | 8 | 10 | $\pm 3.4 \cdot 10^{38}$ | ~3 знака |

**FP16:** узкий диапазон ($\pm 65504$) — градиенты часто underflow (становятся 0). Требует **loss scaling**.

**BF16:** тот же экспонентный диапазон, что у FP32 — градиенты не underflow, loss scaling **не нужен**. Но точность ниже FP16. Стал стандартом для LLM-обучения (GPT, LLaMA, все открытые модели 2023+).

**TF32:** формат NVIDIA Ampere для Tensor Cores, используется автоматически для matmul, прозрачен для пользователя.

**Правило большого пальца:**
- Новое железо (A100, H100, H200) → **BF16**.
- Старое железо (V100, T4) → FP16 с loss scaling.
- Fine-tuning малых моделей → можно FP32.

## Loss Scaling (только для FP16)

Проблема FP16: градиенты часто лежат в диапазоне от $10^{-7}$ до $10^{-3}$. Значения $< 6 \cdot 10^{-8}$ становятся нулём — это **gradient underflow**, и обучение ломается.

**Решение:** умножить loss на большое число $S$ перед backward:

$$\text{loss}_{\text{scaled}} = S \cdot \text{loss}$$

По chain rule все градиенты умножаются на $S$ — сдвигаются в «рабочий» диапазон FP16. Перед шагом оптимизатора градиенты делятся на $S$ обратно (unscale).

### Static loss scaling

$S$ фиксирован (обычно $2^{10}$ – $2^{15}$). Простой, но нужно подбирать вручную.

### Dynamic loss scaling (стандарт)

$S$ меняется автоматически:
1. Начать с большого $S$ (например, $2^{16}$).
2. Если в градиентах появился `Inf`/`NaN` → пропустить шаг оптимизатора, уменьшить $S$ вдвое.
3. Если $N$ шагов прошло без overflow → увеличить $S$ вдвое.

В PyTorch это делает `torch.cuda.amp.GradScaler`:

```python
scaler = torch.cuda.amp.GradScaler()
for x, y in loader:
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

Для **BF16 loss scaling не нужен** — просто `autocast(dtype=torch.bfloat16)`.

## Master Weights в FP32

Даже при mixed precision **веса хранятся в FP32** (master copy). Почему:

- Обновление веса: $w \leftarrow w - \eta g$. Если $w \sim 1$, $\eta g \sim 10^{-7}$, то в FP16 это округлится до нуля — **update underflow**.
- Оптимизаторы Adam/AdamW используют EMA моментов — FP16 недостаточно для стабильного усреднения.

В forward/backward передаётся FP16/BF16 копия весов, а master FP32 обновляется в оптимизаторе. Это увеличивает память на веса (1.5x от FP16), но суммарно всё равно дешевле FP32 тренировки.

## Tensor Cores Utilization

Tensor Cores — специализированные блоки NVIDIA (Volta V100, Ampere A100, Hopper H100), выполняющие $4 \times 4$ или $8 \times 8$ matmul за один такт:

- **V100 FP16 Tensor Cores:** 125 TFLOPS (8x над FP32 CUDA cores)
- **A100 BF16 Tensor Cores:** 312 TFLOPS
- **H100 FP8 Tensor Cores:** 1978 TFLOPS

Чтобы Tensor Cores работали:

1. **Dimensions кратны 8** (а лучше 64/128). Hidden size 768 ✅, 750 ❌.
2. **Batch size кратен 8**.
3. **Типы данных** — FP16, BF16, TF32, FP8, INT8.

Неоптимальные размеры — главная причина, почему «включил AMP, а ускорения нет». Проверять профайлером (`nsys`, `torch.profiler`).

## FP8 и ниже (H100, H200, B200)

С Hopper появился **FP8** (E4M3 для forward, E5M2 для backward). Требует per-tensor scaling (как FP16 loss scaling, но для каждого тензора). Даёт ещё ~2x ускорение поверх BF16.

Используется в NVIDIA Transformer Engine. DeepSeek-V3 обучался в FP8 — первый крупный публичный пример. Дальше — FP4 (Blackwell).

## Практические выгоды

Для LLaMA-7B fine-tuning:
- FP32: ~80 GB VRAM, 1x скорость.
- BF16: ~40 GB VRAM, ~2.5x скорость.
- BF16 + FlashAttention: ~25 GB, ~4x скорость.
- FP8 (H100): ~20 GB, ~5x скорость.

## Когда mixed precision ломается

- **Softmax с очень большими логитами** — overflow в exp. Решение: вычислять softmax в FP32 (автоматически в PyTorch autocast).
- **Loss близок к нулю** — gradient underflow даже с loss scaling. Решение: BF16.
- **Reductions больших размеров** — накопление ошибки. Решение: FP32 accumulator (Tensor Cores делают это автоматически для matmul).
- **Layer norm / batch norm** — стабильнее в FP32, autocast это учитывает.

## Related concepts

- [[Gradient Clipping]] — должен применяться после unscale градиентов
- [[Distributed Training]] — mixed precision критичен для scale
- [[Scaling Laws]] — все большие модели обучаются в BF16/FP8
- [[LoRA]], [[PEFT]] — часто комбинируются с BF16 для экономии памяти
