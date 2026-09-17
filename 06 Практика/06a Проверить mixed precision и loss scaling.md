---
title: Проверить mixed precision и loss scaling
type: practice
status: canonical
last_updated: 2026-09-15
---

# Проверить mixed precision и loss scaling

Смешанная точность может сократить передачу данных и ускорить матричные
операции. При этом малые градиенты иногда округляются до нуля, а большие
значения выходят за диапазон формата. В этой работе отдельно измеряются
скорость, память и численная ошибка, чтобы выбрать формат без потери
корректности обновления параметров.

## Исходные материалы

- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/lecture.pdf|Efficient DL Systems, Week 2 — полная оригинальная лекция]];
- [[02 Areas/ML & DL/05 Источники/Courses/Efficient DL Systems/week02_fast_pipelines/seminar/practice.ipynb|исходный seminar notebook]];
- [[02 Areas/ML & DL/00 Учебник/10 ML Systems/05 Численные форматы и mixed precision]].

## Экспериментальная матрица

На одной небольшой модели выполните одинаковый training step в режимах FP32,
TF32 (если применимо), FP16 с/без scaling и BF16. Зафиксируйте GPU, версии,
формат параметров, activations, gradients и accumulator.

Для каждого режима измерьте:

- loss и gradient norm по слоям;
- долю нулевых градиентов и значений `NaN`/`Inf`;
- peak allocated/reserved memory;
- step time после warmup и полезные samples/tokens per second;
- максимальную абсолютную и относительную ошибку против FP32 reference step.

## Сделать underflow наблюдаемым

У FP16 наименьшее положительное **нормальное** число равно $2^{-14}$, но это
не граница обнуления. Ниже него есть субнормальные числа вплоть до $2^{-24}$.
При округлении к ближайшему $2^{-26}$ уже превращается в ноль.
Начните с этих конкретных значений, а не с произвольного числа «ниже нормы».
Различие `smallest_normal` и `smallest_subnormal` описано в
[документации NumPy](https://numpy.org/doc/2.2/reference/generated/numpy.finfo.html).

Этот маленький CPU-пример показывает потерю градиента при записи в FP16.
Умножение потерь выполняется в FP32, иначе малый коэффициент мог бы потеряться
ещё до обратного прохода. Во втором опыте увеличивается величина, проходящая
через FP16, а исходный масштаб восстанавливается уже в FP32.

```python
# bookvar: fp16-underflow
import torch

values = torch.tensor([2**-14, 2**-20, 2**-24, 2**-26], dtype=torch.float32)
rounded = values.to(torch.float16).float()
print(rounded)

x = torch.tensor(1., dtype=torch.float16, requires_grad=True)
(x.float() * 2**-26).backward()
direct = x.grad.float()

y = torch.tensor(1., dtype=torch.float16, requires_grad=True)
scale = 2**16
(y.float() * 2**-26 * scale).backward()
recovered = y.grad.float() / scale
assert direct.item() == 0.0
assert recovered.item() == 2**-26
print("direct:", direct.item(), "scaled and recovered:", recovered.item())
```

Это демонстрация одного преобразования формата, не готовый рецепт AMP:
в обычном смешанном обучении параметры и их накопленные градиенты часто
остаются FP32. Некоторые аппаратные операции обнуляют субнормальные значения
(flush-to-zero), поэтому отдельно проверьте используемые GPU-ядра. Поведение
одного CPU-преобразования не гарантирует такое же поведение всех CUDA-операций.

В основной модели сравните FP16 без масштабирования, фиксированный масштаб
и динамический loss scaling. До шага оптимизатора градиенты нужно вернуть
в исходный масштаб; ограничение нормы выполняется после `unscale`.

Добавьте проверки:

```python
assert all_finite(unscaled_gradients)
assert relative_parameter_error(fp32_step, amp_step) < tolerance
assert optimizer_step_was_skipped_when_overflow_detected
```

Порог ошибки задайте до запуска и объясните его относительно масштаба весов.

## Проверить реальное ускорение

Profiler trace должен показать dtype матричных операций и используемые kernels.
Если маленькая модель остаётся launch-bound или pipeline ждёт CPU, отсутствие
ускорения — корректный результат. Не выводите hardware throughput из одного
`dtype` тензора.

## Что сдать

Полную конфигурацию среды, raw timings, таблицу численной ошибки, memory
snapshots, trace каждого режима и короткое объяснение выбранной политики
scaling. Работа закончена, если повторный запуск воспроизводит и ускорение (либо
его отсутствие), и специально созданный underflow/overflow case.
