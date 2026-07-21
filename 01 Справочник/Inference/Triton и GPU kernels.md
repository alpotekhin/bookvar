---
title: Triton и GPU kernels для LLM inference
type: concept
status: active
last_updated: 2026-07-21
last_verified: 2026-07-21
---

# Triton и GPU kernels для LLM inference

PyTorch выражает вычисление как операции над тензорами. GPU исполняет kernels —
программы, определяющие, какие threads читают какие адреса, где хранят
промежуточные значения и когда записывают результат. Между этими уровнями
возникает разрыв: композиция математически простых операций может несколько раз
прочитать и записать большой tensor в HBM, хотя промежуточное значение могло
остаться в registers или shared memory.

Triton — язык и compiler для написания специализированных GPU kernels на
Python-подобном уровне. Он не является serving engine и не конкурирует с vLLM
или SGLang: engines используют Triton наряду с CUDA, CUTLASS, FlashAttention,
FlashInfer и другими backends.

## Модель исполнения: программа обрабатывает tile

CUDA обычно начинает объяснение с отдельных threads. Triton предлагает мыслить
крупнее: grid состоит из program instances, каждая instance обрабатывает блок
данных. `tl.program_id(axis)` выбирает координату программы; `tl.arange`
создаёт offsets элементов; `tl.load` и `tl.store` работают с векторами адресов и
masks.

Для сложения векторов логика такова:

1. выбрать `BLOCK_SIZE`;
2. вычислить номер program instance;
3. получить диапазон offsets;
4. замаскировать выход за размер массива;
5. загрузить два блока, сложить и записать.

Compiler распределяет векторные операции по GPU threads. Автор по-прежнему
управляет tiling и memory access, но не прописывает каждую thread вручную.

## Почему fusion ускоряет softmax

Наивный softmax может состоять из max reduction, вычитания, exponentiation,
sum reduction и деления. Если framework материализует промежуточные tensors,
матрица несколько раз проходит через HBM. Fused kernel загружает строку один
раз, выполняет редукции и нормализацию на chip и записывает только результат.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/triton-fused-softmax-benchmark.png]]

*Сравнение fused Triton softmax с реализацией PyTorch на оборудовании авторов
tutorial. График нужно читать как проверку конкретного kernel и shape range, а
не универсальную гарантию ускорения. Источник: OpenAI Triton,
[Fused Softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).*

Выигрыш появляется не потому, что формула softmax изменилась, а потому что
уменьшилось движение данных. На слишком широких строках registers/shared memory
не хватает, occupancy падает, и стратегия должна измениться.

## Matmul: три измерения и память

Для $C=AB$ tile результата $C_{m:n}$ накапливается по блокам оси $K$. Kernel
многократно загружает tiles $A_{m:k}$ и $B_{k:n}$, вызывает `tl.dot`, хранит
accumulator в FP32, а перед записью может применить fused activation.

Главные параметры:

- `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`;
- число warps;
- порядок обхода tiles;
- dtype inputs и accumulator;
- число pipeline stages;
- hardware-specific launch configuration.

## Grouped ordering и L2 locality

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/triton-grouped-vs-row-major.png]]

*Слева row-major обход tiles быстро меняет строки A; справа grouped ordering
несколько раз использует близкие tiles и повышает вероятность попадания в L2.
Источник: OpenAI Triton,
[Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).*

Порядок математически независимых tiles влияет на реальную производительность.
Если соседние program instances повторно используют части A или B до их
вытеснения из L2, уменьшается traffic к HBM. Поэтому layout и scheduling — часть
алгоритма kernel, хотя в формуле матричного умножения их нет.

## Autotuning

Оптимальные block sizes зависят от shape, dtype и GPU. Triton позволяет описать
несколько configurations и выбрать быструю экспериментально для ключа вроде
$(M,N,K)$. Autotuning нельзя путать с обучением модели: это benchmark вариантов
kernel на целевом устройстве.

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/triton-matmul-benchmark-fp16.png]]

*FP16 matmul из официального tutorial: Triton сопоставляется с vendor BLAS на
наборе shapes. Источник: OpenAI Triton,
[Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).*

![[02 Areas/ML & DL/00 Учебник/Assets/Figures/curated/inference-serving/triton-matmul-benchmark-fp8.png]]

*Тот же tutorial отдельно показывает FP8: смена формата меняет доступные
инструкции, точность и сравнительный результат. Это не «бесплатное ускорение»,
а отдельная конфигурация kernel, которую нужно проверять на поддерживаемом GPU и
на допустимой для модели ошибке. Источник: OpenAI Triton,
[Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).*

Результат нельзя переносить на произвольную GPU. Версия compiler, clocks,
precision, shape distribution и warm-up должны быть зафиксированы. В production
важна не максимальная точка графика, а performance на реальных batch и sequence
shapes.

## Где Triton встречается в LLM runtime

- fused RMSNorm, activation и residual operations;
- quantization/dequantization;
- attention prefill/decode kernels;
- paged KV-cache access;
- sampling и logits processing;
- MoE routing, grouped GEMM и token permutation;
- linear attention/state-space scans;
- специализированные MLA/GQA layouts.

Не каждый kernel выгодно писать на Triton. CUDA/CUTLASS/CuTe могут дать лучший
контроль над новыми tensor cores; vendor libraries сильны на стандартных GEMM;
torch.compile способен автоматически fuse часть графа. Triton особенно полезен,
когда layout или fusion специфичны для модели и стандартная библиотечная
операция материализует лишние промежуточные данные.

## От kernel к serving engine

Уровни системы нужно держать раздельно:

| Уровень | Вопрос | Примеры |
|---|---|---|
| Kernel | как выполнить одну операцию и двигать данные | Triton, CUDA, CUTLASS |
| Attention backend | как реализовать prefill/decode для layout cache | FlashAttention, FlashInfer, Triton attention |
| Model executor | как запустить слои и коллективные обмены | vLLM/SGLang workers |
| Scheduler/cache | какие requests и blocks исполнять сейчас | continuous batching, PagedAttention, RadixAttention |
| Serving | как принимать, маршрутизировать и наблюдать запросы | API server, replicas, load balancer |

Быстрый softmax kernel не исправляет плохой scheduler; высокий cache hit rate не
компенсирует медленный MoE kernel. End-to-end performance появляется только при
согласовании уровней.

## Практический маршрут чтения

Официальные tutorials лучше проходить по возрастанию сложности:

1. Vector Addition — grid, offsets и mask;
2. Fused Softmax — reduction и memory traffic;
3. Matrix Multiplication — tiling, `tl.dot`, grouping и autotune;
4. LayerNorm — reductions и fusion;
5. Fused Attention — online softmax и причинная маска;
6. Group GEMM — shapes, характерные для MoE.

После каждого примера следует проверять correctness против PyTorch, а затем
benchmark на нескольких shapes. Измерение только одного удобного размера учит
подгонке графика, а не kernel engineering.

## Связанные страницы

- [[02 Areas/ML & DL/01 Справочник/Inference/vLLM — анатомия inference engine]]
- [[02 Areas/ML & DL/01 Справочник/Inference/SGLang и RadixAttention]]
- [[02 Areas/ML & DL/01 Справочник/Inference/PagedAttention]]
- [[02 Areas/ML & DL/01 Справочник/Inference/FlashAttention]]

## Источники

- [Triton documentation](https://triton-lang.org/main/).
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/).
- Tillet, Kung, Cox, [Triton: an intermediate language and compiler for tiled neural network computations](https://dl.acm.org/doi/10.1145/3315508.3329973), 2019.
- [vLLM Triton attention backend](https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/).
- [SGLang attention backend selection](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md).
