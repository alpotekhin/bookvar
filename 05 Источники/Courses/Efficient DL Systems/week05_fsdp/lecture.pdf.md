---
title: "Week05 Fsdp — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week05_fsdp/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/lecture.pdf?raw=1" title="Week05 Fsdp — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
FSDP Lecture
```

### Page 2

```text
Contents
●   Motivation for developing another sharding method
●   FSDP basic implementation
●   Why is FSDP almost free?
     ○   NCCL Crash Course
     ○   Bytes transferred in FSDP vs DDP
     ○   Torch Profiler crash course
     ○   Forward and backward prefetching
●   Flavors of FSDP
     ○   FULL_SHARD
     ○   SHARD_GRAD_OP
     ○   HYBRID_SHARD
●   FSDP implementations and “how bro are they?”
```

### Page 3

```text
Modern DL model sizes
●   Most models are bigger than 7B
●   LLama family: 7B, 70B, 405B
```

### Page 4

```text
Mixed Precision recap
```

### Page 5

```text
Mixed Precision recap
Static memory consumption:
●   FP32 master weights and
    Adam moments
    (4 bytes x 3 buffers)
●   FP16 copy of weights for
    calculations (2 bytes)
●   FP16 gradients (2 bytes)
Dynamic memory consumption:
●   Activations (depends on
    batch size, etc)
```

### Page 6

```text
Mixed Precision recap
Static memory consumption:
 ●   FP32 master weights and
     Adam moments (4 bytes x 3
     3)
 ●   FP16 copy of weights for
     calculations (2 bytes)
 ●   FP16 gradients (2 bytes)
7B model = 7*10^9 * (4*3 + 2 +2)
bytes = 112GB
70B model = 70*10^9 * (4*3 + 2
+2) bytes = 1.1TB
```

### Page 7

```text
Pipeline parallel
each stage holds contiguous layers




   gpu 3, layer 3
   gpu 3, layer 3
   gpu 3, layer 3

   gpu 0, layer 0
```

### Page 8

```text
Pipeline parallel
●   Micro-batching reduces bubbles
●   Pipeline parallel is very good choice when it’s possible to use zero bubble
    strategies
```

### Page 9

```text
Pipeline parallel
●   Bubbles are hard to get rid of
●   Requires changing training
    code
```

### Page 10

```text
Tensor parallel
Parts of matmul are calculated across several GPUs and then combined
```

### Page 11

```text
Tensor parallel
●   Communication across
    GPUs can’t be overlapped
    with communication
```

### Page 12

```text
Tensor parallel
●   Communication across
    GPUs can’t be overlapped
    with communication
●   (Megatron example with
    1.2 rows)
    Complicated code changes
    are required
```

### Page 13

```text
So what?...
Notice how all GPUs store the same values in their memory in DDP
```

### Page 14

```text
So what?...
Infiniband fabric is idle during forward
```

### Page 15

```text
Lets load the weights right
before we need them.
```

### Page 16

```text
Fully Sharded Data Parallel
●   Separate model into
    parts (FSDP unit)
```

### Page 17

```text
Fully Sharded Data Parallel
●   Separate model into
    parts (FSDP unit)
●   For 32 GPU training
    each GPU stores:
    1/32 of Unit0
    1/32 of Unit1
    1/32 of Unit2
```

### Page 18

```text
Fully Sharded Data Parallel
●   Separate model into
    parts (FSDP unit)
●   For 32 GPU training
    each GPU stores
    1/32 of every unit.
●   On-demand gather
    16-bit parameters for
    forward or backward
```

### Page 19

```text
Fully Sharded Data Parallel
FlatParam has 2 states

●   sharded
●   unsharded
```

### Page 20

```text

```

### Page 21

```text
NCCL Crash course
●   torch.distributed give us collective communication primitives
●   torch.distributed can use several backends
●   NCCL is most popular backend for GPU training
```

### Page 22

```text
NCCL Crash course
●
```

### Page 23

```text
NCCL Crash course
●
```

### Page 24

```text
NCCL Crash course
```

### Page 25

```text
NCCL Crash course
```

### Page 26

```text
FSDP high-level overview
```

### Page 27

```text
FSDP (More details)

                            upcast grads to
                            32-bit (if not already)
●                           update fp32 FlatParam



 fp32 FlatParam copy into
 another 16-bit buffer
```

### Page 28

```text
FSDP for Llama 70B on 128 GPUs

FSDP unit: 1 transformer layer (0.9B parameters)

80 FlatParams: 70B * 4bytes / 128 = 2.1GB

80 Adam’s avg of grads: 70B * 4bytes / 128 = 2.1GB

80 Adam’s avg of grads^2: 70B * 4bytes / 128 = 2.1GB

1 buffer for fp16/bf16 weights: 0.9B * 2bytes = 1.8GB

1 buffer for fp16/bf16 gradients: 0.9B * 2bytes = 1.8GB
```

### Page 29

```text
FSDP for Llama 70B on 1280 GPUs

FSDP unit: 1 transformer layer (0.9B parameters)

80 FlatParams: 70B * 4bytes / 1280 = Basically 0

80 Adam’s avg of grads: 70B * 4bytes / 1280 = Almost 0

80 Adam’s avg of grads^2: 70B * 4bytes / 1280= Almost 0

1 buffer for fp16/bf16 weights: 0.9B * 2bytes = 1.8GB

1 buffer for fp16/bf16 gradients: 0.9B * 2bytes = 1.8GB
```

### Page 30

```text
FSDP for Llama 70B on 128 GPUs
```

### Page 31

```text
FSDP is “almost” free
```

### Page 32

```text
Tracing Crash
Course

Model=resnet18

bs=1024
```

### Page 33

```text
Tracing Crash Course



                       CPU




                       GPU
```

### Page 34

```text
Tracing Crash Course




                                CPU
                                optim
  CPU forward   CPU backward    step




                                 GPU
                                 optim
  GPU forward    GPU backward    step
```

### Page 35

```text
 Tracing Crash Course
CPU issues
AllGather kernel into
stream 32
```

### Page 36

```text
 Tracing Crash Course
CPU synchronizes
stream 7 with stream 32
(stream 7 will wait until
all currently scheduled
tasks in stream 32 end)


CPU issues
convolution kernel in
stream 7
```

### Page 37

```text
  Tracing Crash Course
CUDA streams are just
different queues of
kernels.

Kernels submitted to
one stream always
execute sequentially.
More on the matter:
https://developer.download.nvidia.com/CUDA/
training/StreamsAndConcurrencyWebinar.pdf
```

### Page 38

```text
Tracing Crash Course




                  1.All-Gather
                  finishes for
                  layer i
```

### Page 39

```text
Tracing Crash Course



                                 2.Conv may
                                 start for
                                 layer i




                  1.All-Gather
                  finishes for
                  layer i
```

### Page 40

```text
Tracing Crash Course



                                 2.Conv may
                                 start for layer
                                 i




                                     3.All-Gather
                  1.All-Gather       starts for
                  finishes for       layer i+i
                  layer i
```

### Page 41

```text
Tracing Crash Course
Communication is longer than computation!
GPU IS IDLE!!!
```

### Page 42

```text
Tracing Crash Course




                       1.All-Gather
                       finishes for
                       layer i+2
```

### Page 43

```text
Tracing Crash Course




                                      2. Layer k+1
                                      is still in
                                      progress




                       1.All-Gather
                       finishes for
                       layer k+2
```

### Page 44

```text
Tracing Crash Course




                                               2. Layer i+1
                                               is still in
                                               progress




                                      3.All-Gather
                       1.All-Gather   starts for
                       finishes for   layer i+3
                       layer i+2
```

### Page 45

```text
Tracing Crash Course
●   GPU computation is usually already highly optimized
    (it’s hard to do better than Tri Dao with FA3)
●   Communication is easier to tinker with
```

### Page 46

```text
Tracing Crash Course
●   FSDP allows full computation/communication overlap if All-Gather and
    Reduce-Scatter are fast enough
●   Communication speed can be influenced by
     ○   Hardware speed
     ○   Cluster topology
     ○   NCCL version
     ○   Model sharding strategy (We will focus on this one!)
```

### Page 47

```text
Prefetching
```

### Page 48

```text
Implicit forward prefetching
CPU thread runs “in
front” of GPU streams

All-Gather starts as
soon as it can
```

### Page 49

```text
Break implicit prefetching
Implicit prefetching breaks if CPU thread
waits for GPU

Simplest way to cause CPU-GPU sync is
to copy from GPU to CPU
```

### Page 50

```text
Explicit forward prefetching



●   On first forward records order of All-Gathers
●   Later, submits in pre-forward of Module All-Gather for next module too
```

### Page 51

```text
CPU syncs help limit memory consumption
Limit-all-gather=True

Limit number of
All-Gathers in flight.

Each All-Gather
requires memory.
```

### Page 52

```text
Explicit backward prefetching

Must use explicit `backward` prefetching or else there will be 0 overlap of
communication and computation.

All-Gather i -> Reduce-Scatter i-1 ->All-Gather i+1 -> Reduce-Scatter i

              Backward i                            -> Backward i+1
```

### Page 53

```text
Explicit backward prefetching

First pre-backward submits 2 All-Gathers
```

### Page 54

```text
FSDP sharding levels
From https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.ShardingStrategy
 ●   NO_SHARD (common DDP)
 ●   SHARD_GRAD_OP (no free 16bit-weights after forward)
 ●   FULL_SHARD (was explained in the lecture)
 ●   HYBRID_SHARD
```

### Page 55

```text
NO_SHARD
Recall:
All-Reduce=
Reduce-Scatter + All-Gather

Calculation:

Bytes transmitted in All-reduce =
sizeof(16-bit grads) + sizeof(16-bit grads)
```

### Page 56

```text
SHARD_GRAD_OP
Per forward of FSDP module:
1 All-Gather +
1 Reduce-Scatter




                              no All-Gather
```

### Page 57

```text
SHARD_GRAD_OP
Theoretically, no
communication overhead over
DDP!




                              no All-Gather
```

### Page 58

```text
FULL_SHARD

1 Additional
All-Gather
x1.5 communicated
bytes of NO_SHARD
```

### Page 59

```text
Deepspeed Crash Course
Deepspeed ZeRO is first implementation of weight sharding

●   Shards every nn.Parameter as its own module
●   (personal opinion) really hard to understand code and weird API
●   ZeRO stage 0 = NO_GRAD
●   ZeRO stage 1 = …
●   ZeRO stage 2 = SHARD_GRAD_OP
●   ZeRO stage 3 = FULL_SHARD
```

### Page 60

```text
  Deepspeed Crash Course




ZeRO1


ZeRO2



ZeRO3
```

### Page 61

```text
HYBRID_SHARD
Hybrid is between
NO_SHARD and
FULL_SHARD.
```

### Page 62

```text
HYBRID_SHARD

                     HYBRID_SHARD




               All-Reduce      All-Reduce
```

### Page 63

```text
HYBRID_SHARD
●   GPUs are organized into hierarchical structure
```

### Page 64

```text
HYBRID_SHARD
●   GPUs are organized into hierarchical structure




         2 hops
```

### Page 65

```text
HYBRID_SHARD
●   GPUs are organized into hierarchical structure



               4 hops
```

### Page 66

```text
HYBRID_SHARD
●   All-Gather and Reduce-Scatter inside FSDP group




        FSDP group   FSDP    FSDP                     FSDP
                     group   group
                                           …          group
```

### Page 67

```text
HYBRID_SHARD
●   All-Reduce across cluster-wide DDP groups

    1 (All GPUs from this group own identical weights)




         1       1      1      1       1        1        1   1
```

### Page 68

```text
HYBRID_SHARD
                                             HYBRID_SHARD
● FSDP group size is usually 256-512
  GPUs
● Number of DDP groups depends on
  size of cluster




                                       All-Reduce      All-Reduce
```

### Page 69

```text
FSDP 2 Crash Course

RFC(Request for comments):
https://github.com/pytorch/pytorch/issues/114299

●   Per-parameter sharding
●   Torch.Compile compatibility
```

### Page 70

```text
FSDP 2 Crash Course

●   Per-parameter sharding
●   In FSDP1 optimizer only saw
    part of FlatParam
●   Optimizer couldn’t treat
    different parameters in
    Module differently
```

### Page 71

```text
ФАЙЛ НЕЭФФЕКТИВНОСТИ

●   Каждый DataParallel ранг владеет
    частью fp32 весов
●   Перед forward и backward мы
    собираем bf16 версию весов
●   Как только посчитался градиент для
    всех весов в модуле, запускаем
    Reduce-Scatter
●   Каждый ранг получают градиенты
    для своей части весов и обновляет
    их
```
