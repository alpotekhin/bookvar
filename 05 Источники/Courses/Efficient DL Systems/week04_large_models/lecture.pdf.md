---
title: "Week04 Large Models — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week04_large_models/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week04_large_models/lecture.pdf?raw=1" title="Week04 Large Models — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
Model-Parallel Deep Learning
    Efficient DL, Episode ++i ‘26



  Yandex
  Research
```

### Page 2

```text
 Dealing with large models
Model-Parallel Deep Learning
    Efficient DL, Episode ++i ‘26



  Yandex
  Research
```

### Page 3

```text
      Recap: large models




Image Classification       Machine Translation
    ImageNet               average over WMT

   Source: https://arxiv.org/abs/1811.06965
```

### Page 4

```text
             Recap: Ring allreduce
Bonus quest: you can only send data between adjacent gpus



    GPU1               GPU2




    GPU3               GPU4


           Ring topology         Image: graphcore ipu server

         Answer & more: tinyurl.com/ring-allreduce-blog
```

### Page 5

```text
                          Recap: All-Reduce SGD
                                 arxiv.org/abs/1706.02677
          Idea: get rid of the host, each gpu runs its own computation
                 Q: why will weights be equal after such step?

              Data                                                Get full grads on       new θ are equal
Devices                                                            each device             across devices


GPU1      θ          x1   forward pass    backward pass     ∇θL(x1)      A       ∇θL(x)   step    θ
                                                                         L
                                                                         L
GPU2      θ          x2   forward pass    backward pass     ∇θL(x2)
                                                                         R
                                                                                 ∇θL(x)   step    θ
                                                                         E
                                                                         D
GPU3      θ          x3   forward pass    backward pass     ∇θL(x3)
                                                                         U
                                                                                 ∇θL(x)   step    θ
                                                                         C
          θ          x4   forward pass    backward pass     ∇θL(x4)      E       ∇θL(x)   step    θ
GPU4
```

### Page 6

```text
Q: What if a model is larger than GPU?
```

### Page 7

```text
 Q: What if a model is larger than GPU?
easy mode: cannot fit the right batch size
  hard mode: cannot fit a single sample
  expert mode: not even parameters!
```

### Page 8

```text
 Q: What if a model is larger than GPU?
easy mode: cannot fit the right batch size
  hard mode: cannot fit a single sample
  expert mode: not even parameters!

                   Ideas?
```

### Page 9

```text
          Q: What if a model is larger than GPU?
         easy mode: cannot fit the right batch size
           hard mode: cannot fit a single sample
           expert mode: not even parameters!

Solution: accumulate
grads from several
training batches
```

### Page 10

```text
 Q: What if a model is larger than GPU?
easy mode: cannot fit the right batch size
hard mode: cannot fit one training sample
  expert mode: not even parameters!
```

### Page 11

```text
Gradient checkpointing
           aka rematerialization




     Paper (DL): arxiv.org/pdf/1604.06174.pdf
 TF: github.com/cybertronai/gradient-checkpointing
  Pytorch: pytorch.org/docs/stable/checkpoint.html
```

### Page 12

```text
Gradient checkpointing
           Normal backprop




     Paper (DL): arxiv.org/pdf/1604.06174.pdf
 TF: github.com/cybertronai/gradient-checkpointing
  Pytorch: pytorch.org/docs/stable/checkpoint.html
```

### Page 13

```text
Gradient checkpointing
           Full rematerialization




     Paper (DL): arxiv.org/pdf/1604.06174.pdf
 TF: github.com/cybertronai/gradient-checkpointing
  Pytorch: pytorch.org/docs/stable/checkpoint.html
```

### Page 14

```text
Gradient checkpointing
             Single checkpoint




     Paper (DL): arxiv.org/pdf/1604.06174.pdf
 TF: github.com/cybertronai/gradient-checkpointing
  Pytorch: pytorch.org/docs/stable/checkpoint.html
```

### Page 15

```text
Gradient checkpointing
             Single checkpoint




     Paper (DL): arxiv.org/pdf/1604.06174.pdf
 TF: github.com/cybertronai/gradient-checkpointing
  Pytorch: pytorch.org/docs/stable/checkpoint.html
```

### Page 16

```text
Q: What if a model is larger than GPU?
  easy mode: cannot fit batch size 1
expert mode: not even parameters!

    You still have one GPU… (but not only a GPU)
```

### Page 17

```text
 Memory offloading
L2L: https://arxiv.org/abs/2002.05645




              ●
                Initialize all layers on CPU
              ●
                Move k layers at a time to GPU
              ●
                Remove layers after computation
              ●
                Fetch k+1-st layer while k-th runs
              ●
                Still 20-50% overhead
```

### Page 18

```text
 Memory offloading
L2L: https://arxiv.org/abs/2002.05645
```

### Page 19

```text
       Memory offloading
ZeRO-offload: https://arxiv.org/abs/2101.06840
```

### Page 20

```text
       Memory offloading
ZeRO-offload: https://arxiv.org/abs/2101.06840


●
  Offload in parallel with computation
●
  Use gradient checkpointing
●
  Delayed parameter update
```

### Page 21

```text
       Memory offloading
ZeRO-offload: https://arxiv.org/abs/2101.06840


●
  Offload in parallel with computation
●
  Use gradient checkpointing
●
  Delayed parameter update
```

### Page 22

```text
       Memory offloading
ZeRO-offload: https://arxiv.org/abs/2101.06840


●
  Offload in parallel with computation
●
  Use gradient checkpointing
●
  Delayed parameter update
```

### Page 23

```text
Q: What if a model is larger than GPU?
  easy mode: cannot fit batch size 1
expert mode: not even parameters!
    Can we do it better with
       multiple GPUs?
```

### Page 24

```text
 Model-parallel training

Q: What if a model is larger than GPU?
```

### Page 25

```text
 Model-parallel training

Q: What if a model is larger than GPU?




    model size: O(N)
                            Q: Can we go faster?
    throughput: O(1)
```

### Page 26

```text
                            Pipelining
     GPipe: arxiv.org/abs/1811.06965 – good starting point, not the 1st paper

Idea: split data into micro-batches and form a pipeline (right)




                  model size: O(n)
                  throughput: O(n) – with caveats
```

### Page 27

```text
                            Pipelining
     GPipe: arxiv.org/abs/1811.06965 – good starting point, not the 1st paper

Idea: split data into micro-batches and form a pipeline (right)




                  model size: O(n)
                                                                    Q: Even faster?
                  throughput: O(n) – with caveats
```

### Page 28

```text
                    Reducing the bubble
                          GPipe: arxiv.org/abs/1811.06965

GPipe:




                                                       … to be improved in a moment



         Note: backward takes longer than forward in practice

                  E.g. linear forward has one matmul,
                       backward has two matmuls (dW and dX)
```

### Page 29

```text
                Reducing the bubble
         1F1B pipeline from Megatron: https://arxiv.org/abs/2104.04473

GPipe:




                                                     Colocate forward with backward

1F1B:
```

### Page 30

```text
                           Reducing the bubble (further)
                    1F1B pipeline from Megatron: https://arxiv.org/abs/2104.04473

1F1B:




1F1B interleaved:
```

### Page 31

```text
                                      Reducing the bubble (furtherer)
                                ZB1P: “almost zero bubble” https://arxiv.org/abs/2401.10241




Idea: split backward into two ops:
- w.r.t inputs and w.r.t. weights


Grad w.r.t. weights doesn’t block
backward pass to prev stage
```

### Page 32

```text
   Reducing the bubble (furtherer yet)
Deepseek V1 schedule: https://arxiv.org/abs/2412.19437
```

### Page 33

```text
                       Asynchronous Pipelining
                              PipeDream: arxiv.org/abs/1806.03377

        Idea: apply gradients with every microbatch for maximum throughput


Also neat:
  ●
    Automatically partition
    layers to GPUs via
    dynamic programming
 ●
     Store k past weight
     versions to reduce
     gradient staleness
 ●
     Aims at high latency
```

### Page 34

```text
                            Pipelining Recap

When to use:
- model doesn’t fit on GPU; have multiple GPUs
- if model fits, but not the activations:          ???
- if model doesn’t fit, but you only have one GPU: ???

How to use:
  (just a moment...)
```

### Page 35

```text
                              Pipelining Recap

When to use:
- model doesn’t fit on GPU; have multiple GPUs
- if model fits, but not the activations: just do grad checkpointing!
- if model doesn’t fit, but you only have one GPU: offloading!

How to use:
- Basic implementation (GPipe): github.com/kakaobrain/torchgpipe
```

### Page 36

```text
                              Pipelining Recap

When to use:
- model doesn’t fit on GPU; have multiple GPUs
- if model fits, but not the activations: just do grad checkpointing!
- if model doesn’t fit, but you only have one GPU: offloading!

How to use:
- Basic implementation (GPipe): github.com/kakaobrain/torchgpipe
- PyTorch built-in: pytorch.org/tutorials/intermediate/pipelining_tutorial.html




    uses torch.distributed (torchrun) | supports GPipe, 1F1B, extendable!
```

### Page 37

```text
                              Pipelining Recap

When to use:
- model doesn’t fit on GPU; have multiple GPUs
- if model fits, but not the activations: just do grad checkpointing!
- if model doesn’t fit, but you only have one GPU: offloading!

How to use:
- Basic implementation (GPipe): github.com/kakaobrain/torchgpipe
- PyTorch built-in: pytorch.org/tutorials/intermediate/pipelining_tutorial.html
- DeepSpeed: https://deepspeed.readthedocs.io/en/latest/pipeline.html

Custom pipelines in many applications
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM (transformer-specific)
- Megablocks: https://github.com/databricks/megablocks (mixture-of-experts)
```

### Page 38

```text
                              Pipelining Recap

When to use:
- model doesn’t fit on GPU; have multiple GPUs
- if model fits, but not the activations: just do grad checkpointing!
- if model doesn’t fit, but you only have one GPU: offloading!

How to use:
- Basic implementation (GPipe): github.com/kakaobrain/torchgpipe
- PyTorch built-in: pytorch.org/tutorials/intermediate/pipelining_tutorial.html
- DeepSpeed: https://deepspeed.readthedocs.io/en/latest/pipeline.html

Problems:
- Bubbles = wasted compute time (duh)
- What if model layers aren’t symmetric? (e.g. LLM “head”, local attn, ViT pooling)
                       Balancing a pipeline is a world of hurt.
```

### Page 39

```text
          [short break]
How else can we run a large model
  over multiple GPUs / hosts?
```

### Page 40

```text
                            Tensor-parallel training
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks




                   See also: DP + TP https://arxiv.org/abs/1404.5997
```

### Page 41

```text
                 Tensor-parallel training
                                  Multiply by a        Partial   Scatter
Device   input   all-gather       part of matrix       product   -reduce



GPU1                          x   WEIGHT           =               ∑




GPU2                          x   MATRIX           =               ∑
```

### Page 42

```text
                 Tensor-parallel       training
                    Q: find AllReduce op here

                                  Multiply by a        Partial   Scatter
Device   input   all-gather       part of matrix       product   -reduce



GPU1                          x   WEIGHT           =               ∑




GPU2                          x   MATRIX           =               ∑
```

### Page 43

```text
                 Tensor-parallel       training
                    Q: find AllReduce op here

                                  Multiply by a        Partial   Scatter
Device   input   all-gather       part of matrix       product   -reduce



GPU1                          x   WEIGHT           =               ∑

                                                                  to next
                                                                  step ...
GPU2                          x   MATRIX           =               ∑
```

### Page 44

```text
                Tensor-parallel training
                    https://arxiv.org/pdf/2104.04473
    Mix and match parallelism directions to reduce synchronization

MLP: split over neurons                         Attention: split over heads
```

### Page 45

```text
                Tensor-parallel training
                    https://arxiv.org/pdf/2104.04473
    Mix and match parallelism directions to reduce synchronization

MLP: split over neurons                         Attention: split over heads
```

### Page 46

```text
     Sequence Parallelism
       https://arxiv.org/abs/2309.14509
Avoid storing the all activations on every device
```

### Page 47

```text
       [MOAR] Sequence Parallelism

Early mention of parallelism over sequences
https://arxiv.org/abs/2105.05720

DeepSpeed Ulysses – the method from previous slide
https://arxiv.org/abs/2309.14509

Ring Attention – compute attention dot / softmax in parallel
https://arxiv.org/abs/2310.01889

FLUX – overlap computation and communication with custom kernels
https://arxiv.org/abs/2406.06858
```

### Page 48

```text
Automated parallelism
source: https://sites.google.com/view/icml-2022-big-model
```

### Page 49

```text
Automated parallelism
source: https://sites.google.com/view/icml-2022-big-model
```

### Page 50

```text
Automated parallelism
source: https://sites.google.com/view/icml-2022-big-model
```

### Page 51

```text
Automated parallelism
source: https://sites.google.com/view/icml-2022-big-model
```

### Page 52

```text
Automated parallelism
source: https://sites.google.com/view/icml-2022-big-model
```

### Page 53

```text
          Automated parallelism
          source: https://sites.google.com/view/icml-2022-big-model




Compute graph                                               Device cluster
```

### Page 54

```text
     Automated parallelism
      source: https://sites.google.com/view/icml-2022-big-model



Q: How to partition the graph on the device cluster?
```

### Page 55

```text
             Automated parallelism
             source: https://sites.google.com/view/icml-2022-big-model




Strategy 1                                        Strategy 2




Strategy 3                                         Strategy 4
```

### Page 56

```text
             Automated parallelism
             source: https://sites.google.com/view/icml-2022-big-model



                                                                         Q: have you seen
                                                                         S1/2/3/4 before?

Strategy 1                                        Strategy 2




Strategy 3                                         Strategy 4
```

### Page 57

```text
              Automated parallelism
               source: https://sites.google.com/view/icml-2022-big-model




Pipeline MP                                         DP with offloading or PS




Tensor-parallel v1                                  Tensor-parallel
                                                     Strategy 4     v2
```

### Page 58

```text
              Automated parallelism
              source: https://sites.google.com/view/icml-2022-big-model




            Inter-op parallelism
Pipeline MP                     DP with offloading or PS




             Intra-op
Tensor-parallel v1    parallelism Strategy
                                 Tensor-parallel
                                           4     v2
```

### Page 59

```text
Automated parallelism
source: https://sites.google.com/view/icml-2022-big-model
```

### Page 60

```text
Automated parallelism
source: https://sites.google.com/view/icml-2022-big-model




  Q: how do we find the best strategy
      for partitioning the graph?
```

### Page 61

```text
                     RL-based partitioning
https://people.csail.mit.edu/hongzi/content/publications/placeto-neurips19.pdf
```

### Page 62

```text
Optimization-based partitioning
     https://arxiv.org/abs/2006.16423
```

### Page 63

```text
Alpa: optimization-based + reduced search space
             https://arxiv.org/abs/2201.12023
```

### Page 64

```text
Alpa: optimization-based + reduced search space
             https://arxiv.org/abs/2201.12023
```

### Page 65

```text
Alpa: optimization-based + reduced search space
               https://arxiv.org/abs/2201.12023




                   More details of each pass:
     https://sites.google.com/view/icml-2022-big-model
```

### Page 66

```text
Alpa: optimization-based + reduced search space
                             https://arxiv.org/abs/2201.12023
Not the first algorithm for auto-parallelism…
but the first one that is usable* (* - most of the time)        (benchmarks on AWS V100)
```

### Page 67

```text
Alpa: optimization-based + reduced search space
                             https://arxiv.org/abs/2201.12023
Not the first algorithm for auto-parallelism…
but the first one that is usable* (* - most of the time)



                            auto best strategy


                             works in jax
```

### Page 68

```text
Alpa: optimization-based + reduced search space
                             https://arxiv.org/abs/2201.12023
Not the first algorithm for auto-parallelism…
but the first one that is usable* (* - most of the time)


                      Alpa was deprecated in 2024, but successors exist
                            auto best strategy
                     Jax: use pjit/xmap with improved XLA optimizations

                  PyTorch: https://docs.pytorch.org/xla/master/spmd.html
                  … but forworks  in jax
                            standard  models, DeepSpeed is often enough.
```

### Page 69

```text
                                   </part 2>
+ model larger than GPU
+ faster for small
* typical size: 2-8 gpus
- model partitioning is tricky
   tensor parallelism is easier, but requires ultra low latency
- latency is critical, go buy nvlink
   except for PipeDream
- often combined with gradient checkpointing

Tutorials:
  ●
    Simple pipelining in PyTorch – tinyurl.com/pytorch-pipelining
  ●
    Distributed model-parallel with torch RPC - https://tinyurl.com/torch-rpc
  ●
    Minimalistic tensor parallelism pip install tensor_parallel
```

### Page 70

```text
                                   </part 2>
+ model larger than GPU
+ faster for small
* typical size: 2-8 gpus
- model partitioning is tricky
   tensor parallelism is easier, but requires ultra low latency
- latency is critical, go buy nvlink
   except for PipeDream
- often combined with gradient checkpointing

Tutorials:
  ●
    Simple pipelining in PyTorch – tinyurl.com/pytorch-pipelining
  ●
    Distributed model-parallel with torch RPC - https://tinyurl.com/torch-rpc
  ●
    Automatic tensor parallelism pip install tensor_parallel
        Q: what if you have 1024 GPUs, but the model fits on 8?
```

### Page 71

```text
                                   </part 2>
+ model larger than GPU
+ faster for small
* typical size: 2-8 gpus
- model partitioning is tricky
   tensor parallelism is easier, but requires ultra low latency
- latency is critical, go buy nvlink
   except for PipeDream
- often combined with gradient checkpointing

Tutorials:
  ●
    Simple pipelining in PyTorch – tinyurl.com/pytorch-pipelining
  ●
    Distributed model-parallel with torch RPC - https://tinyurl.com/torch-rpc
  ●
    Automatic tensor parallelism pip install tensor_parallel
        Large-scale training: combine model- and data-parallel
```

### Page 72

```text
So far we’ve been trying to partition for existing models…

  Perhaps there are models that are easier to partition?
```

### Page 73

```text
            Expert Parallelism
Sparsely gated MoE: https://arxiv.org/pdf/1701.06538.pdf
```

### Page 74

```text
MoE Variant: Switch Transformer
    Switch: https://arxiv.org/pdf/2101.03961.pdf
```

### Page 75

```text
MoE Variant: Switch Transformer
    Switch: https://arxiv.org/pdf/2101.03961.pdf
    MLM pre-training objective [BERT-like]
```

### Page 76

```text
MoE Variant: Switch Transformer
    Switch: https://arxiv.org/pdf/2101.03961.pdf
     Pre-training vs downstream quality
```

### Page 77

```text
Alternative: FSDP
  Source: microsoft
```

### Page 78

```text
    DeepSpeed Inference
    Paper: https://arxiv.org/abs/2207.00032


●
  Same techniques, but for inference
●
  Offloading, tensor- & pipeline-parallel
●
  … and a ton of hacks
```

### Page 79

```text
                                 </ZeRO>
Multi-GPU strategies:
* Pipeline model-parallel – allocate layers on different GPUs
* Sharded data-parallel – split optimizer state and/or parameters


Single GPU strategies:
* Small model – gradient checkpointing & virtual batch
* Large model – optimizer state sharding (keep parameters on GPU)


Implementations:
  ●
    DeepSpeed– sharded DP, offload, tensor parallelism, active development
     ●
       Offload – https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html
 ●
   FSDP – most of DeepSpeed features with native PyTorch API
 ●
   Model-specific implementations– https://github.com/NVIDIA/Megatron-LM
```

### Page 80

```text
If we have time…
    (if not, skip)
```

### Page 81

```text
                             </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation
                                                                ???

16GB model and optimizer, 16GB activations → DDP + gradient checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading   |   FSDP (ZeRO)   | Pipeline-parallel | Tensor-parallel
```

### Page 82

```text
                             </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) →???
                                                       grad checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading   |   FSDP (ZeRO)    | Pipeline-parallel | Tensor-parallel
```

### Page 83

```text
                             </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) → grad checkpointing

32GB model and optimizer, 1GB activations → it???
                                               depends…
```

### Page 84

```text
                               </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) → grad checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading   |   FSDP (ZeRO)      | Pipeline-parallel | Tensor-parallel
        ?


                       When is this the best option?
```

### Page 85

```text
                                       </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) → grad checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading         |   FSDP (ZeRO)      | Pipeline-parallel | Tensor-parallel
  e.g. if too few GPUs             ?
   for other methods

                             When is this the best option?
```

### Page 86

```text
                                        </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) → grad checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading         |    FSDP (ZeRO)          | Pipeline-parallel | Tensor-parallel
  e.g. if too few GPUs       no custom model code,          ?
   for other methods          best for large batches

                              When is this the best option?
```

### Page 87

```text
                                        </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) → grad checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading         |     FSDP (ZeRO)        | Pipeline-parallel | Tensor-parallel
  e.g. if too few GPUs       no custom model code, communication-efficient     ?
   for other methods          best for large batches sequential model

                               When is this the best option?
```

### Page 88

```text
                                        </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) → grad checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading         |     FSDP (ZeRO)        | Pipeline-parallel | Tensor-parallel
  e.g. if too few GPUs       no custom model code, communication-efficient     minimal latency
   for other methods          best for large batches sequential model        non-symmetric model

Mix and match: TP within one server, minimal PP between servers, DDP between groups
Parallel code: manual (e.g. Megatron-LM) vs automated (alpa, FSDP, tensor_parallel)
Unconventional hardware: hivemind, petals, varuna, etc
```

### Page 89

```text
                                    </lecture>
Example configuration:
Several GPU w/ 24GB memory | 128GB system memory | 16GBps interconnect

16GB model and optimizer, 128GB activations (batch 32) → grad accumulation

16GB model and optimizer, 16GB activations (batch 1) → grad checkpointing

32GB model and optimizer, 1GB activations → it depends…

DDP + offloading        |   FSDP (ZeRO)        | Pipeline-parallel | Tensor-parallel
  e.g. if too few GPUs   no custom model code, communication-efficient      minimal latency
   for If themethods
        other   model doesbest
                           notforfit, you
                                   large    can alsosequential
                                         batches      quantize   it into submission!
                                                               model      non-symmetric model
                  (more on model compression in a future lecture)
Mix and match: TP within one server, minimal PP between servers, DDP between groups
Parallel code: manual (e.g. Megatron-LM) vs automated (alpa, FSDP, tensor_parallel)
Unconventional hardware: hivemind, petals, varuna, etc
```

### Page 90

```text
</lecture>
```
