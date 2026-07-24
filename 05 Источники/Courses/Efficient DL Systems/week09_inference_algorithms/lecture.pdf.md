---
title: "Week09 Inference Algorithms — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week09_inference_algorithms/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/lecture.pdf?raw=1" title="Week09 Inference Algorithms — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
     LLM inference speedup
     E DL




     by Roman Gorb

ff
```

### Page 2

```text
Кто я такой?
```

### Page 3

```text
Кто я такой?
Роман Горб

• YandexGPT (~5 лет)
• R&D-лаба Huawei (1 год)
• Закончил ФизТех, ШАД
• Призер ВСОШ по математике и
  информатике

• разряд по волейболу
• собираю железяки =)
```

### Page 4

```text
Agenda

 Motivation      Speculative Decoding
 Tradeo          Quantization
 LLM inference   Knowledge Distillation




  ff
```

### Page 5

```text
Motivation
```

### Page 6

```text
Scenarios
Models:                    Tasks:              Stages:

• Decoder-only, e.g GPT    • Generative:       • Context processing
• Enc-Dec, e.g. UL2          • Q&A             • Decoding
• Small vs large             • Dialogs

     • Compute vs memory     • Summary
      bound                • Discriminative:
     • OOM or not            • Classi cation
• Batch size                 • Regression
• SFT vs PEFT                • NER
fi
```

### Page 7

```text
GPU
Architecture and limirations

• Core types: CUDA vs
     Tensor

• Memory transfer and
     compute

• VRAM is 10x slower
     than SM

• VRAM=80GB,
     SM=228KB


* https://habr.com/en/companies/yandex/articles/672396/
```

### Page 8

```text
Production
Compute budget
• RPS — The number of HTTP-style requests completed per second
• Latency — The amount of time (in microseconds) taken for a single request to complete
```

### Page 9

```text
   Пример
   Hands-on case

   2K rps                   <3
• Service has a load of 2k rps
• 20% could be cache hit
                                                 1           6 rps
    service load         sec per request —       GPU
• The other 1600 rps on latency
                         avg can    become
                                 limit
1800 at peak latency limit < 3 sec per request
• 1 GPU provides 6 rps, latency < 3 sec
    20%                                          300              200
• So service demand is 300 GPU                   GPU is service   GPU is computational
• Computational
     cache hit    budget is 200 GPU              demand           budget
• Ooops…

   1,6K (1,8K) rps                               Oops
   on avg       (at peak)
                                                                                         26
```

### Page 10

```text
Tradeoff
```

### Page 11

```text
             Pareto curve
             Quality vs Compute

             • Latency or RPS for x axis                        01



             • Quality metric for y axis                        00



             • In ection point is optimal                      –0 1



             • Should consider compute      Relative quality
                                                               –0 2

               budget conditions
                                                               –0 3


                                                               –0 4


                                                                      1000   2000      3000       4000       5000   6000   7000

                                                                                    Generation time median, ms


        fl
    .
    .
    .
    .
.
.
```

### Page 12

```text
Challenge


   How to add new points?
```

### Page 13

```text
LLM inference
```

### Page 14

```text
Classi cation
or context processing

• Attention with squared mask
• 128k tokens
• 1 forward
• GeMM
• Compute bound




      fi
```

### Page 15

```text
Generation
autoregressive

• Triangle mask
• 1 forward per token
• GeMV
• Loading weights on
  each step

• Memory bound
```

### Page 16

```text
Road to compute bound
Size matters

• batch_size=N:
  • Encoder/context: no change
  • Decoder: matrix x N vectors
  • => For large N similar to GeMM
• Small model size:
  • => not bounded memory transfer
  • => compute bound
```

### Page 17

```text
Encoder-decoder
breaking the rules

• 1 forward (enc) for context
• Cross-Attention
• N forwards (dec) per token      Encoder   Decoder


• Easy strategy to reduce cost:
  • 70% to encoder
  • 30% to decoder
```

### Page 18

```text
Road to compute bound
Architecture tweaks
```

### Page 19

```text
MoE                                            Для DS v3:
проблемы

• Дизбаланс нагрузки по экспертам
• Реальный bs внутри эксперта на decode:
 • bs_expert = bs * active / total = bs / 16
• Хотим быть compute bound
 • т.е. прогружать тензоркоры
• Для этого нужен bs_expert >= 128
 • => bs >= 128 * 16 = 2 048
```

### Page 20

```text
Self-attention
VRAM demand

• На 1 пример =
       2 * 4 (kv_heads) *

       128 (head_size) *
                                 BF16:
       94 (num_layers) *

       17000 (seqlen) *
                             =   3.05GB
       2 (bytes_precision)
```

### Page 21

```text
Потребление VRAM
максимальный батч


• Weights + KV cache + scratches <= GPU vram
• 8xh200: 470 + bs * 3.05 + 20 <= 1128, значит bs <= 209   << 2048

• 8xh100: 470 + bs * 3.05 + 20 <= 640, значит bs <= 49     << 2048



                              💀💀💀
```

### Page 22

```text
Нужно ужиматься по VRAM!
```

### Page 23

```text
Quantization
```

### Page 24

```text
Quantization
idea
```

### Page 25

```text
     Quantization
     types

     • weight-only (wNa16)
       • reduce memory footprint
       • speedup memory bound
     • wNaM
       • also reduce compute
       • speedup compute bound     In ight



fl
```

### Page 26

```text
Stages
and objectives

• Quant: preserve
  benchmarks quality

• Backend support: gain
  speedup

• Could be separated
• But highly interconnected
```

### Page 27

```text
Approaches
types

Post-Training Quantization           Quantization-Aware Training

• No data/~1000 calibration samples • Huge dataset (like for pretrain)
• ~1–10 GPU hours                    • ~FT computational budget
• Calibrating s, z                   • ~100–1000 GPUs go brrr
• min–max, running min–max, MSE • Training s, z, weights using STE
• No weights training           • Forget less -> better quality
• Useful for downstream
```

### Page 28

```text
wNa16
```

### Page 29

```text
GPT-Q
Contribution summary

• W4A16
• 3,25x (1,5x–2x actually)
• decoding only
• Code and CUDA kernels
  published

• Highly popular in open
  source

• LLaMa.cpp CPU
  inference
```

### Page 30

```text
GPT-Q
Benchmarks
```

### Page 31

```text
wNaM
```

### Page 32

```text
Challenges
```

### Page 33

```text
Challenge
Quality drop

• Started from BERT
• Only weights — easy
• Activations — hard
• Outliers — the problem
```

### Page 34

```text
LLM.int8()
Outliers

• Causes:
  • not size
  • low perps
• exponential growth
```

### Page 35

```text
Impact
on top-1 softmax
```

### Page 36

```text
SmoothQuant
```

### Page 37

```text
SmoothQuant
Intuition



• Hard to quant activations
• Migrate some difficulty to weights
```

### Page 38

```text
SmoothQuant
Motivation and method



• W8A8
• Apply to all BMM
• Fuse scaling with prev
  operations
• Per-channel scale is
  accurate but not efficient
• Good s ???
```

### Page 39

```text
SmoothQuant
Scale tradeoff and choice



• 1 — hard W easy A
• 2 — easy W hard A
• 3 — ok W ok A
• Alpha grid search


1.

2.

3.
```

### Page 40

```text
SmoothQuant
Benchmarks
```

### Page 41

```text
SmoothQuant
Memory and latency speedups
```

### Page 42

```text
FP8
```

### Page 43

```text
FP8 Quantization
overview




                   https://arxiv.org/pdf/2309.14592
```

### Page 44

```text
FP8 Quantization
bene ts

• Simple static amax-scaling
      quantization

• Several formats (E5M2, E4M3)
• Per-tensor quantization of W/A
• Works better and faster than other
      quantizations, i.e. SmoothQuant

• x2 compression for weights

                                        https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/
                                                 source/blogs/quantization-in-TRT-LLM.md
 fi
```

### Page 45

```text
FP8 attention & kv cache
impact

• Compress kv cache x2
  • 3.05GB -> 1.52GB
• x2 bs (~x2 rps in practice)
• Flash3 in FP8 precision
• x1.25+ pre ll for 32к seqlen
• almost lossless (<0.5pp)


      fi
```

### Page 46

```text
SpinQuant
```

### Page 47

```text
SpinQuant
rotation idea
```

### Page 48

```text
SpinQuant
rotation and outliers

• removes outliers
• stable&low kurtosis



• stable&low quant error
```

### Page 49

```text
SpinQuant
method

• w4a4kv4
• smooth with
  rotation

• learnable rotation
• fuse rotation to W
  if possible

• else: Hadamard
  rotations
```

### Page 50

```text
SpinQuant
learnable rotations

• init with Hadamard
• minimize quant err
• Cayley SGD
• better variance than
  random Hadamard
```

### Page 51

```text
SpinQuant
results

• ~3 pp loss on w4a4kv4
• best for 70B+
• great speedup
```

### Page 52

```text
FP4
```

### Page 53

```text
FP4
variants
```

### Page 54

```text
Hadamard transform
impact on MXFP4/NVFP4
```

### Page 55

```text
     Benchmarks
     quality

     • NO lossless format
          • -2 p.p.
     • NVFP4/NVINT4
     • MR-GPTQ
     • HT:
          • e ective for INT4
          • not e ective for NVFP4
     ff
ff
```

### Page 56

```text
Speculative Decoding
```

### Page 57

```text
Speculative Decoding
idea
```

### Page 58

```text
Speculative Decoding
idea

 Decoding

   1 forward per token
```

### Page 59

```text
Speculative Decoding
idea

 Decoding

   1 forward per token

 Speculative Decoding

  1. Small model generates K
     tokens
```

### Page 60

```text
Speculative Decoding
idea

 Decoding

   1 forward per token

 Speculative Decoding

  1. Small model generates K
     tokens

  2. Large model veri es




            fi
```

### Page 61

```text
Speculative Decoding
Formal part
```

### Page 62

```text
Speculative Decoding
speedup

     Decoding:

       1 forward per each token

     Speculative Decoding:

       1 veri er’s forward per k tokens + k drafter’s forwards

     Implications:

       less forwards of large model

       ~same cost for forwards on 1 and k tokens

fi
```

### Page 63

```text
Speculative Decoding
Quality

• Полноценная приёмка это
 дорого

• Преимущество SpecDec:
 • Математически – генерации
   те же

 • Погрешности как при смене
   batch_size
```

### Page 64

```text
EAGLE
method

• Drafter is a head instead of model
• Frozen base model’s body🧊
• High AccRate
• Head – 1-layer transformer over
  hiddens of base model

• Tree of hypotheses
```

### Page 65

```text
EAGLE
why

 First method for large batch_size

 Low overhead:

   lightweight head (1/48)

 High acc. rate:

   hidden of model

   sequential mode

                                     https://github.com/hemingkx/Spec-Bench/blob/main/
                                                       Leaderboard.md
```

### Page 66

```text
Tree draft
eagle-2
```

### Page 67

```text
EAGLE 2 vs 3
problem
```

### Page 68

```text
EAGLE 2 vs 3
solution

• “deep” loss on tokens
• feature fusion
• no feature modeling loss
• data scaling
```

### Page 69

```text
KV cache compression
```

### Page 70

```text
Self-attention
“врожденные” травмы

• Arithmetics:
  • O(seqlen) per step
  • O(seqlen^2) in total
• VRAM demand:
  • O(seqlen)
• Memory bound
• Bad scaling
```

### Page 71

```text
      Token-level
      overview

      • Challenges:
        • small bs
        • high latency
      • reduce #tokens in attn kernel
      • o oad KV cache to RAM/SSD/NET




ffl
```

### Page 72

```text
Model-level
overview

• GQA
• DeepSeek MLA/DSA
• Qwen GDN
```

### Page 73

```text
     System-level
     overview

     • Pre x-sharing
       • agentic speci c
     • Load balancing
     • Cascades
     • DPD
     • Heterogeneous clusters


             fi
fi
```

### Page 74

```text
DSA
общеизвестный подорожник 1

• MLA + sparse
• I_t,s + top-k
• Learnable selection:
  • init with dense v3.1
  • warmup indexer


  • sparse train
```

### Page 75

```text
Линейный attention
общеизвестный подорожник 2

• Qwen Next / 3.5:
  • GatedDeltaNet
• 3 GDN + 1 full attn
  • => нужен подорожник 1
• Есть скрытые проблемы
  • CPU overhead (FLA)
  • SpecDec / KV reuse
```

### Page 76

```text
GDN
overview

• NO attention map
• SSM
• 4 streams:
  • conv
  • sigmoid
• GatedDeltaRule:
```

### Page 77

```text
Knowledge Distillation
```

### Page 78

```text
     Knowledge Distillation
     General framework

     • Teacher p(y|x): usually a LLM, e.g. GPT-3 (175B)
       • achieves SOTA quality
       • don’t t inference computational budget
     • Student q(y|x): small LM, e.g. T5 XL (3B)
       • unable to rich teacher’s quality by ordinary training
       • ts inference computational budget
     • Knowledge Distillation (KD): process of teaching the student to imitate teacher’s performance




     fi
fi
```

### Page 79

```text
Hard-Label KD
Idea

• Sample targets from teacher
• SFT on that pairs
• Reproducing full sequences
```

### Page 80

```text
Hard-Label KD
RLCE
```

### Page 81

```text
Soft-Label KD
Idea

• Same as Hard-label, but
  reproduce logits also
• Sampling from teacher by default
• Dirty hack:
  • use targets from original
    dataset
  • compute logits in parallel
```

### Page 82

```text
Soft-Label KD
SLIM

• 128k vocab size, too large to store even in distributed storage
• utilize top-5% logits for e ciency




                    ffi
```

### Page 83

```text
     KL KD
     De nition and Idea

     • “Distance” between distributions
     • Monte-Сarlo estimation




fi
```

### Page 84

```text
KL KD
Issue

• Student’s distribution should
  cover teacher’s distribution
• While naturally student is less
  expressive than teacher
```

### Page 85

```text
Reverse KLD
Solution

• Swap arguments of KL
• Student approx. only the top
  probs
• Entropy regularization
```

### Page 86

```text
Reverse KLD
Solution

• Swap arguments of KL
• Student approx. only the top
  probs
• Entropy regularization
• Another problem occurs!
• Unable to di erentiate by
  sampled y




       ff
```

### Page 87

```text
Speculative KD
simple approach

• student generates
• teacher monitors
• teacher’s top-k
• no backprop
  through sampling
```

### Page 88

```text
Thank you!


             gorb-roman@yandex-team.ru   YandexGPT
```
