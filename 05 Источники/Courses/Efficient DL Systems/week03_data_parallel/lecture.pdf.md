---
title: "Week03 Data Parallel — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week03_data_parallel/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf?raw=1" title="Week03 Data Parallel — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
   Data-Parallel Deep Learning
       Efficient DL, Episode 3, 2026



Yandex
Research
```

### Page 2

```text
             Зачем это всё?
XGBoost training time                        parameters vs time
                             Model size trend up to 2022




       nthreads




                    (single GPU – over 2 weeks)
```

### Page 3

```text
             Зачем это всё?
XGBoost training time                        parameters vs time
                             Model size trend up to 2022
                             As of ’26: trillions of params,
                                       1-2 oom more flops




       nthreads




                    (single GPU – over 2 weeks)
```

### Page 4

```text
              Зачем мы тут?

Заставить много железяк вместе учить одну модель
```

### Page 5

```text
              Зачем мы тут?

Заставить много железяк вместе учить одну модель


               понять общие подходы

               закодить своими руками

                 на python / pytorch
```

### Page 6

```text
               TL;DR our plan
                     next few lectures

- Data-parallel deep learning
  Train BERT-base on wikipedia in 20 minutes or less


- Model-parallel deep learning
  Fine-tune and deploy models with 100B+ parameters
                  like OPT, Llama, Qwen, DeepSeek R1, ...

- Advanced techniques
  Sharding (FSDP), mixed / hybrid parallelism, practice
```

### Page 7

```text
               TL;DR our plan
                     next few lectures

- Data-parallel deep learning
  Train BERT-base on wikipedia in 20 minutes or less


- Model-parallel deep learning
  Fine-tune and deploy models with 100B+ parameters
                  like OPT, Llama, Qwen, DeepSeek R1, ...

- Advanced techniques
  Sharding (FSDP), mixed / hybrid parallelism, practice
```

### Page 8

```text
               TL;DR our plan
                     next few lectures

- Data-parallel deep learning
  Train BERT-base on wikipedia in 20 minutes or less


- Model-parallel deep learning
  Fine-tune and deploy models with 100B+ parameters
                  like OPT, Llama, Qwen, DeepSeek R1, ...

- Advanced techniques
  Sharding (FSDP), mixed / hybrid parallelism, practice
```

### Page 9

```text
              Rules: Process
     foo       bar         baz         qux   ...


Process:
●
  Runs some code
●
  Has some memory
●
  No one else can access your memory
```

### Page 10

```text
              Rules: Process
     foo       bar         baz         qux   ...


Process:
●
  Runs some code
●
  Has some memory
●
  No one else can access your memory
```

### Page 11

```text
                    Rules: Process
       foo           bar           baz   qux   ...


Process:
●
  Runs some code
●
  Has some memory
●
  No one else can access your memory*
* – not if you use shared memory
```

### Page 12

```text
                      Rules: Process
       foo            bar               baz      qux   ...


Process:
●
  Runs some code
●
  Has some memory
●
  No one else can access your memory*†
* – not if you use shared memory
†
  – superuser can still do that (os-dependent)
```

### Page 13

```text
                     Rules: Process
       foo            bar             baz               qux   ...


Process:
●
  Runs some code
●
  Has some memory
●
  No one else can access your memory*†‡
* – not if you use shared memory
†
  – superuser can still do that (os-dependent)
‡
  – attacker can do that through spectre/meltdown/etc
```

### Page 14

```text
                       Rules: Process
        foo            bar           baz   qux   ...


Process:
●
  Runs some code
●
  Has some memory
●
  No one else should access your memory*†‡
*†‡ – not relevant for this course
```

### Page 15

```text
                       Rules: Process
        foo            bar           baz         qux   ...


Process:
●
  Runs some code
●
  Has some memory
●
  No one else should access your memory*†‡
*†‡ – not relevant for this course

          Q: How do we make processes work together?
```

### Page 16

```text
             Rules: Channel / Pipe
Process A:

    foo       bar         baz         qux   result



Process B:

              func       otherfunc


 Channel (pipe):
 ●
   Communication in O(message size)
 ●
   Asynchronous read/write
```

### Page 17

```text
                    Data-parallel training (naive)
                     cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf

Host
           model
CPU
             θ

          .cuda()
Devices

GPU1         θ
```

### Page 18

```text
                    Data-parallel training (naive)
                      cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf

Host
           model    batch
CPU                                               ...
             θ         x

          .cuda()   .cuda()
Devices

GPU1         θ         x        forward pass        backward pass         ∇θL(x1)
```

### Page 19

```text
                    Data-parallel training (naive)
                      cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf

Host
           model    batch
CPU                                               ...            prepare next batch
             θ         x

          .cuda()   .cuda()
Devices                                                                                    new model

GPU1         θ         x        forward pass        backward pass         ∇θL(x1)   step      θ
```

### Page 20

```text
                      Data-parallel training (naive)
                       cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf

Host
           model
CPU
             θ

          replicate
Devices

GPU1         θ

GPU2         θ

GPU3         θ

GPU4         θ
```

### Page 21

```text
                       Data-parallel training (naive)
                          cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf

Host
           model         batch
CPU
             θ        x1 x2   x3      x4

          replicate     scatter
Devices

GPU1         θ                   x1


GPU2         θ                   x2


GPU3         θ                   x3


GPU4         θ                   x4
```

### Page 22

```text
                       Data-parallel training (naive)
                          cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf

Host
           model         batch
CPU                                                       ...
             θ        x1 x2   x3      x4

          replicate     scatter
Devices

GPU1         θ                   x1        forward pass   backward pass       ∇θL(x1)


GPU2         θ                   x2        forward pass   backward pass       ∇θL(x2)


GPU3         θ                   x3        forward pass   backward pass       ∇θL(x3)


GPU4         θ                   x4        forward pass   backward pass       ∇θL(x4)
```

### Page 23

```text
                       Data-parallel training (naive)
                          cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf

Host
           model         batch                                                          full grad          new model
CPU                                                       ...
             θ        x1 x2   x3      x4                                                ∇θL(x)      step      θ

          replicate     scatter                                                         reduce (sum)       replicate
Devices

GPU1         θ                   x1        forward pass   backward pass       ∇θL(x1)                         θ

GPU2         θ                   x2        forward pass   backward pass       ∇θL(x2)                         θ

GPU3         θ                   x3        forward pass   backward pass       ∇θL(x3)                         θ

GPU4         θ                   x4        forward pass   backward pass       ∇θL(x4)                         θ
```

### Page 24

```text
                          All-Reduce data parallel
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

### Page 25

```text
                            Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors



Devices

GPU1      ∇θL(x1)                            A              ∇θL(x)
                                             L
                                             L
GPU2      ∇θL(x2)
                                             R
                                                            ∇θL(x)
                                             E
          ∇θL(x3)                            D              ∇θL(x)
GPU3                                         U
                                             C
          ∇θL(x4)                            E              ∇θL(x)
GPU4
```

### Page 26

```text
                            Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors


                             Naive implementation
Devices

GPU1      ∇θL(x1)                          Sum              ∇θL(x)


GPU2      ∇θL(x2)                                           ∇θL(x)


GPU3      ∇θL(x3)                                           ∇θL(x)


GPU4      ∇θL(x4)                                           ∇θL(x)
```

### Page 27

```text
                            Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors


                              Q: Can we do better?
Devices

GPU1      ∇θL(x1)                          Sum              ∇θL(x)


GPU2      ∇θL(x2)                                           ∇θL(x)


GPU3      ∇θL(x3)                                           ∇θL(x)


GPU4      ∇θL(x4)                                           ∇θL(x)
```

### Page 28

```text
                            Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors


                                 Tree-allreduce
Devices

GPU1      ∇θL(x1)       Sum          Sum             ∇θL(x)   ∇θL(x)


GPU2      ∇θL(x2)                                             ∇θL(x)


GPU3      ∇θL(x3)       Sum                          ∇θL(x)   ∇θL(x)


GPU4      ∇θL(x4)                                             ∇θL(x)
```

### Page 29

```text
                                   Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors


                Butterfly-allreduce – split data into chunks (ABCD)
Devices
                    Part A
GPU1      ∇θL(x1)            Sum     A


GPU2      ∇θL(x2)


GPU3      ∇θL(x3)


GPU4      ∇θL(x4)
```

### Page 30

```text
                                   Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors


                Butterfly-allreduce – split data into chunks (ABCD)
Devices
                    Part A
GPU1      ∇θL(x1)            Sum     A


GPU2      ∇θL(x2)            Sum     B


GPU3      ∇θL(x3)            Sum     C


GPU4      ∇θL(x4)            Sum     D
```

### Page 31

```text
                                   Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors


                Butterfly-allreduce – split data into chunks (ABCD)
Devices
                    Part A
GPU1      ∇θL(x1)            Sum     A    ∇θL(x)
                                           concat   ∇θL(x)


GPU2      ∇θL(x2)            Sum     B


GPU3      ∇θL(x3)            Sum     C


GPU4      ∇θL(x4)            Sum     D
```

### Page 32

```text
                                   Faster allreduce
            Input: each device has its its own vector
            Output: each device gets a sum of all vectors


                    Ring-allreduce – split data into chunks (ABCD)
Devices
                    Part A
GPU1      ∇θL(x1)            Sum     A     concat   ∇θL(x)


GPU2      ∇θL(x2)            Sum     B     concat   ∇θL(x)


GPU3      ∇θL(x3)            Sum     C     concat   ∇θL(x)


GPU4      ∇θL(x4)            Sum     D     concat   ∇θL(x)
```

### Page 33

```text
                    Ring allreduce
Bonus quest: you can only send data between adjacent gpus



    GPU1               GPU2




    GPU3               GPU4


           Ring topology        Image: graphcore IPU server

         Answer & more: tinyurl.com/ring-allreduce-blog
```

### Page 34

```text
                  Ring allreduce
Bonus quest: you can only send data between adjacent gpus




                 [Time to use the whiteboard]




         Answer & more: tinyurl.com/ring-allreduce-blog
```

### Page 35

```text
                          All-Reduce data parallel
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

### Page 36

```text
                           </Data-parallel>

+ easy to implement
+ can scale to 100s of gpus
+ can be fault-tolerant
- model must fit in 1 gpu
- large batches aren’t always
   good for generalization
●
  2-4 GPUs & no time – naive data parallel tinyurl.com/torch-data-parallel
●
  4+ GPUs or multiple hosts – distributed (allreduce) github.com/horovod/horovod
  ●
    Intro to pytorch distributed: tinyurl.com/distributed-dp or in 15 minutes!
●
  Somewhat faulty GPU/network: synchronous data parallel + drop stragglers
●
  Very faulty or uneven resources: asynchronous data parallel (more later)
●
  Efficient training with large batches: LAMB https://arxiv.org/abs/1904.00962
●
  Dynamically adding or removing resources: https://tinyurl.com/torch-elastic
```

### Page 37

```text
          Decentralized training vs real-world tasks
                                      arxiv.org/abs/1706.02677
                     Each gpu has different processing time & delays
                         Q: can we improve device utilization?

              Data
Devices

GPU1      θ          x1       forward pass          backward pass                         ∇θL(x1)   A
                                                                                                    L
                                                                                                    L
GPU2      θ          x2     forward       backward                                        ∇θL(x2)
                                                                                                    R
                                                                                                    E
                                                                                                    D
GPU3      θ          x3         forward pass              backward pass                   ∇θL(x3)
                                                                                                    U
                                                                                                    C
                          network delay                                                             E
GPU4      θ                                    x4        forward pass     backward pass   ∇θL(x4)
```

### Page 38

```text
                                 Recap: Parameter Server
                                          HOGWILD! arxiv.org/abs/1106.5730

            Idea: remove synchronization step alltogether, use parameter server

                                                 Model parameters

Devices

GPU1                         θ      Train step        ∇θL                θ          Train step     ∇θL


GPU2      ...       Train step      ∇θL           θ         Train step       ∇θL            θ              ...
                                                                                                   Train step


GPU3            Train step        ...        Train step         ...            Train step        ...




          Problem: parameter servers need to ingest tons of data over training
```

### Page 39

```text
Decentralized Training with Gossip
    Gossip (communication): https://tinyurl.com/boyd-gossip-2006
Gossip outperforms All-Reduce: https://tinyurl.com/can-dsgd-outperform
```

### Page 40

```text
Decentralized Training with Gossip
     Source: https://tinyurl.com/can-dsgd-outperform
```

### Page 41

```text
Stochastic Gradient Push
  Source: https://arxiv.org/abs/1811.10792
```

### Page 42

```text
Stochastic Gradient Push
  Source: https://arxiv.org/abs/1811.10792




                                         <to be continued>
```

### Page 43

```text
Stochastic Gradient Push
  Source: https://arxiv.org/abs/1811.10792




                                         normal GD step

                                         <to be continued>
```

### Page 44

```text
Stochastic Gradient Push
  Source: https://arxiv.org/abs/1811.10792




                                         <to be continued>
```

### Page 45

```text
Stochastic Gradient Push
  Source: https://arxiv.org/abs/1811.10792




                                             weighted
                                             average
```

### Page 46

```text
Stochastic Gradient Push
     Source: https://arxiv.org/abs/1811.10792


SGP vs ImageNet (ResNet50 + SGD w/ momentum)
```

### Page 47

```text
Stochastic Gradient Push
      Source: https://arxiv.org/abs/1811.10792


SGP vs WMT English-German (Transformer, Adam)
```

### Page 48

```text
Gossip vs All-Reduce




 Your thoughts?
```

### Page 49

```text
      Gossip + All-Reduce
          Source: arxiv.org/abs/2005.00124


 Core idea: run all-reduce in independent groups
You only have to synchronize for your small group
      Swap groupmates between iterations
```

### Page 50

```text
      Gossip + All-Reduce
          Source: arxiv.org/abs/2005.00124


 Core idea: run all-reduce in independent groups
You only have to synchronize for your small group
      Swap groupmates between iterations
```

### Page 51

```text
                        Gossip + All-Reduce
                         Source: arxiv.org/abs/2005.00124



                        Experiment setup: up to 1024 GPU,
                        Natural (or emulated) network latency




 Image Classification           Machine Translation             Reinforcement Learning
ResNet50 @ ImageNet            Transformer @ WMT17                 DDPO on Habitat
```

### Page 52

```text
Q: what if sending tensors during
   AllReduce takes too long?
```

### Page 53

```text
                      Quantized communication
                               https://arxiv.org/abs/1511.04561

TL;DR
- send data in 8-bit
- all computations in 32-bit
- choose best data format


PROFIT: same quality as float16
```

### Page 54

```text
Can we compress further?
  without losing quality
```

### Page 55

```text
                   Error Feedback + PowerSGD
                     https://arxiv.org/abs/1901.09847 - EF theory
                     https://arxiv.org/abs/1905.13727 - PowerSGD
       TL;DR - use extreme compression, e.g. 1-bit or top-5% gradients
                 - if you lose something in compression, reuse it on the next step

     compute
ϴ
    local grad
                                             time flows left to right
        g
```

### Page 56

```text
                   Error Feedback + PowerSGD
                     https://arxiv.org/abs/1901.09847 - EF theory
                     https://arxiv.org/abs/1905.13727 - PowerSGD
       TL;DR - use extreme compression, e.g. 1-bit or top-5% gradients
                 - if you lose something in compression, reuse it on the next step

     compute
ϴ
    local grad
                                             time flows left to right
        g




     Compress!      C(g)

                        Use C(g) in AllReduce or Gossip
```

### Page 57

```text
                   Error Feedback + PowerSGD
                     https://arxiv.org/abs/1901.09847 - EF theory
                     https://arxiv.org/abs/1905.13727 - PowerSGD
       TL;DR - use extreme compression, e.g. 1-bit or top-5% gradients
                 - if you lose something in compression, reuse it on the next step

     compute
ϴ
    local grad
                                     compression
        g                               error

                                       e=g - C(g)
                                             C(g)

     Compress!      C(g)
```

### Page 58

```text
                   Error Feedback + PowerSGD
                     https://arxiv.org/abs/1901.09847 - EF theory
                     https://arxiv.org/abs/1905.13727 - PowerSGD
       TL;DR - use extreme compression, e.g. 1-bit or top-5% gradients
                 - if you lose something in compression, reuse it on the next step

     compute                                                           next
ϴ
    local grad                                              ϴ       local grad
                                     compression
        g                               error                         g+e

                                       e=g - C(g)
                                             C(g)

     Compress!      C(g)                             “Feed” the error            C(g)
                                                      back to grads!
```

### Page 59

```text
                    Error Feedback + PowerSGD
                      https://arxiv.org/abs/1901.09847 - EF theory
                      https://arxiv.org/abs/1905.13727 - PowerSGD
         TL;DR - use extreme compression, e.g. 1-bit or top-5% gradients
                  - if you lose something in compression, reuse it on the next step

      compute                                                           next
ϴ
     local grad                                              ϴ       local grad
                                      compression
       g+e                               error                         g+e

                                        e=g - C(g)
                                              C(g)
Error from the                                        “Feed” the error
previous step!       C(g)                                                         C(g)
                                                       back to grads!
```

### Page 60

```text
Error Feedback + PowerSGD
 https://arxiv.org/abs/1901.09847 - EF theory
 https://arxiv.org/abs/1905.13727 - PowerSGD
```

### Page 61

```text
PowerSGD: low-rank approx grads + Error Feedback
            https://arxiv.org/abs/1901.09847 - EF theory
            https://arxiv.org/abs/1905.13727 - PowerSGD
```

### Page 62

```text
                 Read More: gradient compression
https://arxiv.org/abs/1901.09847 - EF theory   https://arxiv.org/abs/2106.05203 - better EF’21

https://arxiv.org/abs/1905.13727 - PowerSGD    https://arxiv.org/abs/2110.03294 - more EF’21
```

### Page 63

```text

```

### Page 64

```text
   Summary: operation parallelism

Data-parallel:    ???


Model-parallel:   ???
```

### Page 65

```text
   Summary: operation parallelism

Data-parallel:   one process applies all model on partial data
                  best for smaller model, more computations
Model-parallel: one process applies partial model on all data
                  best for larger model, fewer computations

                    Which one is better..
                              for ResNet50?
                              for Llama 70B?
                              In general?
```

### Page 66

```text
   Summary: operation parallelism

Data-parallel:   one process applies all model on partial data
                  best for smaller model, more computations
Model-parallel: one process applies partial model on all data
                  best for larger model, fewer computations

                    Which one is better..
                              for ResNet50?      It depends…
                              In Llama 70B?      - on model size
                                                 - on compute
```
