---
title: "Week01 Intro — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week01_intro/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week01_intro/lecture.pdf?raw=1" title="Week01 Intro — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
Efficient Deep Learning Systems
       Course introduction
           Max Ryabinin




                2026
```

### Page 2

```text
                  Why are we here?
• DL as a field is getting mature:
  • Neural networks are becoming more and more widespread in practice
  • Scaling trends everywhere (model size, dataset size, coauthor list size)
  • .ipynb-based development is no longer viable even for basic stuﬀ :)
• Training and running large models is necessary for frontier-level tasks
• Engineering knowledge is now essential for SOTA research
• For practical applications, performance and maintainability are key factors
```

### Page 3

```text
  Bird's eye view of DL

      Training                                  Inference
How to achieve the best quality?                 Is my model useful?
                                              Is my model good enough?

  How to run the experiments
                                    Is performance good enough for my use case?
       quickly enough?


         Do I utilize my
                                   How do I ensure the model can serve the demand?
    resources to the fullest?

     How many resources
  do I need in the first place?    …How do I not run out of money in the process? :)
```

### Page 4

```text
                     Goal of the course
• Most DL courses do not cover practical details and overall systems:
  •   Small code changes can make your training/inference much faster

  •   Deployment of trained networks, both on their own and as a part of a larger system

  •   Streamlined maintenance by treating ML models like any other code (testing, versioning, etc.)


• Knowledge about this is scattered around the Internet and unstructured
• We want to give you these useful bits of practical knowledge!
• …no bleeding-edge methods or last-week papers (with some exceptions)
```

### Page 5

```text
                                               Plan
1. (You are here) Intro, basics of GPU architecture & benchmarking

2. Profiling DL pipelines, techniques for eﬃcient training                  Training pipelines and performance

3. Data-parallel training, All-Reduce, torch.distributed intro

4. Memory-eﬃcient training, model parallelism
                                                                            Large-scale training
5. Sharded data parallel training, optimizations for distributed training

6. Large-scale training arithmetics

7. Basics of web service deployment

8. Inference optimizations: systems                                         Deployment in production

9. Inference optimizations: algorithms
```

### Page 6

```text
                              Logistics

• Lectures&seminars: every Wednesday, 18:00 – 21:00, via Zoom

• Course repo: github.com/mryab/eﬃcient-dl-systems

• YSDA LMS for submitting assignments

• Channel with announcements: see HSE FCS wiki/course page in LMS

• Resources: Yandex Cloud VM + DataSphere (HSE), YSDA GPUs + DataSphere (YSDA)
```

### Page 7

```text
                                Grading
• 3 assignments:
   1. Fast pipelines (1 part)

   2. Large-scale training (4 parts)

   3. Deployment (3 parts)



• Each assignment consists of sub-assignments given each week (except this one)

• Final grade: Gtotal = 0.1G1 + 0.5G2 + 0.4G3
```

### Page 8

```text
GPU architecture: a brief overview
• As the name suggests, originally used for graphics
• Highly parallel execution model: objects can be rendered simultaneously
• Since ~2007, simple GPGPU API started to appear (CUDA, OpenCL, Metal)
• GPU-trained AlexNet/DanNet sparked the DL revolution in early 2010s




             3dfx Voodoo2: 12MB RAM               NVIDIA GB200 NVL72: 186GB RAM
```

### Page 9

```text
GPU architecture: a brief overview




docs.nvidia.com/cuda/cuda-c-programming-guide
```

### Page 10

```text
            CUDA computation model
   • In CUDA, we launch kernels from the host
     that are executed in parallel on the device

   • Kernels are executed by threads grouped in
     thread blocks of limited size

   • A GPU is composed of multithreaded
     Streaming Multiprocessors (SMs)
     that are assigned diﬀerent thread blocks

   • Multiple thread blocks are arranged in grids
     (can be 1D, 2D, or 3D)



developer.nvidia.com/blog/cuda-refresher-cuda-programming-model
```

### Page 11

```text
   GPU computations: hardware side
   • SIMT (Single Instruction, Multiple Thread)
   • On a physical level, threads are executed in
      groups of 32 called warps

   • A warp executes one instruction at a time:
      in case of branching, all paths need to be taken

   • This does not aﬀect correctness but has major
      performance implications

   • Warp-level primitives can be leveraged for
      parallel computation


developer.nvidia.com/blog/using-cuda-warp-level-primitives
```

### Page 12

```text
      Why does all of this matter?
• The most popular operation in DL is matrix
  multiplication

• Executing this in parallel can have two
  potential eﬀects when dividing the work

• Tile Quantization: matrix size is not
  divisible by the thread block tile size




docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
```

### Page 13

```text
      Why does all of this matter?
• The most popular operation in DL is matrix
  multiplication

• Executing this in parallel can have two
  potential eﬀects when dividing the work

• Tile Quantization: matrix size is not
  divisible by the thread block tile size

• Wave Quantization: total number of tiles is
  quantized to the number of SMs

• Both eﬀects can be quite noticeable for
  small or irregular shapes!


docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
```

### Page 14

```text
                                  Memory access
      • GPU has a separate memory unit (called
         device memory)

      • Need to copy from host memory and back
         (PCIe 4.0 x16 — 32GB/s peak)

      • Memory transfer is often a bottleneck
      • Pinned (page-locked) memory access is
         much faster




https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
```

### Page 15

```text
                                  Memory access
      • GPU has a separate memory unit (called
         device memory)

      • Need to copy from host memory and back
         (PCIe 4.0 x16 — 32GB/s peak)

      • Memory transfer is often a bottleneck
      • Pinned (page-locked) memory access is
         much faster

      • Memory hierarchy is a thing,
         just like on CPUs!

https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
```

### Page 16

```text
                       GPU Performance 101
     • GPUs are exceptionally good
         at number crunching

     • However, the memory bandwidth of
         GPUs themselves hardly keeps up




horace.io/brrr_intro
```

### Page 17

```text
                       GPU Performance 101
     • GPUs are exceptionally good
         at number crunching

     • However, the memory bandwidth of
         GPUs themselves hardly keeps up




horace.io/brrr_intro
```

### Page 18

```text
                       GPU Performance 101
     • GPUs are exceptionally good
         at number crunching

     • However, the memory bandwidth of
         GPUs themselves hardly keeps up




horace.io/brrr_intro
```

### Page 19

```text
                       GPU Performance 101
     • GPUs are exceptionally good
         at number crunching

     • However, the memory bandwidth of
         GPUs themselves hardly keeps up

     • The key optimizations are often about
         minimizing or hiding data movement!

     • The principle applies both at the GPU
         level and the multi-GPU level

     • Recent accelerators introduce
         asynchronous/specialized memory
         copies to alleviate that problem

horace.io/brrr_intro
```

### Page 20

```text
                       The Roofline Model™
     • How to find where the bottleneck is?
     • Usually, the tradeoﬀ is between the data
         transfer and compute operations

                                 Computation FLOPs
         Arithmetic intensity:
     •                            Transferred bytes

     • Roofline: plot peak achievable FLOPs
         against the arithmetic intensity




jax-ml.github.io/scaling-book/roofline
modal.com/gpu-glossary/perf/roofline-model
```

### Page 21

```text
                       The Roofline Model™
     • How to find where the bottleneck is?
     • Usually, the tradeoﬀ is between the data
         transfer and compute operations

                                 Computation FLOPs
         Arithmetic intensity:
     •                            Transferred bytes

     • Roofline: plot peak achievable FLOPs
         against the arithmetic intensity

     • You can use this to optimize both the code
         and input shapes for your algorithms!

jax-ml.github.io/scaling-book/roofline
modal.com/gpu-glossary/perf/roofline-model
```

### Page 22

```text
                  Asynchronous execution
     • By default, CUDA kernel calls
         and device transfers are
         asynchronous

     • You can send several kernels
         and wait for results

     • Recent versions of CUDA oﬀer
         better concurrency mechanisms
         (streams, graphs)


https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/
```

### Page 23

```text
                              DL specifics
With PyTorch as an example:

• Kernel execution is asynchronous,
  which hides the latency of Python

• Be careful when benchmarking though!
• Calling Tensor.item() triggers a D2H copy
• Allocated memory is not released
  immediately to simplify caching

• torch.backends.cudnn.benchmark=True
• CUDA streams, graphs etc. are available
  in latest releases
```

### Page 24

```text
           Measuring performance
• Benchmarking is a key step of understanding your bottlenecks and measuring the impact
  of optimizations

• Basically, just run the code several times or measure large workloads
• Can be done via %timeit or timeit.Timer (mind the synchronization)
• Due to possible side-eﬀects (preallocation, caching), warmup and randomization are often
  necessary

• In PyTorch, you can use torch.utils.benchmark
• Don't overoptimize!
  • …at least before you find the true bottleneck
```
