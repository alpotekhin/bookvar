---
title: "Week02 Fast Pipelines — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week02_fast_pipelines/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week02_fast_pipelines/lecture.pdf?raw=1" title="Week02 Fast Pipelines — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
Efficient Deep Learning Systems
  Optimizing training pipelines
           Max Ryabinin




                2026
```

### Page 2

```text
                       Plan for today
• Understanding performance limits
• Mixed precision training
  • When and why to use it
  • How to enable it and utilize it to the fullest
  • Dealing with stability in training
• Storing and loading training data eﬃciently
• Profiling DL code
```

### Page 3

```text
      DL performance indicators
• When do we want to optimize our code?
  • When do we know we’re “good enough”?
• Ultimately, hardware performance is the limiting factor
• If most of the time is spent on useful computations, then our code is close
  to optimal

• How do we know if we're close to the limit?
```

### Page 4

```text
                                       nvidia-smi
   • On most Linux systems with a GPU
      driver, you have an easy way to check
      GPU status




arthurchiao.art/blog/understanding-gpu-performance
```

### Page 5

```text
                     nvidia-smi (the wrong way)
   • On most Linux systems with a GPU
      driver, you have an easy way to check
      GPU status

   • However, its utilization metric does
      not show what you think!

   • It shows the percentage of time when
      the GPU was running anything

      • A dummy kernel waiting for CPU-
         GPU sync could result in 100%
         “utilization”
arthurchiao.art/blog/understanding-gpu-performance
```

### Page 6

```text
   The right way: Model FLOPS Utilization
   • Defined as the ratio of observed FLOPS (floating point operations/second) to theoretical
      maximum FLOPS on given hardware [1]

   • A system-independent metric that indicates end-to-end performance
   • Rule of thumb: MFU >45% is a good eﬃciency target
   • Many benchmarks published [2], but beware of diﬀerent ways to compute FLOPS!




[1] PaLM: Scaling Language Modeling with Pathways. Chowdhery et al., 2022
[2] github.com/mosaicml/llm-foundry/tree/main/scripts/train/benchmarking#results
```

### Page 7

```text
The rightest way: MFU it depends :)
• MFU assumes a fixed compute type for everything
• Sometimes FLOPs might be input/operation-
  dependent

• We might not account for operations beside forward/
  backward/step

  • HFU (Hardware FLOPs Utilization) could be a
     solution

• If you’re interested in the system performance,
  tokens/second could do

• Various indicators from DCGM report actual use of
  full GPU resources
```

### Page 8

```text
     GPU compute saturation: takeaways

    • The performance limits are defined by how much of hardware we can
       eﬀectively utilize

    • Use MFU/HFU as a first approximation
       • Or hardware counters from DCGM [1] for most accurate measures
    • Beware of diﬀerent ways to compute FLOPs (both for model
       computations and for hardware)


[1] docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html#profiling-metrics
```

### Page 9

```text
          Floating point numbers
• Neural networks require real numbers…
• …which need to be represented in finite memory
• Single precision (FP32) is the default format with 4 bytes of storage



                                                           23

                                                  (                    )
                    value = (−1)sign × 2E−127 ×                b23−i2−i
                                                           ∑
                                                      1+
                                                           i=1

• Special values (0, NaN, ±inf) are encoded by exponent values
```

### Page 10

```text
         Why use low precision?
• Can we go smaller than 32 bits? Should we?
• Key benefits:
  • Reduced memory usage (duh)
  • Faster performance (due to higher arithmetic intensity or smaller
    communication footprint)

  • Can use specialized hardware for even faster computation
• Makes your code prone to spectacular explosions :)
```

### Page 11

```text
                   Floating point formats
 • Naive FP16 is not the only option!
 • Specialized formats preserve dynamic range for computations




src: developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores
```

### Page 12

```text
      Switching to lower precision
• FP16 exists since CUDA 8, just allocate the tensor/cast it to half

• BF16 is available on CPUs, TPUs and recent GPUs [1], Tensor.bfloat16() in PyTorch

• TF32 can be enabled for you on Ampere GPUs
  (was enabled in PyTorch by default until 1.12)

   • Never exposed as a data type, only as a type for specific operations [2]




[1] pytorch.org/xla/release/1.9/index.html#xla-tensors-and-bfloat16
[2] developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores
```

### Page 13

```text
                                  Tensor Cores
 • Specialized computation units available in latest generations of NVIDIA GPUs (since Volta)

 • Allow the user to speed up D = A × B + C by up to 8-16x (claimed)




nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf
```

### Page 14

```text
                                  Tensor Cores
 • Specialized computation units available in latest generations of NVIDIA GPUs (since Volta)

 • Allow the user to speed up D = A × B + C by up to 8-16x (claimed)
 • Enabled not only for TF32/FP16/BF16 (Ampere), but even for INT8/INT4
 • You do not specify their usage manually!




nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf
```

### Page 15

```text
                         Utilizing Tensor Cores
     • To enable them, you either need recent CUDA or specific size constraints:




[1] docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
[2] developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf
```

### Page 16

```text
                         Utilizing Tensor Cores
     • To enable them, you either need recent CUDA or specific size constraints:




[1] docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
[2] developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf
```

### Page 17

```text
                         Utilizing Tensor Cores
     • To enable them, you either need recent CUDA or specific size constraints:
     • Run GPU profiler to check if they are used ([i|s|h](\d)+ in kernel names)
     • Also, DL profilers can indicate Tensor Core eligibility and usage




[1] docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
[2] developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9926-tensor-core-performance-the-ultimate-guide.pdf
```

### Page 18

```text
           Mixed precision training
• Training in pure FP16 hardly works
• Some operations (e.g. matrix multiplication) can work, others (softmax, batch normalization)
  need higher precision

• Mixed precision training casts layer activations to appropriate data types
• Supported in popular DL frameworks (e.g. torch.cuda.amp)




• Increases the training throughput due to the use of Tensor Cores (MFU trickier to compute)
• Decreases the memory usage by half… or not?
```

### Page 19

```text
         Memory savings of AMP
• Let’s count the number of bytes per parameter for standard training with Adam:
 FP32:                                  AMP:

 • Parameters — 4 bytes                 • Parameters — 2 bytes
 • Gradients — 4 bytes                  • Master parameters — 4 bytes
 • Optimizer statistics — 8 bytes       • Gradients — 2 bytes (sometimes 4)
 16 bytes per parameter in total        • Optimizer statistics — 8 bytes
                                        Also 16 bytes per parameter!

• The only major savings come from reduced activation memory
```

### Page 20

```text
                                         FP8 training
      • On latest hardware (e.g., H100), we have
         even lower precision formats

      • E4M3 is used for weights and activations,
         E5M2 is best for gradients




FP8 Formats for Deep Learning. Micikevicius et al., 2022
```

### Page 21

```text
                                         FP8 training
      • On latest hardware (e.g., H100), we have
         even lower precision formats

      • E4M3 is used for weights and activations,
         E5M2 is best for gradients

      • Extra tricks, e.g. per-tensor or per-block
         (MX) scaling, required to maintain accuracy

      • In PyTorch: github.com/pytorch/ao/tree/
         main/torchao/float8

      • Also, github.com/NVIDIA/
         TransformerEngine can leverage this
FP8 Formats for Deep Learning. Micikevicius et al., 2022
```

### Page 22

```text
                  AMP: takeaways

• Use more eﬃcient data types when available
• Mind the sizes/operation types to preserve accuracy
• Don’t expect significant memory savings for large models
• In many cases, this is easy to integrate through standard tools
```

### Page 23

```text
           Bottlenecks in data loading




colin-scott.github.io/personal_website/research/interactive_latency.html
```

### Page 24

```text
           Bottlenecks in data loading




colin-scott.github.io/personal_website/research/interactive_latency.html
```

### Page 25

```text
           Bottlenecks in data loading




colin-scott.github.io/personal_website/research/interactive_latency.html
```

### Page 26

```text
     Bottlenecks in data loading
• Sometimes the models aren’t so compute-intensive…
• We still want to process the data eﬃciently!
• Need to be mindful of hardware/network performance and the CPU code
• Two components: what to read and how to read
• Obvious part: read data in parallel
  (several processes, asynchronously with computation)
```

### Page 27

```text
                   Storage formats
• Raw files are often easy to visualize, but storage-ineﬃcient
  (especially when accessing external storage)

• In some cases, you might benefit from better formats:
  • For structured data, Apache Arrow/Protobuf/msgpack etc.
  • For images, apply non-random “heavy” processing before training
  • For language data, tokenize the texts and store integer indices only
```

### Page 28

```text
 Minimizing preprocessing time


• Reading the data and feeding it into the model can also be slow
  • For large images, you can be bound by CPU operations
  • For sequence data, you can waste time on padding tokens
```

### Page 29

```text
 Performance of image loading
• When reading images, consider the code that reads them :)
  • Default PIL.Image.Open can be highly ineﬃcient!
    Use at least Pillow-SIMD

  • Use better decoders (e.g. jpegturbo, nvJPEG from DALI)
```

### Page 30

```text
 Performance of image loading
• When reading images, consider the code that reads them :)
  • Default PIL.Image.Open can be highly ineﬃcient!
    Use at least Pillow-SIMD

  • Use better decoders (e.g. jpegturbo, nvJPEG from DALI)
• Heavy groups of augmentations can also slow you down
  • Consider moving them to GPU (e.g. kornia, DALI)
  • In most cases, you can switch to eﬃcient implementations
```

### Page 31

```text
   Optimal sequence processing
• For sequential data, padding in batches is necessary
• However, padding the ENTIRE dataset can lead to redundant timesteps
• It’s usually better to store samples without padding and use collate_fn
• Also, bucket examples by length to further minimize padding
• …or, even pack multiple examples into the same sequence
```

### Page 32

```text
      Data pipelines: takeaways


• Consider the performance/size of your storage when loading the data
• Use better deserialization primitives when available
• Try to avoid obvious ineﬃciencies when building task-specific pipelines
```

### Page 33

```text
          Profiling: what and why


• In benchmarking, we measure the speed of our program as a black box
• Profiling is a process of determining the runtime of parts of your program
• More of a “white box” approach
```

### Page 34

```text
    How to profile Python code?
• cProfile as a standard tool built into Python
• Sampling-based profilers (scalene etc.)
• Some of them (e.g. py-spy) even allow to attach to running code!
```

### Page 35

```text
           How to profile GPU code?
 • nvprof is the low-level profiling tool
 • Gives you the performance of low-level kernel launches and copies




developer.nvidia.com/blog/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/
```

### Page 36

```text
  How to profile PyTorch code?


• High-level: torch.utils.bottleneck
• Older API: torch.autograd.profiler
• Newer one: torch.profiler
```

### Page 37

```text
    PyTorch Profiler + trace viewer




pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
```

### Page 38

```text
    Nsight Systems/Nsight Compute




developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9339-profiling-deep-learning-networks.pdf
```

### Page 39

```text
                 Profiling: typical patterns




paulbridger.com/posts/nsight-systems-systematic-optimization
```

### Page 40

```text
                 Profiling: typical patterns




paulbridger.com/posts/nsight-systems-systematic-optimization
```

### Page 41

```text
    • This is the best case!
    • You need to optimize the model itself (lower precision, faster kernels etc)
paulbridger.com/posts/nsight-systems-systematic-optimization
```

### Page 42

```text
   • Operations run faster than kernels are scheduled
   • Also happens during inference
   • You need to optimize the CUDA API calls (torch.compile, TorchScript) or
       have more compute-intensive operations (e.g. larger batches)
paulbridger.com/posts/nsight-systems-systematic-optimization
```

### Page 43

```text
   • CPU and GPU processing are too heavily interleaved
   • Remove unnecessary synchronization points, execute as much work on the
       GPU as possible

paulbridger.com/posts/nsight-systems-systematic-optimization
```

### Page 44

```text
              Profiling: takeaways

• A very useful tool for understanding the performance of your pipeline
• Can be applied to both CPU and GPU code
• Depending on the required granularity of measurements,
  you can use diﬀerent approaches
```
