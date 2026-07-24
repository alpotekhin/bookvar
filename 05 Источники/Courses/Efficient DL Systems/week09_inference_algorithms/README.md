---
title: "Week 9: Efficient model inference"
type: external-resource
status: imported-source
language: original
source_kind: readme
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week09_inference_algorithms/README.md`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/README.md) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.

* Lecture: [link](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/lecture.pdf)
* Seminar: [link](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/seminar.ipynb)
* Homework: see the [homework](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week09_inference_algorithms/homework) folder

## Further reading
* [GPU MODE Lecture 14: Practitioners Guide to Triton](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/#auto-tuning)
* [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/)
* [Deep Dive on the Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/)
* [Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)
* [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
* [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
* [Accelerating 2D Dynamic Block Quantized Float8 GEMMs in Triton](https://pytorch.org/blog/accelerating-gemms-triton/)
* [SmoothQuant paper](https://arxiv.org/abs/2211.10438)
* [SmoothQuant repo](https://github.com/mit-han-lab/smoothquant)
