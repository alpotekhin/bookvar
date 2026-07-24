---
title: "Week 8: LLM inference optimizations and software"
type: external-resource
status: imported-source
language: original
source_kind: readme
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week08_inference_software/README.md`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/README.md) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.

* Lecture: [link](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/lecture.pdf)
* Seminar: [link](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/seminar.ipynb)
* Homework: [link](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/homework/homework_week8.ipynb)

## Further reading
* [What is the KV cache?](https://mett29.github.io/posts/kv-cache/) - explains the key-value cache mechanism used to speed up autoregressive generation
* [What is Continuous Batching?](https://huggingface.co/blog/continuous_batching) - introduces continuous batching for efficient batched inference
* [How does Paged Attention work?](https://docs.vllm.ai/en/latest/design/paged_attention/) - describes the Paged Attention algorithm for memory-efficient attention
* [vLLM Documentation](https://docs.vllm.ai/en/latest/) - official documentation for the vLLM inference engine
* [SGLang Documentaton](https://docs.sglang.io) - official documentation for the SGLang inference framework
* [Inference Optimization Techniques](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) - NVIDIA blog post on mastering LLM inference optimization techniques
