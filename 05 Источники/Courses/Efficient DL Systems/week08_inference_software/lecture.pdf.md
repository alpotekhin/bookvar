---
title: "Week08 Inference Software — lecture slides"
type: external-resource
status: imported-source
language: original
source_kind: slides
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week08_inference_software/lecture.pdf`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/lecture.pdf) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.



The embedded PDF is the primary visual version. The page-separated text below is included for search and quotation; it was extracted mechanically and has not been rewritten.

<iframe class="source-pdf" src="https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week08_inference_software/lecture.pdf?raw=1" title="Week08 Inference Software — lecture slides" loading="lazy"></iframe>

## Extracted slide text

### Page 1

```text
LLM Inference
Efficient DL, Episode VIII ’26
```

### Page 2

```text
ChatGPT
```

### Page 3

```text
YaGPT
```

### Page 4

```text
GigaChat
```

### Page 5

```text
    TTFT
Time To First Token




                      Context decoding
```

### Page 6

```text
     TPS
Tokens Per Second




                    Generation step
```

### Page 7

```text
                Обработка запроса



       запрос

2+2?                    Будет




                       LLM




                2+2?
```

### Page 8

```text
                Обработка запроса



       запрос
                                        равн
2+2?                    Будет                           4
                                         о




                       LLM            LLM            LLM




                2+2?            2+2? Будет     2+2? Будет равно
```

### Page 9

```text
                    Обработка запроса

                 KV                 KV                KV
                cache              cache             cache
                  0                  1                 2


       запрос
                                              равн
2+2?                       Будет                               4
                                               о




                          LLM               LLM              LLM




                   2+2?               2+2? Будет     2+2? Будет равно
```

### Page 10

```text
                    Обработка запроса

                                    K                  K
                KV
                                    V                  V
                cache
                                    1                  2


       запрос
                                          равн
2+2?                        Будет                                 4
                                           о




                           LLM           LLM                    LLM



                                                                      равн
                    2+2?            KV         Будет       KV
                                                                       о
```

### Page 11

```text
                        Обработка запроса

                                                 K                  K
                    KV
                                                 V                  V
                    cache
                                                 1                  2


       запрос
                                                       равн
2+2?                             Будет                                         4
                                                        о




                               LLM                    LLM                    LLM
KV cache —
главное ускорение

Первый шаг дольше
                                                                                   равн
всех остальных
                        2+2?                     KV         Будет       KV
                                                                                    о

                    context decoding (prefill)
```

### Page 12

```text
                    Обработка запроса

                                             K                      K
                KV
                                             V                      V
                cache
                                             1                      2


       запрос
                                                   равн
2+2?                         Будет                                             4
                                                    о




                           LLM                    LLM                        LLM



                                                                                   равн
                    2+2?                     KV         Будет           KV
                                                                                    о

                context decoding (prefill)          шаг генерации
```

### Page 13

```text
                               TPS
                      Tokens Per Second



                       TPS for single query      TPS for single instance




          Assistant   40 TPS is OK (or is it?)        Economics




           Of ine          Do we care?             Most ef cient way




     fi
fl
```

### Page 14

```text
What about batches?
```

### Page 15

```text
                         Batch




Source: https://github.com/bytedance/e ective_transformer
                    ff
```

### Page 16

```text
                         Batch




Source: https://github.com/bytedance/e ective_transformer
                    ff
```

### Page 17

```text
                                  Batch

Tesla V100, oat16, maximum sequence length=64, average sequence length≈40




         Source: https://github.com/bytedance/e ective_transformer
                             ff
   fl
```

### Page 18

```text
  Flash attention




Source: https://arxiv.org/abs/2205.14135
```

### Page 19

```text
  Flash attention




Source: https://arxiv.org/abs/2205.14135
```

### Page 20

```text
  Flash attention




Source: https://arxiv.org/abs/2205.14135
```

### Page 21

```text
          Flash attention


                                           SRAM 19TB/s (20MB)
                                             HBM 1.5TB/s (40GB)
                                               DRAM 13GB/s (> 1TB)


GPU 0      …       GPU N




        Source: https://arxiv.org/abs/2205.14135
```

### Page 22

```text
  Flash attention




Source: https://arxiv.org/abs/2205.14135
```

### Page 23

```text
  Flash attention




Source: https://arxiv.org/abs/2205.14135
```

### Page 24

```text
  Flash attention




Source: https://arxiv.org/abs/2205.14135
```

### Page 25

```text
 Flash attention 2




Source: https://arxiv.org/pdf/2307.08691.pdf
```

### Page 26

```text
 Flash attention 3




Source: https://arxiv.org/pdf/2407.08608
```

### Page 27

```text
     Continuous batch




Source: https://github.com/InternLM/lmdeploy
```

### Page 28

```text
What about long sequence
       batches?
```

### Page 29

```text
 Paged Attention




Source: https://blog.vllm.ai/2023/06/20/vllm.html
```

### Page 30

```text
 Paged Attention




Source: https://blog.vllm.ai/2023/06/20/vllm.html
```

### Page 31

```text
Huge KV caches
```

### Page 32

```text
KV cache reuse
```

### Page 33

```text
Tensor parallel
```

### Page 34

```text
Tensor parallel
  Performance of GPT-20B
```

### Page 35

```text
                         Inference


• Prompt processing
• Autoregressive steps
• Hardware utilization



 • Serving queries
 • Business cases
```

### Page 36

```text
        Frameworks


https://github.com/vllm-project/vllm
https://github.com/sgl-project/sglang
https://github.com/NVIDIA/TensorRT-LLM


https://github.com/ggerganov/llama.cpp
```

### Page 37

```text
                                     Business

     • Di erent scenarios
     • Di erent sources of context
     • Sampling control
       • Cycles
       • Images
       • Censorship




ff
ff
```
