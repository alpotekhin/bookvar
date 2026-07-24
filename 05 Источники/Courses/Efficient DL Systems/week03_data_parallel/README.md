---
title: "Week 3: Data-parallel training and All-Reduce"
type: external-resource
status: imported-source
language: original
source_kind: readme
source_commit: e632aa89ca9e6638d52e1b686095e7442faffbb0
---

> [!note] Original course material
> The material below is preserved in its original language from
> [`week03_data_parallel/README.md`](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/README.md) in *Efficient Deep Learning Systems* at commit
> `e632aa89ca9e6638d52e1b686095e7442faffbb0`. License: [MIT](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE). Bookvar changed only the
> publication markup and link paths.

* Lecture: [link](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/lecture.pdf)
* Seminar: [link](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/practice.ipynb) + [DDP Demo](https://github.com/vorobyov01/llm-training-sandbox)
* Homework: see the [homework](https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week03_data_parallel/homework) folder
* Video: -, -

## Further reading
* [Numba parallel](https://numba.pydata.org/numba-doc/dev/user/parallel.html) - a way to develop threaded parallel code in python without GIL
* [joblib](https://joblib.readthedocs.io/) - a library of multiprocessing primitives similar to mp.Pool, but with some extra conveniences
* BytePS paper - https://www.usenix.org/system/files/osdi20-jiang.pdf
* Alternative lecture: Parameter servers from CMU 10-605 - [here](https://www.youtube.com/watch?v=N241lmq5mqk)
* Alternative seminar: python multiprocessing - [playlist](https://www.youtube.com/watch?v=RR4SoktDQAw&list=PL5tcWHG-UPH3SX16DI6EP1FlEibgxkg_6)
* [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html) (pay attention to `fork` vs `spawn`!)
* [PyTorch Distributed tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
* [Collective communication protocols in NCCL](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)
* There's a ton of links on the slides, please check the PDF.
