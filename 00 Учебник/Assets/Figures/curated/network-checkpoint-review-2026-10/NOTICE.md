# Original illustrations for network transfer and asynchronous checkpoints

Selected, inspected and integrated on 2026-10-14 into `44g Network, storage и distributed checkpoints.md`. Central-registry records match the two originals; the PyTorch BSD notice is also shipped in `site/public/licenses/pytorch-website-d1050dad-BSD-3-Clause.txt`.

Both PNG files are byte-for-byte upstream originals. No translation, redraw, crop, resampling, or generated graphics were applied. The adjacent `assets.json` contains the two provenance records.

## GPUDirect RDMA: host staging versus a direct device path

- Local file: `openhackathons-gpudirect-rdma.png`, 1726 × 517 pixels, with transparency.
- Original: [`gpudirect_rdma.png`](https://github.com/openhackathons-org/nways_multi_gpu/blob/bc2a5887505bba17672882b3ec7a62c04ecf0d24/labs/CFD/English/C/images/gpudirect_rdma.png).
- Commit: `bc2a5887505bba17672882b3ec7a62c04ecf0d24`.
- SHA-256: `9dc56a7b95b6a48732bdee06b63937c793486c2304b08a40a75d346d2a4838d6`.
- Teaching context and explicit licensing statement: [CUDA-aware MPI notebook, including its Licensing section](https://github.com/openhackathons-org/nways_multi_gpu/blob/bc2a5887505bba17672882b3ec7a62c04ecf0d24/labs/CFD/English/C/jupyter_notebook/mpi/cuda_aware.ipynb).
- Rights holder and attribution: Copyright © 2022 OpenACC-Standard.org, in collaboration with NVIDIA Corporation.
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Full license retained as `LICENSE-CC-BY-4.0.txt`.

The notebook expressly licenses the teaching material under CC BY 4.0 and embeds this image in its GPUDirect discussion. Its licensing statement also preserves applicable third-party rights. The [contribution policy](https://github.com/openhackathons-org/nways_multi_gpu/blob/bc2a5887505bba17672882b3ec7a62c04ecf0d24/CONTRIBUTING.md) independently distinguishes CC BY 4.0 documentation from software licenses. The `labs/CFD/LICENSE` applies a BSD-style notice to the inherited code; it is **not** the license basis used for this illustration.

### Why this image

The two panels answer one question: which intermediate memory buffer disappears when the NIC can access GPU memory directly? The original retains GPU memory, host memory, CPU/chipset, and the network adapter; it makes no claim that a checkpoint becomes smaller or that a specific amount of storage I/O vanishes. There are no latency or bandwidth estimates to inherit.

### Proposed caption

> Слева данные перед отправкой проходят через память хоста. Справа GPUDirect RDMA позволяет сетевой карте читать память GPU напрямую. Промежуточное копирование в host RAM исчезает; размер передаваемых данных не меняется. Управляющий код, регистрация памяти и настройка сети остаются необходимы. Источник: OpenACC / NVIDIA, N-Ways to Multi-GPU Programming, CC BY 4.0.

This is a conceptual comparison, not a wiring diagram for every server. It illustrates a data-transfer path, not direct persistence into a remote durable checkpoint. Preserve its light/white background: the original is transparent and contains black text. Do not display it against an unmodified dark background.

## PyTorch: staging followed by background persistence

- Local file: `pytorch-async-checkpoint-flow.png`, 1600 × 286 pixels.
- Original: [`assets/images/reducing-checkpointing-times/fg3.png`](https://github.com/pytorch/pytorch.github.io/blob/d1050dad1fe8ff2b28e1672d6b9f3eaa33e91320/assets/images/reducing-checkpointing-times/fg3.png).
- Commit: `d1050dad1fe8ff2b28e1672d6b9f3eaa33e91320`.
- SHA-256: `4abf122a9cd4a80a26569e8fc82728ba29bffbdd27483fffbc685cbfc314d2a6`.
- Article: [Reducing Model Checkpointing Times by Over 10x with PyTorch Distributed Asynchronous Checkpointing](https://pytorch.org/blog/reducing-checkpointing-times/), 2024-06-12.
- Article authors: Lucas Pasqualin, Less Wright, Iris Zhang, Chien-Chin Huang, Swaminathan Sundararaman, Saransh Gupta, and Raghu Ganti.
- Rights notice: Copyright (c) 2018, Facebook Inc. All rights reserved.
- License: **BSD-3-Clause**, from the [pinned website-repository LICENSE](https://github.com/pytorch/pytorch.github.io/blob/d1050dad1fe8ff2b28e1672d6b9f3eaa33e91320/LICENSE). Full unchanged notice retained as `LICENSE-pytorch-BSD-3-Clause.txt`.

### Why this image

Two lanes distinguish the training thread from the checkpointing work. The figure shows the previous checkpoint's completion being awaited, the GPU-to-CPU copy, and serialization overlapping the next training step. It contains no measured pause times, model sizes, or claim that an API return alone means durable storage.

### Proposed caption

> В этом варианте `async_save` обучение ждёт завершения копирования GPU→CPU, но запись подготовленной копии идёт параллельно следующему шагу. Перед запуском нового сохранения ожидают завершения предыдущего. Рисунок показывает блокирующий staging и фоновую запись; отдельный режим асинхронного staging требует дополнительной синхронизации перед изменением сохраняемых тензоров. Источник: PyTorch, 2024, BSD-3-Clause.

The article's diagram predates the chapter's pinned PyTorch version. Use it to explain the blocking-staging workflow, not as a promise about the default behavior of every `DefaultStager` configuration. Neither this figure nor its caption establishes crash atomicity, durable replication, or restore correctness. Those guarantees remain the subject of the chapter's text and tests.

The 5.6:1 aspect ratio makes this a full-width, zoomable figure. At a narrow 600 px column its labels become too small. Do not crop away either lane or force it into a thumbnail/card; preserve access to the 1600 px original.

## Selection and validation

- Personally inspected both chosen PNGs at their original resolution and confirmed the visible labels, arrows, and source context.
- Verified that the pinned PNG downloads and the copied local files have identical SHA-256 hashes; image decoding and dimensions checked locally.
- Reviewed and rejected the EDLS H100 host map for this replacement: its per-link/aggregate bandwidth annotations would introduce another numerical interpretation problem into a section being repaired for that reason.
- Also inspected the NVIDIA/Open Hackathons multi-GPU slides (PDF page 41, printed slide 43), which explicitly carry CC BY 4.0. The selected comparison PNG is clearer because it shows both transfer paths together and does not require keeping the slide's title and surrounding empty space.
- Inspected additional PyTorch checkpoint figures showing benchmark curves; excluded them because they would answer a different question and require reproducing workload-specific measurements.
- No live distributed run, GPU benchmark, chapter edit, shared-registry edit, site build, or browser layout acceptance is claimed by this source-selection work. Chapter placement and browser QA remain with the integrating editor.

## License-file checksums

- `LICENSE-CC-BY-4.0.txt`: `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411`.
- `LICENSE-pytorch-BSD-3-Clause.txt`: `9e8b5a3ee55cbf30f0ce2b43b25daff2461f1251862d40e012f5c45916603dd2`.
