# Serving engines: original source illustrations

Prepared and personally inspected on 2026-10-14 for the canonical chapter
"55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer".

All three PNGs are unchanged upstream originals. No crop, redraw, translation,
resizing, or generated image was used. The adjacent assets.json is a proposal
for the central asset registry, not evidence that integration or browser QA has
already completed.

## vLLM: separate API, EngineCore and TP workers

- Asset: vllm-v1-processes-tp4.png, 2816 × 1536.
- Author: vLLM contributors.
- Repository and version: vllm-project/vllm, v0.31.0,
  db9527a46873454610df6dbedf79a36d6bf1a7f6.
- [Original PNG](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/assets/design/arch_overview/v1_process_architecture_tp4.png).
- [Teaching context](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/docs/design/arch_overview.md).
- SHA-256: 8602fecbabf3b5fb4ceeb831f608dc178d14ec4a686e2f2cff58b1af78f8951d.
- License: Apache-2.0 under the [pinned repository LICENSE](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/LICENSE).
  Exact license copy: vllm-LICENSE.txt.

The diagram shows responsibilities and process boundaries without a benchmark
claim. Its TP=4 configuration is specifically one API process, one EngineCore
and four GPU workers. The four workers are not four independent model replicas.
The chapter's caption scopes the process count to this example. Preserve the
whole image, including the process labels and ZMQ connection.

## TensorRT-LLM: PyExecutor components

- Asset: tensorrt-llm-pyexecutor.png, 1928 × 710.
- Author and rights holder: Copyright (c) 2011-2025 NVIDIA CORPORATION &
  AFFILIATES. All rights reserved.
- Repository and version: NVIDIA/TensorRT-LLM, v1.2.1,
  376f7e1bd8ed543f75014309e3fd4b237e9b0e73.
- [Original PNG](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/media/TRTLLM_Architecture_Overview.png).
- [Teaching context](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/docs/source/developer-guide/overview.md).
- SHA-256: 2083d1860246bc30cac0e8d2ebab8540a185cc0e76c9b2ad773d2bd73dd28318.
- License: Apache-2.0 under the [pinned repository LICENSE](https://github.com/NVIDIA/TensorRT-LLM/blob/376f7e1bd8ed543f75014309e3fd4b237e9b0e73/LICENSE).
  Exact full license bundle, including the NVIDIA notice and third-party
  notices: tensorrt-llm-LICENSE.txt.

The expanded Rank0 executor is on the left; other executors are on the right.
Scheduler, KVCacheManager, ModelEngine and Sampler are distinct components.
This diagram describes the PyTorch/PyExecutor path, not every TensorRT-engine
configuration. In particular, "ModelEngine" is not proof of a prebuilt
TensorRT engine. The chapter's text checks the API inheritance separately.

## FlashInfer: ragged storage and sequence offsets

- Asset: flashinfer-ragged.png, 896 × 325.
- Author: FlashInfer contributors.
- Image repository: flashinfer-ai/web-data,
  d7cc3d229f23d7c0a81c1e45d8591f4337c18efb.
- [Original PNG](https://github.com/flashinfer-ai/web-data/blob/d7cc3d229f23d7c0a81c1e45d8591f4337c18efb/tutorials/ragged.png).
- [Teaching context in FlashInfer v0.7.1](https://github.com/flashinfer-ai/flashinfer/blob/0b46115d238d29ea2e92c6b9b2cb077d40fbc352/docs/tutorials/kv_layout.rst).
- SHA-256: efc4f5e9d414679f0bffb3bf3b96956b62c62ecc4eed621aff903e0b23b600d3.
- License: Apache-2.0 under the [image repository LICENSE](https://github.com/flashinfer-ai/web-data/blob/d7cc3d229f23d7c0a81c1e45d8591f4337c18efb/LICENSE).
  Exact license copy: flashinfer-web-data-LICENSE.txt.

The figure has numeric boundaries [0, 5, 8, 12], so color is not the sole
carrier of meaning. It illustrates the interface between the caller's
description of independent sequences and an attention kernel. It does not
claim that putting sequences in one buffer permits cross-request attention.
The chapter supplies an executable NumPy example of the required mask.

The figure's image repository has its own pinned LICENSE. We do not infer
licensing merely from its appearance in the documentation or from the
software repository's license.

## Selection and retired embeds

EDLS week08's inference-software lecture and CS336 lecture10 remain useful
theoretical/practical references. Their prefill/decode and cache illustrations
were not duplicated here because those mechanisms belong to chapters 55/55b.
Three source-authored architecture/interface diagrams better match this
chapter's remaining explanation.

These existing assets were removed from chapter 55c only; no source/reference
article or binary was changed. Integrator should remove only this chapter's
used_in relationship from their central registry records:

- curated-vllm-engine-loop-a1b2c3d4e502 — inference-serving/vllm-engine_loop.png.
- curated-vllm-engine-constructor-a1b2c3d4e501 — inference-serving/vllm-engine_constructor.png.
- curated-vllm-forward-pass-a1b2c3d4e503 — inference-serving/vllm-fwd_pass.png.
- curated-sglang-program-a1b2c3d4e50b — inference-serving/sglang-program-figure2.png.
- curated-sglang-radix-a1b2c3d4e50c — inference-serving/sglang-radixattention-figure3.png.

The Gordić reference article remains unchanged, with its original illustrations.
The SGLang program crop cuts into the paper caption and is not used as a new
licensed original. The radix-tree walkthrough belongs to the detailed
reference/scheduling explanation rather than being repeated in this chapter.
The old registry's user-confirmed-open-materials status is not reinterpreted
as a named license.

## Validation boundaries

Personally inspected each selected original at full resolution, including
labels, topology and layout. Downloads and local copies have matching hashes;
the PNGs decode successfully at the dimensions above. Preserve aspect ratios
and zoom access in the site; this source inspection is not browser acceptance.

The CPU checker executes the chapter's actual Python block and additional
isolation/layout tests. It does not call any serving engine, FlashInfer,
CUDA or NCCL. No GPU compatibility, performance, live HTTP handling, site
build, EN parity, registry integration, deployment or publication is claimed.

License-file SHA-256:

- vllm-LICENSE.txt:
  c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4.
- tensorrt-llm-LICENSE.txt:
  5d8f6f1a2f855c68a273674d6a3925c6349d7632d0864d089b46f94d8ea76eff.
- flashinfer-web-data-LICENSE.txt:
  c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4.
