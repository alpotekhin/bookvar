# 55c Serving engines — source and content audit, 2026-10-14

Superseded: the user rejected this prose revision. The source-text rebuild and
current evidence are recorded in 2026-10-14-serving-originals.md. The technical
checks below are historical evidence, not acceptance of the current wording.

## Scope and delivered files

- Canonical Russian chapter:
  00 Учебник/14 Inference и оптимизация/55c Serving engines — vLLM, SGLang, TensorRT-LLM и FlashInfer.md.
- CPU checker: publishing/tools/check_serving_engine_examples.py.
- Three unchanged upstream PNGs, three complete LICENSE copies, NOTICE.md and
  central-registry proposals in assets.json:
  00 Учебник/Assets/Figures/curated/serving-engines-review-2026-10/.

The chapter's canonical path, title, H1 and two source-unit IDs were preserved.
All 11 former H2 anchors remain reachable by unchanged headings or explicit
anchors. No Gordić reference article, immutable source, adjacent chapter,
EN page, navigation file, shared test or central registry was edited by this
subtask. The integrating editor corrected the TensorRT caption's left/right
description; that correction is preserved.

## What the content now explains

The full original chapter was read before editing. The revision follows the
life of a concrete request instead of enumerating product features:

1. Chat template, tokenization and request state.
2. Scheduling and KV placement as inputs to model execution.
3. Model execution, logits and sampling as separate responsibilities.
4. Why the sampled token acquires its own KV only on the following ordinary
   autoregressive pass; EOS needs no additional forward call.
5. Detokenization/streaming boundaries, completion and safe memory reuse.
6. vLLM API/EngineCore/worker separation, TP versus replicas, model runner and
   the scope of CUDA Graph replay.
7. SGLang's optional programming frontend versus independent HTTP runtime;
   its concrete Python TokenizerManager → Scheduler → DetokenizerManager
   path and the dependencies that remain under CPU/GPU overlap.
8. TensorRT-LLM's PyTorch/PyExecutor and separately prepared TensorRT-engine
   paths. ModelEngine as a component name does not prove a compiled engine.
9. FlashInfer's caller/kernel interface: ragged boundaries, separate Q/KV
   lengths, physical layout, plan versus run, and request isolation.
10. The distinct roles of Triton language and NVIDIA Triton Inference Server;
    llama.cpp is neither CPU-only nor inherently single-user.
11. Matching task, precision, workload and measurement boundaries before
    comparing engines. No product leaderboard or unsupported speed claim.

Detailed paging, radix-tree operations, continuous batching and prefix-cache
policy remain in chapters 55/55b and their reference pages, rather than being
repeated here.

## Primary evidence and pinned versions

Checked official live documentation and version-pinned repository sources.
The links in the chapter use full commit hashes.

| Project | Reviewed version / commit | Evidence used |
|---|---|---|
| vLLM | v0.31.0, db9527a46873454610df6dbedf79a36d6bf1a7f6 | docs/design/arch_overview.md, especially V1 Process Architecture and Model Runner |
| SGLang | v0.5.21, e00930c5489053f26d86b179cee0d087f846acbb | python/sglang/srt/entrypoints/http_server.py launch_server; managers/scheduler.py event_loop_overlap; docs/docs/basic_usage/openai_api.mdx |
| TensorRT-LLM | v1.2.1, 376f7e1bd8ed543f75014309e3fd4b237e9b0e73 | docs/source/developer-guide/overview.md, llmapi/llm.py, quick-start-guide.md and legacy/architecture/workflow.md |
| FlashInfer | v0.7.1, 0b46115d238d29ea2e92c6b9b2cb077d40fbc352 | docs/tutorials/kv_layout.rst and flashinfer/prefill.py |
| llama.cpp | 6868a52cb337450658006ad2f9281c2eb63f1666 | README, CPU/GPU/hybrid support and server entry point |
| Stanford CS336 | 8b59b50730766695c2ffedd1a79c50cd09b9eb91 | lecture_10.py, inference constraints and serving-system context |
| Efficient DL Systems | e632aa89ca9e6638d52e1b686095e7442faffbb0 | week08_inference_software/lecture.pdf, full extracted 38-slide text; existing practice/source links retained |

The original SGLang paper motivates branching programs and prefix reuse but
does not alone describe the selected current runtime. Its historical figures
were not relicensed by inference from public accessibility.

### Documentation pitfalls resolved

- The old TensorRT-LLM architecture URL remains available with a 2025 revision.
  The pinned v1.2.1 developer guide is the current evidence used here.
- A second architecture file in the pinned TensorRT repository still contains
  a restrictive "only greedy decoding" description. The chapter does not
  repeat it: the newer guide and actual LLM API establish the two execution
  paths without requiring that stale restriction.
- TensorRT v1.2.1 LLM inherits _TorchLLM. Its argument validation explicitly
  directs TensorRT-engine users to a separate entry point. No API stability
  beyond this pinned version is asserted.
- SGLang's older developer-guide URL is not a reliable current documentation
  entry point. The server route is instead checked against version-pinned
  launch_server and scheduler code.
- FlashInfer's web-data image repository has its own Apache-2.0 LICENSE;
  the license is verified there, not merely inferred from the code package.

## Executed CPU checks

Command:

    /Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 publishing/tools/check_serving_engine_examples.py

Observed:

    PASS Markdown block and independent scalar reference; max error 2.220e-16
    PASS request isolation; wrong global causal mask leaks across both boundaries
    PASS offset guards and NHD/HND permutation; relabel/reshape control fails
    PASS stipulated prefill/decode token-versus-KV accounting
    CPU-only mathematical checks passed; no engine/GPU/performance claim.

The checker extracts and executes the actual fenced Python block in the
Markdown chapter. It then compares the result to a separate scalar reference
(no packed mask or matrix multiplication), changes the first request's V by
100, and verifies that other requests stay unchanged with correct boundaries.
A global triangular mask is a negative control and leaks the changed values.

Additional checks use two heads to verify a real NHD↔HND permutation and reject
mere relabel/reshape. Six malformed offset arrays are rejected under this
example's explicit nonempty-sequence contract. Empty-request handling is not
claimed as a FlashInfer API restriction. The token/KV table is checked as
stipulated bookkeeping, not as a run of a language model.

Also executed:

- python3 -m py_compile publishing/tools/check_serving_engine_examples.py — PASS.
- Scoped git diff --check for the chapter and checker — PASS.
- All 11 former H2 anchors — preserved.
- Both source-unit IDs — preserved.
- All 16 distinct wiki targets — exist as files/directories or appended .md
  source notes, including the course lecture.pdf.md / seminar.ipynb.md notes.
- All three new local asset hashes — equal metadata and fresh downloads of
  their pinned upstream PNG URLs.
- All three license_copy targets — exist.
- All three PNGs — decode at the recorded dimensions and were personally
  visually inspected in full.

## Figure provenance and integration

Complete records are in the adjacent assets.json; none require cropped-PDF
or user-confirmed-open-materials licensing rules. Each record has a pinned
PNG source_url, separate source_page_url, raw source_image_url, commit,
source/output SHA-256, Apache-2.0 license URL and exact local license copy,
rights_status=source-license-verified, derivation=source-raster-copy, and
provenance_confirmation=pinned-upstream-repository.

| File | Pixels | SHA-256 |
|---|---|---|
| vllm-v1-processes-tp4.png | 2816 × 1536 | 8602fecbabf3b5fb4ceeb831f608dc178d14ec4a686e2f2cff58b1af78f8951d |
| tensorrt-llm-pyexecutor.png | 1928 × 710 | 2083d1860246bc30cac0e8d2ebab8540a185cc0e76c9b2ad773d2bd73dd28318 |
| flashinfer-ragged.png | 896 × 325 | efc4f5e9d414679f0bffb3bf3b96956b62c62ecc4eed621aff903e0b23b600d3 |

The new images illustrate process boundaries, executor composition and
sequence layout respectively. They make no hardware speed claim. Caption
qualifications prevent TP workers being read as replicas and prevent
ModelEngine being mistaken for a compiled TensorRT artifact.

Remove only 55c's used_in relationship from these old registry entries:

- curated-vllm-engine-loop-a1b2c3d4e502.
- curated-vllm-engine-constructor-a1b2c3d4e501.
- curated-vllm-forward-pass-a1b2c3d4e503.
- curated-sglang-program-a1b2c3d4e50b.
- curated-sglang-radix-a1b2c3d4e50c.

Their binaries and other readers/reference uses are preserved. Gordić's
source article is unchanged. NOTICE.md records why these embeds were retired
from this chapter and why the replacements are clearer.

## Remaining integration / non-claims

Central-registry integration, public license copies, full publication tests,
site build and desktop/mobile browser acceptance are owned by the parent
editor. This audit does not mark those stages complete.

No installed vLLM/SGLang/TensorRT-LLM/FlashInfer server was executed. No GPU,
CUDA Graph, NCCL, HTTP cancellation, memory lifetime or performance benchmark
was measured. The CPU example demonstrates the mathematical isolation
requirement only. EN parity and deployment are outside this subtask.

## Primary-editor integration

The primary editor read the full revised chapter, the added NumPy block and
the checker; reran the checker successfully; inspected all three original
diagrams; verified the vLLM process documentation and TensorRT-LLM LLM API
inheritance against the pinned source. The lead-in to the NumPy example now
names request boundaries, not the preceding axis-layout discussion.

The three assets and their complete public license copies are integrated.
Five retired embeds no longer name55c as a consumer; their other uses remain.
On the14:49:37 build, desktop1280×900 and mobile390×844 showed all three
images loaded, no page-level overflow and no KaTeX errors. Process counts,
left/right layout, captions and ragged offsets were visually inspected.
Public license URLs returnedHTTP200. Full-batch acceptance and revision
hashes will be recorded after the56 integration and final checks.
