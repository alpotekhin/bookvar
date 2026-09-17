---
type: editorial
status: editorial
last_updated: 2026-09-15
---

# Course follow-through: FP4 and evaluation cases

Status: authored additions ready for independent review, not a claim that all four deferred source units are integrated. Existing EN57 parity repair is included. Shared maps, ledgers, hubs, registry and importer were not changed.

## Scope and reading

Fully read before editing and fully reread after editing:

1. 00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md.
2. en/00 Textbook/14 Inference and optimization/57 Quantizing language models.md.
3. 00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md.
4. 00 Учебник/18 Evaluation и методология/59a Responsible systems.md.

Navigation has no authored EN counterpart for 59 or 59a Responsible systems. No new translation/navigation entry was created. Independent reviewer /root/course_state reread EN57 and confirmed substantive alignment and coherent arithmetic. The requested EN57 last_verified correction to 2026-09-15 was applied afterward; its final hash appears below.

Pre-edit byte-preserving snapshots were created with apply_patch under .superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/course-fp4-eval-before/, retaining each full relative page path. They contain the task-entry text, not Git HEAD; unrelated pre-existing edits are preserved.

The llm-wiki workflow informed source/authoring separation and explicit partial coverage. Verification-before-completion required executed checks before readiness claims. No full-site build, git operation, new agent, new image or raw-course edit was performed.

## Source boundaries and outcomes

Frozen lecture commit: 8b59b50730766695c2ffedd1a79c50cd09b9eb91. Trace step numbers are one-based.

| Topic | Exact frozen trace | Authored outcome | Exact-unit status |
|---|---|---|---|
| FP4 | lecture_02.json steps 111–188; FP4 specifically 111–117, source lines 174–181 | RU already explained E2M1, MXFP4/NVFP4 scales/layouts/payload. Added two-block range-error example and qualified pretraining case. Existing EN also gained FP8 limits, format table, W4A16/W8A8 traffic ledger, rotation/fusion mechanisms and corresponding sources/practice. | FP4 narrative covered; oversized unit still source-only. |
| ARC interactive | lecture_12.json context 201–220; image step 215, line 277 | Added information-gathering actions, trajectory evidence, per-level action-efficiency example, compute-cost and transfer limits. Linked inspected frozen original screen. | Narrative added; image unit still source-only. |
| AIR-Bench | lecture_12.json steps 230–234; image step 233, line 298 | Added historical taxonomy, category-specific grader, refusal rate versus mean score, numerical example and usefulness/legal-claim limits. | Narrative added; image unit still source-only. |
| GCG | lecture_12.json steps 235–241; image step 239, line 305 | Added conceptual gradient-guided discrete search, search objective versus policy violation, budget/denominator example, held-out checks and historical-transfer limits. No attack suffix, script or execution. | Narrative added; image unit still source-only. |

The FP4 source unit also contains CPU/GPU transfer, einsum/reduce/rearrange, and FLOPs versus FLOP/s material. Those actual steps were read. A paragraph about FP4 cannot stand in for the whole unit.

### Primary-source checks

- [NVIDIA NVFP4 introduction](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) and [Transformer Engine 2.15](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.15/user-guide/features/low_precision_training/nvfp4/nvfp4.html): scaling and separate 1D/2D layouts.
- [OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) and [CUDA 13.0.2 E2M1](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-math-api/cuda_math_api/struct____nv__fp4__e2m1.html): format definitions.
- [Nemotron 3 Super report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf): accessed edition dated 3 April 2026; frozen reference says 11 March. The chapter names the read edition. An evolving PDF URL is not a pinned March artifact.
- [ARC-AGI-3 report](https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf): accessed edition dated 22 April 2026; lecture says March. The example avoids the upper-cap region and does not implement the complete aggregate scorer. The report's prose cap and displayed squared-min expression are not reconciled by this task; exact scorer reproduction requires a pinned implementation.
- [AIR-Bench 2024 v1](https://arxiv.org/html/2407.17436v1): taxonomy and category-specific grader. [GCG v2](https://arxiv.org/html/2307.15043v2): discrete candidate search and historical multi-prompt/model transfer.
- [SpinQuant v2](https://arxiv.org/html/2405.16406v2): rotation placement and Cayley optimization checked for EN parity.

### Visual preservation and remaining integration

All 18 existing embeds across the four pages remain. No original heading or source ID was removed. The EN page retains both original quantization figures.

Inspected the actual frozen originals, not just filenames:

- images/arc-agi-3.png: interactive game screen, level 2/7.
- var/files/image-5993188f3fa9dc78b85f9866fcee27ac-https_crfm_stanford_edu_helm_assets_air-overview-DpBbyagA_png: AIR taxonomy overview, not a performance chart.
- images/gcg-examples.png: historical model-response montage, not the search algorithm.

No matching entries for these filenames were found in 05 Источники/asset-registry.yml. New reproduction rights and publication/render evidence were not established. The archive originals remain untouched. No substitute diagram was generated. Their visual units must not become integrated solely because the topics now have prose.

## Exact coverage-overlay deltas for root

Only replace each listed reason under editorial-map.yml → coverage. Keep its existing disposition source-only and evidence unchanged. These are not new semantic units and do not increase integrated counts.

### lecture-02-section-111-fp4

Reason: FP4 narrative at trace steps 111–117 is covered in 00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md#fp4-mxfp4-и-nvfp4 and #общая-шкала-связывает-соседние-значения, with authored EN parity. The encompassing source unit also includes CPU/GPU transfer, einops and FLOPs at steps 119–188; those are not dispositioned by this FP4 follow-through, so the whole unit remains source-only.

### lecture-12-figure-step-215-rendering-1

Reason: Interactive-benchmark narrative is now covered in 00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md#arc-agi-3-действие-может-добывать-информацию. The frozen original screen was inspected and linked, but this image unit still lacks reviewed reproduction/registry/render evidence for canonical embedding.

### lecture-12-figure-step-233-rendering-1

Reason: AIR-Bench narrative is now covered in 00 Учебник/18 Evaluation и методология/59a Responsible systems.md#air-bench-покрытие-заданной-таксономии. The frozen taxonomy overview was inspected; this visual unit remains source-only until reproduction/registry/render evidence is completed.

### lecture-12-figure-step-239-rendering-1

Reason: GCG mechanism and adversarial-evaluation limits are now covered in 00 Учебник/18 Evaluation и методология/59a Responsible systems.md#gcg-статический-отказ-ещё-не-означает-устойчивость. The frozen historical-response montage was inspected; this visual unit remains source-only until reproduction/registry/render evidence is completed.

Root may update course-state narrative to distinguish completed topical exposition from pending exact-unit/visual integration. Generated artifacts must be regenerated by their owner rather than edited directly.

## Executed verification

Runtime: /private/tmp/bookvar-editorial-venv/bin/python, PyTorch 2.8.0, CPU. Native NVFP4 kernels, training, attacks and throughput benchmarks were NOT executed.

- FP4 nearest-code emulation: separate scales MSE 0; shared scale 16 reconstructed first values [0,0,0,8], MSE 0.2890625 over 32 values.
- Enumerated all 256 raw-byte encodings of each FP8 dtype. E4M3FN: maximum 448, minimum normal 0.015625, minimum positive subnormal 0.001953125. E5M2: 57344, 0.00006103515625, 0.0000152587890625.
- Payload ledger: BF16 weights 32 MiB; W4 codes 8 MiB plus 256 KiB scales; decode input 8 KiB; 256-token BF16 input 2 MiB; INT8 weights/input 16 MiB/1 MiB.
- Float64 QR rotation: output difference 1.5543122344752192e-15; orthogonality residual 4.440892098500626e-16.
- ARC toy level contributions 0.25/0.0625; AIR refusal rate 0.7 versus mean 0.8; illustrative unaided/attacked outcomes 0.05/0.25.
- npm exec vitest run tests/links.test.ts tests/check-built-links.test.ts from publishing: 2 files / 12 tests PASS. Unit tests are not a full-site render.
- Scoped filesystem checks: no missing slash-qualified wiki targets, no removed original embeds/headings, no trailing whitespace. Retained plain-basename links are not newly validated by this filesystem check.
- Four complete after-edit pages reread; the subsequent EN metadata-only date correction was reread separately. Full published desktop/mobile review remains the primary/reviewer gate.

### Reproduction of numerical checks

Run the following Python in the runtime above. These are illustrative calculations, not model-quality measurements.

    import torch
    levels = torch.tensor([0,.5,1,1.5,2,3,4,6], dtype=torch.float64)
    x = torch.zeros(32, dtype=torch.float64)
    x[:4] = torch.tensor([.5,1,2,6])
    x[16:20] = torch.tensor([8,16,32,96])
    def quant(x, s):
        z = x / s
        return levels[(z[:,None]-levels).abs().argmin(dim=1)] * s
    local = torch.cat([quant(x[:16],1), quant(x[16:],16)])
    shared = quant(x,16)
    assert torch.equal(local,x)
    assert ((shared-x)**2).mean().item() == .2890625
    for typ, maximum, normal, sub in [
        (torch.float8_e4m3fn,448,2**-6,2**-9),
        (torch.float8_e5m2,57344,2**-14,2**-16),
    ]:
        info = torch.finfo(typ)
        values = torch.arange(256,dtype=torch.uint8).view(typ).float()
        positive = values[torch.isfinite(values) & (values>0)]
        assert (info.max, info.tiny, positive.min().item()) == (maximum,normal,sub)
    N = 4096**2
    assert [N*2/2**20,N/2/2**20,N/128*2/2**10,
            4096*2/2**10,256*4096*2/2**20,N/2**20,
            256*4096/2**20] == [32,8,256,8,2,16,1]
    torch.manual_seed(0)
    X = torch.randn(3,8,dtype=torch.float64)
    W = torch.randn(8,5,dtype=torch.float64)
    R = torch.linalg.qr(torch.randn(8,8,dtype=torch.float64))[0]
    assert (X@W-(X@R)@(R.T@W)).abs().max() < 1e-12
    assert (R.T@R-torch.eye(8)).abs().max() < 1e-12
    assert (12/24)**2 == .25 and (12/48)**2 == .0625
    assert 14/20 == .7 and (14+4*.5)/20 == .8
    assert 2/40 == .05 and 10/40 == .25

## Hash ledger

Before hashes describe exact task-entry snapshots; after hashes describe pages at handoff.

    {"path":"00 Учебник/14 Inference и оптимизация/57 Квантизация языковых моделей.md","before_sha256":"80ca92ad901bbba52c290013ed2dee0404fa04138109d4537cd8f6c3824a63fc","after_sha256":"cda51ab3783d41beaf37a6cfcd94d0453126a2ecb367820f6a01935dd88abb15","before_lines":373,"after_lines":399,"embeds":2,"lostEmbeds":[],"lostHeadings":[],"whitespace":[],"unresolved_paths":[]}
    {"path":"en/00 Textbook/14 Inference and optimization/57 Quantizing language models.md","before_sha256":"4315a2ccf80f341c9b30361f6655cdfead23d2b1e996cc7fe95ee61a25175458","after_sha256":"048baec70fca25bea4f92043f12a818feafb3b8053f7165afa05c59ecbd5485e","before_lines":182,"after_lines":326,"embeds":2,"lostEmbeds":[],"lostHeadings":[],"whitespace":[],"unresolved_paths":[]}
    {"path":"00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md","before_sha256":"99ff867723b878f91f3d0c8256f5d9a4dcbc3a2f9a1694aac06e346f168f369a","after_sha256":"819a889699e5b9ea3757a1cdc661946b8396c724a55253b576f4e67e2b4883f5","before_lines":435,"after_lines":467,"embeds":7,"lostEmbeds":[],"lostHeadings":[],"whitespace":[],"unresolved_paths":[]}
    {"path":"00 Учебник/18 Evaluation и методология/59a Responsible systems.md","before_sha256":"a895d0782ac9af08b7935b14abaf1b513c67788fb767df8d50388ca864a5fadc","after_sha256":"072e39dead161f8e2d12305cabd692764b801815f03ed1532426ae49ccddde72","before_lines":286,"after_lines":344,"embeds":7,"lostEmbeds":[],"lostHeadings":[],"whitespace":[],"unresolved_paths":[]}
    {"trace":"lecture_02.json","sha256":"dea0edc07279ec3469c8d9f2e48165d5c0eca362c82b3243b5c33bc829505486"}
    {"trace":"lecture_12.json","sha256":"1f71bd19541b5902c35cc4ead4442ec6e25445e442ac746029558ad55259b35f"}
