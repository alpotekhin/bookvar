# Chapter 58a2: separate prefill/decode serving

Date: 2026-10-20. Canonical RU chapter:
`00 Учебник/14 Inference и оптимизация/58a2 Раздельное обслуживание prefill и decode.md`.

SHA-256: b7b7f12fe4726130c9d5d75697e7cda4d3314216fbb3a8195a474af77cb5efe4.

Local route: <http://127.0.0.1:4330/textbook/inference/disaggregated-serving/>.

## Scope and reading

The full original 460-line chapter was read before editing. The final 918-line
chapter was reread in order in two complete, untruncated parts after the
mechanism, wording and heading corrections. The review covers this chapter's
defined scope, not completion of the entire book or complete ingestion of all
linked courses.

Local deliverables include the canonical chapter, eight source snapshots,
one verbatim English source paragraph, three source-paper figure extractions,
their registry/notice, a CPU reference checker and this report. No English page,
commit, push or deployment was made. Shared original images and unrelated dirty
files were preserved.

## Explanation and source review

| Mechanism | Source scope read | Explanation and concrete check |
|---|---|---|
| Phase interference and isolation | DistServe sections 2.1-2.3 and 3.1-3.3; Figure 2 inspected on PDF page 5 | 10 ms decode versus 120 ms full prefill; four 30 ms chunks trade 40 ms token intervals against 150 ms prefill; toy schedule explicitly distinguished from mixed-kernel measurements |
| Pool structure and hardware placement | DistServe sections 4.1-4.3 and Figure 6; Splitwise section IV A-E; relevant section V methodology | Separate model copies, TP/PP and device counts; Splitwise's mixed pool and cluster/machine schedulers; total P+D resource cost rather than D-only speed |
| KV handoff and memory ownership | DistServe section 4.3 pull buffering; Splitwise section IV-C and implementation discussion; SGLang PD documentation | Pending first output is not yet in KV; 1 GiB / 25 GiB/s = 40 ms; TP2-to-TP1 axes, page IDs and compatibility; cancellation must not recycle a buffer with an in-flight DMA |
| Layer-wise transfer | Complete Splitwise section IV-C, including original source paragraph and Figure 11; Mooncake v1 section 4.2 | Four compute releases and a shared-link recurrence; 200 ms serial versus 140 ms streamed; even the fast-link case retains the final 10 ms tail |
| First-token delivery policy | Request path and phase dependencies from the above sources; explicitly constructed teaching schedule | P-first TTFT 120 ms / first ITL 70 ms versus D-ready 180 ms / 10 ms; both produce the second token at 190 ms |
| Prefix-aware routing | Mooncake v1 complete sections 3-6, including section 5 reread independently after an initially truncated combined output | Exact-prefix hash chains, queued local reuse versus cold compute versus remote fetch; explicit 140.96 / 163.84 / 70.96 ms cases; slower network changes the winner |
| Admission, reservations and delayed load | Mooncake v1 section 6; SGLang decode allocation policy, restrictions and cleanup sections | 16384-position budget includes 4096 already promised positions; late versus early physical allocation; system-level future-load forecast is not exact output-length prediction |
| Capacity and queueing | DistServe sections 3.1-3.3 and equations 1-3; Splitwise section V and Table VI | D produces the remaining E[O]-1 outputs, not the first token already selected on P; 5P + 2D at TP2/TP4 means 18 GPU; average-flow lower bound does not guarantee queue stability; M/D/1 TP2/PP2 worked means at 4 and 7 RPS |
| Joint SLO provisioning | Complete Splitwise Figure 12 and Table VI with experimental setup | All nine TTFT/TBT/E2E percentile constraints, uncontended DGX-A100 reference and the specific 27P/3D optimum; not a universal hardware ratio |
| Chunked pipeline prefill | Mooncake v1 section 4.1 | Two stages and two input chunks with layer-local KV; causal dependency explanation and two-attention-layer NumPy example; source SP terminology distinguished from Megatron norm/dropout SP |
| Modern backend protocol | Complete Dynamo v1.5.1 architecture/router articles; SGLang PD intro through decode-policy restrictions and routing/bootstrap/cleanup | Backend-specific metadata, direct KV data path, KV-ready barrier, disabled decode-prefix assumptions and no implicit aggregated fallback after P dispatch |

The full Mooncake v1 sections 4 and 5 were read separately to recover text
omitted by the initial combined tool output. No inference is made from the
unread content of newer Mooncake revisions. The source's wording of the transfer
versus recompute comparison is not copied as an inequality: the chapter's
explicit candidate-cost example is a teaching model, not a claim to reproduce
every branch of Algorithm 1.

Dynamo's old design-doc URL returned 404. The current official index was used
to locate the two versioned v1.5.1 articles. SGLang's documentation is scoped to
the 20 October 2026 snapshot; its late-allocation restrictions are not presented
as universal support across TP/CP/PP.

## Figures and provenance

Six source figures are embedded. The two inherited DistServe figures and the
inherited Mooncake Figure 1 remain byte-for-byte unchanged. Their existing
registry rights metadata is preserved; this review does not claim a new legal
clearance for those inherited assets.

Three new Splitwise v2 figures retain all original panels, labels, scales and
captions: Figure 10 architecture, Figure 11 serialized/streamed handoff, and
Figure 12 nine-panel provisioning. Both complete PDF pages were visually
inspected, followed by the selected crops and the actual rendered chapter.
The source PDF is version-pinned; the abstract-page license link was checked:
CC BY-NC-SA 4.0. Attribution and the same license accompany the extracted assets.

Final exports use pdftoppm at a 6000 px maximum page dimension, not bitmap
upscaling. The complete figure crops are 1780 x 1880, 2090 x 930 and
2090 x 1950 px. Exact coordinates and source hashes are in the adjacent
assets.json. An initial lower-resolution export was superseded after browser
inspection of small labels. The final checker reproduces all three crops
pixel-for-pixel from the versioned PDF.

The Harvard PD graphic was removed from the narrative because its fixed
migration timing and HBM/network labels do not support the mechanism being
explained. The oversized, unversioned Dynamo overview was replaced with a
versioned protocol table. No shared original was deleted; the Dynamo asset's
English consumer remains registered.

The Splitwise layer-transfer paragraph is preserved in its original English
rather than translated. The checker compares its three sentences with the
source TeX, permitting whitespace-only normalization. The original attribution
and CC BY-NC-SA notice are adjacent.

## Separate style review

The no-ai-slop skill and eval were read. The entire finished chapter was checked
for generic setup, mixed-language filler, unsupported universal claims, empty
contrasts and definition-only sections. The retained technical terms identify
actual mechanisms and APIs. Heading wording was simplified, while explicit old
anchors preserve inbound links. No source quotation, source label, formula or
code identifier was rewritten for style.

The check is separate from mechanism acceptance. The chapter retains worked
intermediate states, units, assumptions, caveats and examples; it is not
shortened to a glossary.

## Executed verification

The chapter checker executes its actual ten Python blocks, then verifies:
200 timestamp cases, 1363 TP layout conversions, 500 layer-transfer schedules,
300 first-token-policy cases, 200 prefix routing cases, 500 reservation cases,
300 queue-formula cases, 400 chunked pipeline calculations and 400 post-handoff
continuations against independently looped full causal attention. Negative
controls cover invalid time arrays, equal byte counts with wrong axes,
forgotten reservations and repeating the last prompt input.

It also checks the displayed byte/capacity/goodput arithmetic, the joint-SLO
counterexample, eight snapshot hashes, one exact source excerpt, six figure
hashes, three independently reproduced PDF crops, retained heading anchors and
the previous/next chapter links.

These are CPU/reference/source-fidelity tests. No GPU, serving engine,
RDMA operation, measured network transfer, performance benchmark or quality
evaluation was run. The hypothetical numerical examples are labeled as such.

The local publication build succeeds: 1961 pages. Asset-registry tests:
38 passed. Built-output tests: 12 passed. Built-link check: zero broken internal
routes, fragments or files. Targeted git diff --check: clean.

## Browser evidence and limitations

Actual viewports: 1365 x 900 and 390 x 844. All six final images load with the
expected original dimensions. Figure boundaries, captions and surrounding
explanation were visually inspected. Six display equations render without
KaTeX errors or horizontal equation overflow. Mobile document width is 375 px
inside the 390 px viewport; content width is 343 px. Code and tables retain
horizontal scrolling.

Wide multi-panel diagrams have small labels inline on a phone. Opening the
layer-transfer image was verified to reach the actual 2090 x 930 PNG, not a
thumbnail. This limitation remains; successful image loading is not described
as full inline mobile label legibility. No new source artwork was invented.

Proof images:
- /tmp/bookvar-pd-desktop-2026-10-20.png
- /tmp/bookvar-pd-mobile-2026-10-20.png

The desktop proof was embedded in the conversation. Temporary viewport overrides
were reset. A local preview tab was retained as a deliverable.

## Remaining work

No unresolved mechanism gap was identified within this chapter's reviewed
scope. Mobile reading of fine figure details still requires the original-image
link. No fresh rights clearance is claimed for inherited figures. The English
counterpart and live deployment remain unchanged. The larger textbook goal
stays active. Next page: 58b, benchmarking, SLO and inference operations, followed
by 58c queueing/capacity planning; those pages are not accepted by this report.

