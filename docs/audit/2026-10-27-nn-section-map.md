# Network Architectures: reviewed section decisions

Date: 2026-10-27. Source pin `2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd`, SHA-256
`64ef2c6bab3a09846bdb089a33c0ee355ee32ecbf291447493708c276f18d9a6`, all 5256 source lines already read.

The table reconciles the earlier whole-chapter comparisons and the bounded
additions. It records where each source section's mechanisms are explained,
which source claims are qualified or rejected, and which current revision
was reviewed. It is not a second synthetic textbook chapter and does not
count a source link or a passing test as proof of explanation quality.
The camera-budget addition is accepted separately in
`docs/audit/2026-10-27-wildlife-budgets.md`; its arithmetic is conditional,
not a device benchmark or a battery-life result.

## Section decisions

Destination IDs resolve to the exact files and revisions below. Source
ranges partition the pinned QMD, including code, captions and editorial
sections; native objects have their own 24-object decision record.

| Source lines | Section | Destinations | Explanation retained or added | Claims not adopted as facts |
| --- | --- | --- | --- | --- |
| 1–55 | Front matter and purpose | source archive | Editorial framing is retained in the immutable source; it adds no separate model mechanism. | — |
| 56–318 | Architectural principles and engineering trade-offs | mlp, cnn, rnn, attention, practice | Local connectivity, shared parameters, temporal state and learned pairwise mixing have explicit forward/backward cases; the practice treats family choice as candidate generation followed by hard constraints. | No fixed total ordering of architecture quality or sample requirements. |
| 319–381 | Lighthouse models and profiles | practice, cnn, ranking | Named profiles illustrate distinct workloads; model parameter/activation/operator ledgers are separated in the destinations. | Historical catalogue values and illustrative utilization are not current device measurements. |
| 382–589 | Workload signatures and training/inference footprints | roofline, arithmetic, practice | Work, stored state, actual traffic, dtype and execution phase are separated; roofline and Transformer ledgers have worked numbers. | Family labels alone do not determine arithmetic intensity, peak residency or latency. |
| 590–1188 | MLPs: patterns, algorithm, mapping and implications | mlp, tensors, gpu | Affine map, parameter count, nonlinear composition, loss/backward and batched GEMM already have complete examples; Tensor/GPU chapters supply shapes, layout and tiling. | Finite-data quality and universal-approximation existence do not guarantee optimization or deployment performance. |
| 1189–1440 | CNN pattern needs and locality | cnn | Shared local filters, spatial axes/channels and their gradient sums are worked numerically; geometric equivariance is treated separately. | Translation equivariance does not imply arbitrary pose invariance; padding and stride qualify it. |
| 1441–1720 | CNN algorithm and group equivariance | cnn | Channels, padding/stride/dilation, pooling, exact C4 lifting, transformed feature fields and full original escnn animations are integrated. | Group-equivariant feature types are not Conv2d(groups); arbitrary rotations require resampling. |
| 1721–1845 | CNN lowering and memory mapping | cnn, gpu | Complete native window-row SVG, both input channels, all outputs 356/392/464/500 and transposed unfold explain lowering and duplicated entries. | Source row/column caption corrected adjacent; generic im2col 5–10x speedup not adopted. |
| 1846–1974 | CNN compute, movement and efficient variants | cnn, roofline | MAC convention, depthwise-plus-pointwise constraint, parameter counts and explicit traffic bounds are retained. | Fewer FLOPs do not ensure a proportional speedup on either mobile CPUs or data-centre GPUs. |
| 1975–1990 | Depthwise-separable CNN and keyword spotting | cnn, practice | The CNN explains spatial/channel factorization; the practice now gives its general cost ratio and distinguishes an always-on audio trigger from a visual species classifier. | Number of output classes alone does not establish representational capacity, and an architecture name does not establish whole-device power. |
| 1991–2233 | RNNs: recurrence and system implications | rnn | Scalar forward/BPTT, actual language-model training, input projections, chunks versus truncation, multilayer dependency grid and weight-gradient reductions are present. | Parallel input projection does not remove the hidden-state dependency or make generated future inputs available. |
| 2234–2929 | Attention: algorithm, mapping, memory and movement | attention, transformer, arithmetic | Finite Q/K/V example, scores/softmax, all projection gradients, permutation qualification, training/generation and materialized cost are explained; non-identifiability has a finite worked example. | Input-dependent dense weights do not require random addressing or a dynamic execution graph; square score storage is implementation-specific. |
| 2930–3458 | Transformers: blocks, training and inference | transformer, llmNorm, arithmetic, collectives | Encoder/decoder/cross-attention, masks/PAD, shifted loss, residual/norm/FFN, modern causal block and phase-specific memory/parallel ownership have detailed examples. | 2017 translator, modern decoder-only model, training activations and autoregressive KV state are not interchangeable. |
| 3459–3690 | Sparse embedding architectures and DLRM | ranking, collectives | Exact source interaction method, pooled IDs including empties/repeats, Gram order, BCE, gradients, table/row/column ownership and three original TorchRec figures are integrated. | Sparse access is not universally uncacheable or the sole bottleneck for every DLRM shape; GB and GiB remain distinct. |
| 3691–3715 | Shared building blocks | norm, llmNorm, ffn | Shared maps, skip paths, normalization and multiplicative gates are developed in their own chapters rather than duplicated as a second introduction. | — |
| 3716–3725 | Dense linear maps | mlp, ffn | Rows, shared token-wise weights, expansion dimensions and both projection gradients are explicit. | — |
| 3726–3741 | Skip connections and depth | norm, llmNorm, cnn | Identity target, compatible shapes, direct gradient branch and projection shortcut are worked. | — |
| 3742–3761 | Plain-network gradient products and ReLU | norm, mlp | Scalar and matrix derivative products, initialization assumptions, active/dead paths and cancellation are explicit. | Spectral radius bounds on individual changing Jacobians do not establish the asserted product bound. |
| 3762–3869 | Residual computation and deep-network evidence | norm, llmNorm, cnn | I+J_F, F=-x cancellation, post-addition ReLU, full pre/post Jacobians and executable ResNet-18 are retained. | A shortcut does not guarantee unchanged or nonzero gradients; historical evidence is not universal performance. |
| 3870–3948 | BatchNorm, LayerNorm and comparisons | norm, llmNorm | Axes, train/eval statistics, numerical examples, LN/RMS VJP, epsilon, shift-null direction and normalization placement are explained. | Universal Jacobian spectrum [0.5,2], fixed minimum batch size and inversion of arbitrary batch statistics are rejected. |
| 3949–3958 | Gating | ffn, llmNorm | Sigmoid/SiLU gates, both branch gradients, parameter-matched widths and local TP reconstruction have worked cases. | Dense gating is not sparse expert routing; a historical kernel timing is not universal LLM throughput. |
| 3959–4035 | Building-block synthesis | norm, llmNorm, ffn, roofline | The existing explanations connect composition to activation lifetime, recomputation, precision and kernel traffic. | No universal normalization speedup or optimization guarantee is adopted. |
| 4036–4176 | Core computational primitives | tensors, cnn, kernels | Contraction/broadcast axes, exact convolution windows, stable reductions and a full reduction tree with original animation have explicit cases. | Four-way fan-in art does not establish a reduction's work/depth or near-peak execution. |
| 4177–4240 | Memory access primitives | tensors, gpu, kernels | Transpose versus reshape, strides, counted sectors, banks/XOR, indexed operations and source contracts are represented. | Cache-sector counts are not unconditionally HBM traffic; model-family tags are not access-pattern proofs. |
| 4241–4434 | Data movement primitives | kernels, collectives | Gather/scatter with repeated indices, transpose gradients, rooted versus all-rank outputs and uneven all-to-all are worked; original source diagrams retained. | Local indexing and network collectives are different; no universal 10x gather penalty is transferred. |
| 4435–4559 | System impact of primitives | gpu, roofline, kernels, collectives | Tiling/fusion, neutral masks and synchronization, work/traffic bounds, stored state, streams and ownership are already explicit. | No universal energy ratio, fixed hardware utilization or peak-performance guarantee is adopted. |
| 4560–4565 | Architecture selection framework | practice, systems | Candidate family, required task quality and resource constraints are distinguished. | — |
| 4566–4575 | Data-to-architecture mapping | practice, cnn, rnn, ranking | Mappings generate candidates; all candidates re-enter hard constraint checks. | A data modality does not have exactly one admissible architecture. |
| 4576–4581 | Computational complexity | arithmetic, cnn, roofline, practice | Shapes, batch/context scaling, MAC versus FLOP, stored activation versus transfer bytes and operator-specific terms are retained. | One architecture-wide fitted exponent cannot replace an operator ledger. |
| 4582–4589 | Production requirements | practice, collectives, roofline | Latency/throughput, critical dependency paths, failures, model/activation/optimizer state and operating envelope are separated. | Fixed family efficiency ranges and mandatory square attention residency are not transferred. |
| 4590–4643 | Hardware mapping and optimization | cnn, gpu, kernels, llmNorm | Full Winograd 1D/2D derivation, channel GEMM, valid tiling, precision counterexample, GEMM tiling, fusion and scoped numerical formats are integrated. | 2.25 counts variable products, not time; no blanket FP16 impossibility or universal tile size. |
| 4644–4727 | Decision framework and original flow | practice | Complete native selection SVG, unchanged four-step English procedure and five checked constraints are integrated. | The flow suggests candidates rather than proving them deployable; a failed quality check cannot be waived by a score. |
| 4728–4735 | Inductive-bias hierarchy | practice, cnn, rnn, attention | Actual sharing, state and mixing assumptions are explained and the practice distinguishes assigned bias indices from learning curves. | No universal scalar hierarchy of architectural bias or guaranteed sample-count reduction is adopted. |
| 4736–4807 | Architecture in practice: throughput ceiling | practice, roofline | Work times event rate and conditional sustained service capacity are distinguished; 75 ms per event is compared explicitly with sequential 30-fps arrival. | Peak arithmetic headroom alone does not prove latency, sustained utilization, power or backlog stability. |
| 4808–4993 | Worked wildlife camera case | practice | Source constants reconstructed: partial memory 65.045056 MB, matched-unit conditional 75 ms, active energy 1.5 J/day; completed disjoint-state battery example separates idle draw and event rate. | Source floating/integer work mismatch, inferred 90% quality, capacity from class count, automatic GPU exclusion and arbitrary 10% future capacity are not accepted. |
| 4994–5215 | Fallacies and pitfalls | practice, roofline, cnn, attention, ranking | Complexity/quality, biased candidate selection, data requirements, state, traffic and hardware dependence are tied to concrete explanations and counterexamples. | Illustrative percentages and catalogue anecdotes do not become empirical findings by appearing in a recap. |
| 5216–5256 | Summary | practice, roofline, arithmetic | Existing chapter transitions and worked constraints provide the synthesis; the complete original recap stays available in the course archive. | No mandatory quadratic resident score matrix or fixed family bottleneck is adopted. |

## Current destination revisions

| ID | Canonical file | Reviewed SHA-256 | Review evidence |
| --- | --- | --- | --- |
| practice | `06 Практика/18a Сравнить архитектуры по ресурсным ограничениям.md` | `48c3d043fddf4bd75e57d79dfb7887d39d96bbdc784bd5be985b4c4038c601d8` | `docs/audit/2026-10-27-wildlife-budgets.md` |
| mlp | `00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md` | `17e763a94b26580e6c4d015c6ab092c1d33d57a9f6c77e34a598fdc27379507b` | `docs/audit/2026-10-27-module-hooks.md`; removing the independent hook-observation addition restores the complete 838-line module-state revision and its prior architecture explanation |
| cnn | `00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md` | `136cde481f65d808db9505969c6ef7fb8fee15fda1c760ebaf74b4d4d28b2b7f` | `docs/audit/2026-10-27-winograd.md` |
| rnn | `00 Учебник/04 RNN, LSTM и Seq2Seq/01 RNN и BPTT.md` | `9e7990cd4670623c0f8b4cc78fc233d30dfca3696143dee5ad01f87f2fa136b4` | `docs/audit/2026-10-26-rnn-execution.md` |
| attention | `00 Учебник/05 Attention и Transformer/02 Self-Attention — Q, K, V.md` | `073f6e573481b59e93f2a2bee4bdd2e86c7ab5e1c9091764e9c506bfd18acc3c` | `docs/audit/2026-10-26-attention-transformer.md` |
| transformer | `00 Учебник/05 Attention и Transformer/03 Полный Transformer.md` | `8bda8af63d82960da7d5849e132be647542b90d4a9a9daca05ff5a0f2e66d356` | `docs/audit/2026-10-26-attention-transformer.md` |
| ranking | `00 Учебник/02 Рекомендательные системы/03 Ranking и нейронные рекомендательные модели.md` | `86e293dab73545f32be8b908ab0b8a663e9b247f454223c93f44427e50e1f1cb` | `docs/audit/2026-10-27-dlrm.md` |
| norm | `00 Учебник/01 Основы нейронных сетей/05 Инициализация, нормализация и residual connections.md` | `019c8f6a0270bcdb4f282ade039108c289f5d7d391dde779e664fd046fa61435` | `docs/audit/2026-10-27-shared-blocks.md` |
| llmNorm | `00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual.md` | `ba4dde89572fa4c17be5addc0bde7ea24bbffb300120d15467c2282fb120ced6` | `docs/audit/2026-10-27-shared-blocks.md` |
| ffn | `00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md` | `861b4a9b9724ef3efb0446f09ae522b8117a71758c693dca64c5258c819a5b6a` | `docs/audit/2026-10-27-shared-blocks.md` |
| tensors | `00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md` | `7764790a11f7cbf12fbb9e662d62d56bd94ad8d39863a8995423f29161ff6ebb` | `docs/audit/2026-10-27-frameworks-native-figures.md`; removing only its bounded storage addition restores the entire previously accepted tensor chapter and its earlier review |
| gpu | `00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md` | `d432d06ee1687fca66fe6151028a1c4a2bb2313a7e23d9d2b7d877b96cb48d81` | `docs/audit/2026-10-27-tensors-gpu-primitives.md` |
| roofline | `00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md` | `ee3b591a0085982c18c2ae691889e38f2f624b505cb2a414520c947c1f57323b` | `docs/audit/2026-10-27-tensors-gpu-primitives.md` |
| kernels | `00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md` | `4fab5d0025274bd232fb3122c1152dc939cf8529a41389401c3d261e1db94a23` | `docs/audit/2026-10-27-tracing-control-flow.md`; the previous indexed movement/reduction/compiler explanations are preserved byte-for-byte outside the bounded branch-capture addition |
| collectives | `00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md` | `e1e9b7999ca10a61e9a48a24484f704b13cb7918cede4d5ba050b06876cbc8db` | `docs/audit/2026-10-27-collective-destinations.md` |
| systems | `00 Учебник/10 ML Systems/01 Модель как часть системы.md` | `c958fbd61b171e67163951269d76375e26376e7a54c69d76b4e3ad24c7b6676d` | `docs/audit/2026-10-26-architecture-selection.md` |
| arithmetic | `00 Учебник/10 ML Systems/04 Арифметика Transformer и MoE.md` | `2d49342ee5b2998d93980151ff95f2fdbf5a464c1cbb6911f9e7c3c4dce0cafa` | `docs/audit/2026-10-23-transformer-arithmetic.md` |
## Acceptance boundary and remaining work

The pinned H-V1-06 chapter now has a section-by-section thematic disposition,
including the previously missing Winograd and wildlife-budget explanations.
H-F1-06 has all 24 original objects individually reviewed and retained;
only suitable originals are reused in teaching. A factual source finding is
not hidden by restoring its rendering or by retaining its archive.

This is not full Harvard course completion. Lab/deck pins remain separate;
Marimo UI, hardware measurements and the full remaining Harvard/HSE/EDLS
units are not established by this chapter review. English parity is also
not accepted: self-attention and the complete Transformer have older separate
EN bodies, while other destinations use explicitly marked fallback. No EN
body is edited here.

`publishing/tools/check_nn_section_map.mjs` verifies the line partition,
source/destination hashes and the presence of revision-specific review
records. Its negative controls reject a missing range or stale destination.
That is an integrity guard for these manual decisions, not an automated
semantic-completeness test. Future content changes require reconciling the
relevant row and review, not merely refreshing the hash.

Next course work must be selected from unresolved units in the complete
transfer matrices; do not reopen accepted RL or repeat the already completed
5256-line source read. Broader source/rights findings and live publication
remain outside this local chapter acceptance.

