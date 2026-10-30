# Chapter 44c: tensor, sequence and context parallelism

Date: 2026-10-13. Scope: canonical RU chapter, one CPU reference checker,
three original upstream images and their local NOTICE. No EN changes,
publication build, commit, push or deployment in this subtask.

## Editorial and mathematical changes

The complete previous 221-line chapter was read, not pattern-replaced.
The chapter now follows one two-rank MLP through column and row splitting,
forward and backward, before extending the same reasoning to SwiGLU,
attention, SP and long-context attention. Important clarifications:

- Column concatenation versus row summation; PyTorch's transposed weight
  storage; the second matmul consumes the first projection's output.
- Nonlinearity placement explains why paired projections save a collective.
  A replicated output bias is added once, not once per partial sum.
- A logical replicated gradient is not summed a second time. Weight shards
  own different parameters, while input-gradient pieces must be summed.
- SwiGLU columns must align across both input projections; both local
  input-gradient branches can be combined before their all-reduce.
- Whole-head TP, GQA/MQA KV replication and aggregation of gradients only
  for copies of the same KV head. Splitting inside a head is not the same
  algorithm.
- SP preserves the full feature dimension for RMSNorm/LayerNorm. Its
  replicated norm parameters need the SUM of contributions from different
  token rows; independent DP averaging is a separate operation.
- AG/RS adjoints, residual/dropout placement, transient gather memory and
  extra backward AG when only the local normalized input was saved.
- Communication totals are outgoing bytes for a specified ring algorithm,
  not message-buffer size or measured NCCL runtime. The base SP boundary
  volume equals TP here, but regather and startup costs are not omitted.
- Ulysses is an explicit sequence/head permutation; forward traffic counts
  three QKV tensors and one output, not six tensors. MHA assumptions are stated.
- Ring uses global masks and online-softmax rescaling, including empty
  masked blocks. It does not average independent block-softmax outputs or
  remove full attention's quadratic arithmetic.
- TP×SP reuses one group. TP×CP has independent axes; GQA may prevent a
  second uniform Ulysses head split. FSDP groups gather TP-local weights,
  and EP is not automatically an extra independent world-size multiplier.

All five `source_unit_id` values and the explicit course anchor are retained.
Renamed old headings retain explicit anchors. Original chapter filename,
previous/next chapter links and source-course links are unchanged.

## Sources actually consulted

- Megatron-LM, arXiv:1909.08053v4, §3, paired MLP and attention projections,
  forward/backward conjugate operators.
- Korthikanti et al., arXiv:2205.05198v1, §4 and Figures 4–6: SP boundaries,
  activation accounting, and backward regather for the first weight gradient.
- CS336 Spring 2026 `lecture_08.pdf`, pinned
  `8b59b50730766695c2ffedd1a79c50cd09b9eb91`, PDF pages 39–54; local PDF text
  was read for width-parallel arithmetic, memory and composition of axes.
- Ulysses, arXiv:2309.14509v2, §3: partition permutation and forward exchange.
- Ring Attention, arXiv:2310.01889v4, §3 and Algorithm 1: fixed local queries,
  cyclic KV movement, blockwise attention and backward/memory caveats.
- PyTorch 2.14 tensor-parallel API: ColwiseParallel, RowwiseParallel,
  SequenceParallel and its local-input/parameter-initialization contract.
- HF Ultra-Scale Playbook, pinned `c93890dd4b22356ee18ed9b863e6f2d0909729bc`,
  original `src/index.html` TP/SP/CP sections and selected artwork.

Source narratives are not adopted indiscriminately. In particular, the
Playbook's broad claim about maximum SP activation size is bounded here by
the actual transient full all-gather; its statement about why the gather is
needed is explained as the requirement to process every token with every
TP weight shard, not as a missing hidden dimension in an SP row.

## Figures and rejected candidates

The chapter has four inline source figures, all personally inspected:

1. Existing Harvard `ml-systems/harvard/distributed/tensor-parallel-split.svg`:
   unchanged, with clarified caption explaining logical concatenation and
   already-replicated input. Existing CC BY-NC-SA 4.0 provenance stays intact.
2. New HF `hf-attention-tp.png`: original full attention-head diagram.
3. New HF `hf-tp-sp-regions.png`: complete TP versus TP+SP comparison.
4. New HF `hf-causal-zigzag.svg`: original vector causal mask; the caption
   enumerates row ownership and verifies 34 allowed pairs per rank.

New files are in `00 Учебник/Assets/Figures/curated/tensor-sequence-review-2026-10/`.
PNG pixels were inspected directly; SVGs were inspected using Quick Look
rendered previews. No image crop, rasterization or handwritten diagram was
added to the chapter. These inspections do not claim final site-layout QA.

Rejected: HF `tp_diagram4.png` has inconsistent second-projection numbers
(for example, 20 multiplied by the displayed 10 does not give its displayed
0). It was downloaded for inspection and removed, not embedded. The zoomed
SP diagram was also replaced with its complete two-panel original.

The old CS336 `systems/l08-p47.png` full-slide embed was removed from this
chapter, but its equation, variable definitions, source link and limitations
remain. Do not delete the shared image or other chapters' references.

## Executed evidence

Command:

```sh
/Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 publishing/tools/check_tensor_sequence_examples.py
```

NumPy 2.3.5, float64, one CPU process. Ten check groups passed:

1. Written column/row example, two-projection forward `[2,12]`, input
   gradient `[4,5]`, negative controls for bias and pre-reduction nonlinearity.
2. Actual chapter Python block with TP=1/2/4; forward equivalence and
   central finite differences for every element of X, A and B.
3. SwiGLU forward/gradients and finite differences for X and all three weights.
4. SP all-gather/reduce-scatter adjoint identities, MLP row-sharded output
   and input gradients.
5. RMSNorm X/gamma finite differences, SUM versus incorrect average,
   incorrect hidden-local norm negative control, consistent dropout/residual.
6. MHA TP output projection and GQA replicated-KV gradients, including finite
   differences for the genuinely shared KV tensors.
7. Ulysses actual split/concatenate sends with forward and gradient permutations.
8. Ring c=1/2/4, causal/noncausal outputs, masked empty blocks, large logits
   and finite-difference Q/K/V gradients versus dense attention.
9. Written memory/communication totals, TP×CP head constraints and all
   16 zigzag-mask row counts.
10. Retained course IDs, anchors, written numerical fixtures and four
    existing figure files.

No torch.distributed/DTensor process group, GPU, NCCL, kernel validation,
throughput or peak-device-memory measurement was performed. Python examples
use NumPy; system `/usr/bin/python3` here has no NumPy, so the bundled runtime
was used. Site rendering and shared registry integration belong to the parent
integration pass, not this test result.

## Metadata proposals for parent integration

Shared registry was not edited. For all three additions:

- `used_in`: `00 Учебник/11 Pre-training и Scaling/44c Tensor и sequence parallelism.md`
- `author`: Nouamane Tazi, Ferdinand Mom, Haojun Zhao, Phuc Nguyen, Mohamed
  Mekkouri, Leandro von Werra, Thomas Wolf; Hugging Face
- `commit`: `c93890dd4b22356ee18ed9b863e6f2d0909729bc`
- `license`: `Apache-2.0`
- `license_url`: `https://huggingface.co/spaces/nanotron/ultrascale-playbook/blob/c93890dd4b22356ee18ed9b863e6f2d0909729bc/README.md`
- `metadata_status`: `verified`
- `provenance_confirmation`: `pinned-upstream-repository`
- `last_verified`: `2026-10-13`
- `modifications`: Original upstream file copied byte-for-byte; no cropping,
  redrawing, re-encoding or translated labels.

Use `source_url` = pinned HF `blob/<commit>/assets/images/<original>` and
`source_image_url` = pinned HF `resolve/<commit>/assets/images/<original>`.

| Local name | Original name | Derivation | SHA-256 |
|---|---|---|---|
| `hf-attention-tp.png` | `tp_full_diagram.png` | `source-raster-copy` | `01f9fd3bc4b0a97b167d6ce2d47b0a207492441592d1e2a51fab1e8bfad9962e` |
| `hf-tp-sp-regions.png` | `tp_sp_diagram.png` | `source-raster-copy` | `d2463f346a6a3e16d447329a94eb8e9120d38effacf1230637ca25cd35d4c250` |
| `hf-causal-zigzag.svg` | `cp_zigzagmask.svg` | `source-svg-copy` | `8b8894a769d91f542743f7757122a0c0e8987d1fac3cbb4ddaca0f398a35ab0f` |

Remove only chapter44c's `used_in` from the old CS336 l08-p47 record.
Keep its other uses and the Harvard figure's chapter44c reference.
