# Next-token prediction — Russian editorial review

Date: 2026-10-13. Scope: Russian chapter
`00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md`, its
new source illustrations, and `publishing/tools/check_ntp_examples.py`.
The full original chapter and the full revised chapter were read, not only
searched for undesirable phrases. Existing unrelated edits were preserved.

## Content changes

The original chapter had useful MTP detail but a disproportionately short
explanation of ordinary next-token training. The revision expands the latter
without replacing the good independent-head and DeepSeek-V3 explanations.

- Chain rule versus independence; full sequence probability versus a prefix;
  BOS/EOS roles; finite available context; masked loss versus full likelihood.
- One consistently indexed example: `<bos> кот спит <eos>`, input-to-target
  table, shifted-loader versus shifted-loss contracts, and separate attention
  and target masks.
- Parallel teacher-forced forward pass versus sequential generation; why an
  observed prefix does not expose the target; bounded exposure-bias claims.
- Logits, softmax, negative log-likelihood, the `p - onehot` gradient, and
  stable log-sum-exp; the shared-parameter gradient caveat is retained.
- Perplexity as an exponential/geometric mean; explicit logarithm units;
  worked probabilities 1/2, 1/4, 1/8 with and without a masked final target;
  BPB and fixed raw bytes; context-window/stride and single target counting.
- NTP's relation to corpus content and downstream behavior, without implying
  that low NLL guarantees truthfulness or instruction following.
- Token count, microbatch weighting, gradient accumulation and logger
  differences; processed tokens versus unique corpus coverage.
- Executable masked PyTorch loss with one shift, an empty-target guard and a
  runnable numerical example. EOS and PAD roles cannot be inferred solely
  from a shared token ID.
- Concrete verification methods and primary-source reading paths, with
  previous chapter 41a and next chapter 43 links.

All previous H1–H3 headings remain, as do explicit anchors
`multi-token-training-objective` and `оптимизация`, and the source-unit ID
`lecture-04-deepseek-mtp`. No route or common navigation was changed.

## Source checks

1. Stanford CS336 Assignment 1, pinned revision
   `a158843b20107949f1a8d7df1b05cd33b9166712`, §4.1 / PDF pages 28–29:
   [cross-entropy, stable computation and perplexity](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf#page=28).
   Read the locally retained source PDF text, including the whole relevant
   section. It is not replaced with a screenshot.
2. Andrej Karpathy, `build-nanogpt` revision
   `6104ab1b53920f6e2159749676073ff7d815c1fa`: inspected the attention,
   `GPT.forward`, and `DataLoaderLite.next_batch` implementations. The loader
   shifts `x`/`y`, whereas the loss accepts those already shifted targets.
   [Pinned code](https://github.com/karpathy/build-nanogpt/blob/6104ab1b53920f6e2159749676073ff7d815c1fa/train_gpt2.py).
   Also checked [Zero to Hero](https://karpathy.ai/zero-to-hero.html) and
   [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/).
   The earlier unsupported attribution of a detailed NTP-specific diagnostic
   recipe to Karpathy was replaced with the directly supported one-batch and
   initial-loss checks.
3. Jurafsky and Martin, [SLP](https://web.stanford.edu/~jurafsky/slp3/) and
   [N-gram slides](https://web.stanford.edu/~jurafsky/slp3/slides/lm_jan25.pdf):
   chain rule and perplexity. Both old local slide images were visually
   checked; their clipped mathematical text is now typeset in the chapter.
4. D2L, [Language Models](https://d2l.ai/chapter_recurrent-neural-networks/language-model.html)
   and [GPT/GPT-2](https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#gpt-and-gpt-2):
   sequence shift and causal training illustration. Inspected original SVGs,
   page descriptions and final PNG renders; precise source revision and
   licensing recorded in the asset NOTICE.
5. Hugging Face, [causal language-model training](https://huggingface.co/learn/llm-course/chapter7/6)
   and [fixed-context perplexity](https://huggingface.co/docs/transformers/perplexity):
   collator/model shift conventions and context/stride dependence.
   These pages were used as sources, not copied wholesale. The test script
   explicitly counts scored positions after the shift; no untested remote
   example is presented as locally validated.
6. Gloeckle et al., [MTP v1](https://arxiv.org/html/2404.19737v1): checked shared
   trunk, separate transformer heads, shared output projection and sequential
   head backpropagation; retained the existing original Figure 1.
7. [DeepSeek-V3 v2, §2.2](https://arxiv.org/html/2412.19437v2#S2.SS2): checked
   true intermediate-token input, successive modules, prediction offset,
   loss weight and the published D=1 configuration. Kept the source link;
   no new figure was imported on an assumed arXiv reuse license.

The research workflow separated paper claims from illustrative arithmetic;
the PDF workflow was used to inspect the source mathematical presentation.
No new custom diagrams, translations of other chapters, or source snapshots
were produced.

## Illustrations and registry handoff

New directory:
`00 Учебник/Assets/Figures/curated/ntp-review-2026-10/`.
Full provenance, SHA-256 values, pinned raw URLs, license text and exact
rasterization procedure are in `NOTICE.md` there.

| Published asset | Dimensions | SHA-256 |
|---|---|---|
| `d2l-language-model-data.png` | 2000 × 292 | `96122632db141d7aed7457171a234e07b0d0cf944e6e5c4c0b0f169db04d0491` |
| `d2l-gpt-decoder.png` | 2000 × 1056 | `ab061b4721eac955985c274afaf476d9a4ee00aca2d9f6769bf9afdaa2c56d91` |

Both are complete rasterizations of original SVGs, not screenshots of pages
or newly drawn illustrations. The local original SVGs are saved alongside
them. Rights: CC BY-SA 4.0, as declared in the pinned D2L repository license.
All four D2L authors and the source page are credited in each chapter caption.

Remove only this chapter from the `used_in` arrays for
`curated/training-systems-courses/42-slp-chain-rule.png` and
`curated/training-systems-courses/42-slp-perplexity.png`.
Do not delete the old assets or other consumers. The original
`curated/course-follow-through/mtp-overview.png` remains in use.

## Executed evidence

Command, run from the worktree root:

```sh
/opt/homebrew/bin/uv run --with torch --python /Users/aleksan.potekhin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 python publishing/tools/check_ntp_examples.py
```

Passed with torch 2.14.1 on CPU:

- both actual Python fences extracted and executed from the chapter;
- NLL 1.3863 / PPL 4.0 and masked NLL 1.0397 / PPL 2.8284;
- equivalence to already shifted targets, double-shift negative control,
  boolean-mask and empty-target guards;
- uniform loss, constant-logit invariance, stable large logits, analytical
  gradient, zero gradient of excluded rows and EOS/PAD role distinction;
- deterministic causal-attention prefix invariance and a failing noncausal
  negative control, using a small CPU attention fixture;
- full-batch versus token-weighted microbatch loss **and shared-parameter
  gradient**, with incorrect equal-microbatch weighting as negative control;
- sliding-window target ledger and MTP valid pair counts 4/3/2;
- one-sequence bigram overfit through the chapter's loss:
  NLL 1.419184 → 0.003800 after 200 small optimizer steps.

Additional local checks: 13 Obsidian references outside code fences all resolve;
no pre-existing H1–H3 heading was removed; source-unit ID retained;
`git diff --check` passed for scoped changes. All three displayed figures
(two new D2L, existing MTP) were personally inspected.

## Limits and remaining integration

This is CPU arithmetic and toy-model verification, not training a pretrained
GPT, a distributed-training test, a GPU benchmark, or empirical confirmation
of published MTP improvements. No Hugging Face model was downloaded.

The main editor owns final registry changes, full publication/link tests,
build and rendered desktop/mobile inspection. Those checks are not claimed
here. English, common CSS/navigation/registry, other chapters, commits, push
and deployment are outside this subtask and were not modified or run.
