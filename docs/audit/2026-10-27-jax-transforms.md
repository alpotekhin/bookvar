---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/01 Основы нейронных сетей/01 Нейрон, градиент и backpropagation.md
route: /textbook/neural-networks/backpropagation/
before_sha256: f4a436da9642cc59dfec871c9e2134119871e81e7993b0cc2dcd68af68d31b82
after_sha256: 8795146abe1f9c7a8de8a232c958aae472e3788d963f99aba557b90690fee2c0
---

# Individual gradients, function transforms and one functional update

The preceding RL answer verified already-accepted chapter hashes. It was a
status answer, not progress on the remaining Frameworks work. This pass
continues the actual pending JAX addition; it does not reopen PPO/DPO/RLVR/GRPO.

## Reading and preservation

Read the complete expanded chapter in consecutive ranges 1–450, 451–1003
and 1004–1364. Then clarified the first new paragraph: grad differentiates
the first argument by default and requires a scalar output. Read that final
paragraph in context and on the rendered desktop page. Final length is
1366 lines. No accepted explanation was shortened into a definition list.

Removing the one independent bounded JAX addition recovers every byte of the
accepted 1193-line AD revision, SHA f4a436da…31b82. The older inverse-baseline
helper still recovers the complete 784-line chapter, SHA d329325d…a198.
The 15 older Python fences, all seven source figures, equations, captions,
anchors and chapter navigation are unchanged. There are now 20 Python fences:
19 runnable blocks and the original non-standalone micrograd class fragment.

## Source comparison

Harvard ML Systems Frameworks is pinned at
2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd. The complete source chapter was
already read in the preceding Frameworks pass. Revisited platform analysis,
especially JAX and the full lst-jax-transformations example, lines 3993–4123.
The entire original Python example and English comments are reproduced
unchanged, with adjacent Harvard authorship, permalink and CC BY-NC-SA 4.0.

Read the complete 549-line official JAX Transformations guide pinned to
dc4af684446e65ee98eee8090e1ff6f8211473fb, tag jax-v0.12.0.1. Compared
scalar-valued grad, default argnums, vmap/in_axes, per-example gradients,
tracing and composition limitations. Read the complete short PyTorch
torch.func introduction pinned to 5c4886908584029761b579af026dcfb627c84070,
tag v2.14.1. The JAX license was read in full. Exact file hashes and reading
boundaries are in the source archive README. The separate JAX cookbook was
archived but not read in full or claimed integrated.

The source's exclusivity claim for JAX function transforms is not repeated:
the printed torch.func example demonstrates the same mathematical operation.
Nor are the source's universal elimination of Python overhead or automatic
multi-device communication accepted. The new section teaches the selected
gradient composition; it does not claim full JAX distributed/runtime coverage.

## Explanation and executed examples

The complete original definition-only example is followed by concrete data
and intermediate quantities. Parameters w=(1,-1), b=.5 produce predictions
(-.5,-.5), errors (-1.5,.5), losses (2.25,.25) and mean loss 1.25.
Individual weight gradients are [(-3,-6),(3,4)] and bias gradients (-3,1).
The parameter dictionary and vmap example axis are tied to an explicit
four-row leaf-shape table. The mean gradient is (0,-1) and bias -1.

A functional SGD step at .05 returns w=(1,-.95), b=.55 without mutating
the original tree. Its predictions (-.35,-.25) and mean squared loss 1.1925
are derived in the text and asserted in the actual printed code. Independent
examples are required for the gradient-mean equivalence; current-batch
BatchNorm is an explicit counterexample to replacing a batch function by
independent calls.

Changing targets from shape (2,) to (2,1) creates a 2×2 broadcast error array.
The erroneous objective still has initial value 1.25 but weight gradient
(-2,-3). The negative control therefore checks more than scalar-loss agreement.
The torch.func.grad/vmap version reproduces both individual and aggregate
results and leaves .grad empty. JIT results agree with the uncompiled CPU
calculation. No timing, CUDA, accelerator speedup or multi-GPU claim is made.

publishing/tools/check_jax_transform_examples.py extracts and executes all
five new fences in order, checks the exact Harvard code and the inverse
chapter baseline, then independently compares NumPy analytic gradients with
JAX grad/vmap/JIT and PyTorch grad/vmap for 12 batch/feature combinations.
It also checks wrong-shape broadcasting and an erroneous update that would
introduce a parameter batch axis. Final execution passed on the SHA above.
The older checker was also rerun on that final SHA: its five baseline
runnable blocks and nine AD additions passed, including 81 adjoint identities,
nine HVP cases, broadcasting and first-/second-order custom-rule checks.

The first extra-case run failed because a Python-float bias became float32
in PyTorch while NumPy arrays retained float64. This was a checker defect;
the printed chapter code already specified float64. The checker now fixes
and asserts every PyTorch input dtype. Final versions: JAX/JAXlib 0.12.0.1,
PyTorch 2.14.1, CPU, JAX x64 enabled. Numerical agreement is not a general
claim about all operators, random-number streams or backends.

## Style, figures and rendered acceptance

Applied no-ai-slop without rewriting licensed source comments or removing
the numerical explanation. Necessary contrasts, terminology and mathematical
notation remain. The original source language is preserved. Seven accepted
figures are byte-preserved by the inverse-baseline check and all seven load
in the final browser page. No generated teaching diagram was substituted.

Actual desktop at 1280×900 shows the source attribution and unchanged code.
At 360×800 the complete new leaf-shape table and its four result rows fit:
328/328 px, including both right columns. The numerical formula and its
surrounding derivation were inspected in a separate mobile view. All 30
display equations and inline formulas fit; KaTeX errors and duplicate IDs
are zero. Body width/scrollWidth are 360/360. The EN route has the new section
and an explicit untranslated-content notice; it is not an accepted translation.

Screenshots in docs/audit/screenshots/:

- 2026-10-27-jax-desktop.png
- 2026-10-27-jax-shapes-mobile.png
- 2026-10-27-jax-gradient-mobile.png
- 2026-10-27-jax-en-fallback.png

## Build evidence and remaining scope

566 unit tests in 80 files passed. The final site build produced 1969 pages
in 23.21 seconds. All 149 built-output tests passed, including the root and
explicit EN-fallback checks for this addition and the retained older AD tests.
The built-link checker reported zero broken routes, fragments or files.
Source coverage passed, but its counters are structural, not proof of full
course integration. Logs are /var/tmp/bookvar-jax-{unit,site-build,output,
links,coverage}-final.log and bookvar-jax-transforms-final.log.

H-V1-07 remains partial. The full Frameworks section map, platform/deployment
and framework-selection comparison, the three held native figure candidates,
and source/rights findings remain separate open requirements. No source bytes,
English-locale bodies, accepted RL chapters, git publication or live
infrastructure were changed. The overall textbook goal remains active.
