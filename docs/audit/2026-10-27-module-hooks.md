---
status: content-and-local-ui-reviewed
chapter: 00 Учебник/01 Основы нейронных сетей/01 Нейрон и MLP.md
route: /textbook/neural-networks/mlp/
before_sha256: 5e7aa6cd8ffc7af8ff995943582f13cbcc59c784ae2399801c16b0ee4f7547e0
after_sha256: 17e763a94b26580e6c4d015c6ab092c1d33d57a9f6c77e34a598fdc27379507b
---

# Module hooks: which derivative is being observed

This is the concrete missing mechanism identified by the Frameworks section
map, not a new rewrite of the accepted MLP explanation. The complete 838-line
chapter was read before editing. The independent addition preserves every
byte of that accepted revision: removing it recovers SHA 5e7aa6cd…547e0.
Final length is 1053 lines. All four source figures, seven original runnable
blocks, XOR reasoning, training example, state/serialization and FFN remain.

## Source and explanation comparison

Read the complete Harvard lst-module_hooks example and its surrounding
module/platform text, including the source's original English comments.
The whole code fence is reproduced unchanged with authorship, pinned link
and CC BY-NC-SA 4.0. Source commit is 2bd97c509923dc8d7cb0b3e2f489a7282fee5fbd.
The decorative three-framework margin graphic is not an explanation of hooks;
it was not substituted for a mechanism figure.

Read the complete register_full_backward_hook contract and implementation,
plus the complete register_forward_hook contract and registration body in
the pinned PyTorch module.py, lines 1465–1561 and 1692–1808. Inspected the
following call-wrapper path for context. This is not a claim to read all
of module.py. The existing source and BSD license bytes are unchanged.

The source's first observed Linear has output shape [32,20], while final y
is [32,5]. The text explains why y.sum does not give an all-ones derivative
at that earlier layer: the second weight and ReLU mask transform it. The
source's activation mean/std mix batch and feature axes and cannot establish
each feature's behavior.

The numerical case reuses the state example's initial weight [[2,-1],[1,3]]
and bias [.5,-.5]. Input [1,2] gives [.5,6.5] and loss z1+2z2=13.5.
Output derivative [1,2], input derivative [4,5] and weight derivative
[[1,2],[2,4]] are derived separately; their norms are sqrt(5), sqrt(41)
and 5. The bias [1,2] is tied to the one-example case, not a universal identity.

A complete observation function records detached clones and removes handles
in finally. All four input/parameter grad-flag combinations are explained in
a table. eval, no_grad, removed handles and direct-forward bypass each have
actual assertions. Positional-input tuples are distinguished from module
parameters and keyword arguments. The output-side firing warning observed
in PyTorch 2.14 is explained rather than suppressed. Hook return values can
modify behavior; these observers return None. In-place view restrictions and
recording overhead remain explicit limits.

No-ai-slop review preserves the source language, complete code, numerical
derivation, caveats and useful table mappings. It does not turn hook APIs
into an isolated list of definitions or assert that observation is free.

## Executed evidence

publishing/tools/check_module_hook_examples.py extracts and executes all four
new printed fences, asserts exact original Harvard code and the whole inverse
838-line baseline, then independently checks eight grad-flag/evaluation cases
and twelve differently shaped matrix cases. It reconstructs the original
Sequential first-layer derivative through the later weight and ReLU mask,
and verifies that weight-grad shape differs. Direct forward bypass, cleanup,
no-grad and an invalid in-place mutation of the backward-hook view are checked.

The older MLP checker excludes only the independent hooks addition, retaining
all its seven original fences and existing source, XOR, CE/SGD, state, container,
checkpoint, finite-difference and FFN assertions. The first runs of both
checkers passed; a subsequent prose-only warning clarification and table
layout changes changed the chapter hash. Final-SHA execution and rendered
evidence are recorded below. CPU/PyTorch 2.14.1 only; no CUDA, distributed, compile or
performance claim is made.

## Final revision and rendered evidence

The final-SHA hook checker passed all four fences, eight flag/mode and twelve
independent matrix cases. The old seven-block checker passed before the final
table-only changes; those changes do not alter its extracted code or inverse
baseline. Full unit suite before the last layout changes: 572 tests in 82
files passed. After the final changes, all twelve affected hook/module-state/
NN-map/Frameworks-map tests passed again. The final 1969-page site build took
23.10 seconds; 150 built-output tests passed and built links reported zero
broken routes, fragments or files. Structural source coverage passed; this
does not prove all source mechanisms have been integrated.

The whole 838-line pre-change chapter and the complete new addition were
read, then the warning and both final table reflows were read in context.
No original body, equation, code or figure was removed. The first mobile
render exposed widths of 424 and 376 pixels in 328-pixel table viewports.
Shorter headings, a two-line matrix cell and an explicit norm definition
retain the information. Both final tables are 328/328 at viewport 360×800,
with all four flag combinations visibly inspected together. Body is 360/360.
All 19 display equations and inline math fit; no KaTeX errors or duplicate IDs.
All four original figures load. Twelve pre blocks include eleven Python
fences and the existing printed-model block; widths stay inside 326 pixels.
The long new flag-case fence was actually scrolled to its 566/566-pixel
horizontal limit. This is not merely a claim from CSS overflow settings.

Desktop 1280×900 shows the full three-derivative explanation and table in
context. The actual EN route has the new sections and the explicit
untranslated-content notice; no English body or accepted translation was
created. Screenshots in docs/audit/screenshots/:

- 2026-10-27-hooks-derivatives-desktop.png
- 2026-10-27-hooks-derivatives-mobile.png
- 2026-10-27-hooks-flags-mobile.png
- 2026-10-27-hooks-all-flags-mobile.png
- 2026-10-27-hooks-en-fallback.png

Logs: /var/tmp/bookvar-module-hooks-{examples,scoped,unit,build,output,links,
coverage}-final.log and bookvar-module-hooks-old-final.log. The published
transfer-matrix update was included in the final build, not left as a stale
generated page. The NN and Frameworks maps bind the final chapter hash with
inverse-preservation evidence; hook acceptance is distinct from the pending
platform comparison in the same source interval.

## Remaining overall scope

Broader
platform/deployment/selection, JAX runtime and native figure candidates remain
open. No EN body, immutable source, accepted RL or git publication was changed.
