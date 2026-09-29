# Sequence-model alternatives: page review

Date: 2026-09-28. Scope: the canonical Russian chapter on SSM, Mamba,
RWKV, RetNet, linear attention, TTT and hybrid models, its source figures,
and two Stanford Lecture 4 visual-review records. This is not acceptance
of the whole textbook. No commit, push, or deployment is part of this review.
No English translation was authored.

## Text and mathematics

Chapter: `00 Учебник/09 Dense FFN и Mixture of Experts/03 Mamba, RWKV, RetNet и гибридные архитектуры.md`.
SHA-256: `fdadb7d3bba105cde4f8cc2cba739f842e8c3293ae7cef86af208c3c4c4d489e`.

The chapter was read in full before editing and again in full during review.
Two independent reviewers checked mathematics and model/system claims.
Their findings were compared with primary papers, model documentation and
reference implementations before incorporation. Later prose changes were
reviewed as individual passages, not inferred from keyword searches.

Corrections and expanded explanations:

- Bahdanau attention uses the generated prefix to query encoder states;
  the decoder does not first know the next word.
- Mamba-1 distinguishes shared continuous A from input-dependent discretized
  transitions, and exact frozen-on-step discretization from the implementation's
  simplified input term. A numerical example separates decay from writing.
- Associative scan preserves token order. The two affine-map orders are
  evaluated explicitly.
- Mamba-2 uses a consistent N-by-P state, column-vector convention and a
  scalar transition per head. The direct branch and discretization factors
  are stated explicitly.
- Mamba-3 MIMO increases input/output rank and arithmetic at the same state
  size. Similar decode latency is restricted to the reported memory-bound
  measurements, not generalized to arbitrary workloads.
- Linear attention distinguishes all-position reassociation from causal
  prefix sums. Positive features, normalization, repeated-key averaging and
  a numerical counterexample to overwriting are explained.
- Delta rule replaces an association along a normalized key while preserving
  orthogonal directions, not all other associations.
- TTT distinguishes the illustrative sequential update from the paper's
  mini-batch gradients and separates TTT layers from TTT-E2E.
- Hybrid interleaving and parallel branches are distinct mechanisms;
  a repeating layer schedule is a subtype of interleaving. Model-specific
  layer counts are not presented as controlled architectural comparisons.
- MiniMax M2's account retains the unresolved larger-scale validation caveat.
- Serving covers snapshots versus recomputation, prefix branching,
  speculative rollback, convolution buffers, P/D transfer metadata, precision
  and request isolation.

Numeric checks executed independently: affine scan order; RWKV WKV outputs
and state for (2,4,8); RetNet recurrence for the same values; normalized linear
attention and delta replacement; 16 GiB full KV versus 4 GiB KV plus 6.75 MiB
recurrent state in the specified exercise.

Renamed headings retain explicit aliases for their former published anchors.
The existing overview, linear-attention and hybrid-sequence anchors remain.

## Figures and provenance

There are 35 source images in the chapter. Two were replaced, not 35 newly added:

- RWKV-4 Figure 3: original SVG from arXiv 2305.13048v2.
- RWKV-7 Figure 2: original SVG from arXiv 2503.14456v1.

Both new files retain original bytes. Their versioned URLs, CC BY 4.0
licenses and SHA-256 hashes are in the asset registry. No diagrams were
redrawn. The former RWKV assets remain available to unrelated pages.

Corrected captions/source attribution: the SSM A/B/C/D signal-path diagram;
Maarten Grootendorst's Mamba block; Gu and Dao's selective SSM; RetNet's two
depicted computation forms; Mamba-2 Figure 6; Berkeley's Bahdanau slide 60;
CMU's scan slide 30 and linear-attention slide 34; Nemotron's combined
architecture and evaluation panel.

The existing Grootendorst and RetNet images still have unconfirmed reuse
metadata; correcting their attribution does not establish a license.
This review does not certify all inherited asset rights.

All 35 desktop and 35 narrow-screen figure screenshots were personally
inspected. Images fit the article without CSS cropping. Dense course slides
and multi-panel charts require enlargement on a phone. Figure links open
the source image on a separate page; the site does not currently provide a
lightbox. Several inherited paper crops still include source captions or
adjacent text; they have not all been replaced in this pass.

## Verification

Two publication regressions were reproduced and fixed with tests: the adapter
now removes the duplicate source H1 while preserving preceding empty anchors;
asset provenance validation accepts unchanged version-pinned SVGs while
retaining hash, license and source-version checks. The landing-page test was
also stale: it now checks the six route cards and the two existing university
course cards in their separate navigation sections.

Final verification:

- Site build: 1961 pages, exit 0.
- Publisher unit tests: 222/222.
- Generated-output tests: 12/12.
- Site tests: 60/60.
- Stanford/Berkeley importer tests: 46/46.
- Stanford frozen-snapshot check and both course-ledger checks pass.
- Built links: zero broken internal routes, fragments or files.
- Browser: 1440px and 390px, RU and EN fallback, all HTTP 200.
- Each rendered route has 35 decoded images, one H1 and one overview anchor;
  zero KaTeX errors, overflowing display-math containers, document overflow
  or page JavaScript errors. This is 140 image-load checks across four cases.
- The EN route displays "This content is not available in your language yet."
  It is not counted as a translated chapter.
- `git diff --check`: clean.

Temporary browser evidence is stored in
`/private/tmp/bookvar-sequence-qa-20260928/`; logs use
`/private/tmp/bookvar-*-20260928-final.log`. These are diagnostic evidence,
not published artifacts. The preview runs locally on port 4330. Opening it
in the app returned `queued`; the direct URL was checked independently.
