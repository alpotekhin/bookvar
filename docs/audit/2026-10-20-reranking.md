# Reranking and late interaction — local editorial review

Date: 2026-10-20. Branch: `editorial-route-alignment`.

Chapter: `00 Учебник/15 Embeddings, Retrieval и RAG/62 Reranking — cross-encoder и late interaction.md`.
Route: `/textbook/retrieval/reranking/`.
Chapter SHA-256: `5d7522f493395addabf8cac39a95b6cc4ef0c8c134f5afa13ac488d5059657c6`.

Status: locally accepted in the selected scope. The original and expanded
chapter were read in full; current numerical/source checks and actual browser
inspection are recorded below. The broader textbook goal remains unfinished.
No English translation, commit, push or deployment is in scope for this pass.

## Sources read and selected scope

`05 Источники/Originals/reranking-2026-10-20/references.json` records downloaded
source identities and distinguishes collection-only files from chapter evidence.
This is selected-section review, not ingestion of whole courses or repositories.

- Sentence Transformers, pinned to `8a880ec7f1524640619fe382940ee319144f74ca`:
  complete Cross-Encoder applications and Retrieve & Re-Rank explanations;
  Cross-Encoder Loss Overview table and surrounding explanation; pretrained
  model usage and MS MARCO section. BCE implementation, ListNet implementation
  including label softmax and padding, RankNet constructor and loss conventions,
  and the scoring/weighting passage of LambdaLoss were checked. Anecdotal loss
  rankings and hardware throughput figures were not adopted as new claims.
- MS MARCO MiniLM-L6-v2 model card and config, revision
  `fbf9045f293a58fa68636213c5e0cb8a2de5d45e`: both fixed English Berlin passages,
  raw scores and BERT configuration. The model-card API example was adapted,
  not presented as an unchanged upstream program.
- Original ColBERT, arXiv 2004.12832v2: complete §3.1–3.6 across PDF pages 4–6,
  and Figure 2 on page 2. The diagrams were compared with the pre-existing crop.
  No old speedup/quality table is used as a current benchmark claim.
- ColBERT-QA, published TACL 2021: Figure 1(a,b) and caption, first-page CC-BY
  notice, §3.1–3.2 including weak answer-string labels, retrieval rounds and
  question splits. The training-loop figure is explicitly identified as retriever
  training, not cross-encoder distillation.
- ColBERTv2, published NAACL 2022: complete method §3.1–3.5; supervision,
  residual representation, indexing and approximate candidate generation versus
  full-matrix scoring. Its Figure 1 credits another work with permission and was
  not silently treated as an original unrestricted figure; the chapter instead
  extracts the original TACL diagram with its own license notice.
- ColBERT repository pin `cc4f3dc91c0b45d2d08c251d9d95178285c65f1c`: README method
  introduction, complete MIT license, query/doc encoding and MaxSim reduction
  code, and complete selected residual-decompression method. The latter explicitly
  normalizes reconstructed token vectors. The 370-pixel README diagram was
  inspected but rejected for publication in favor of a higher-resolution PDF
  extraction; other repository files were not exhaustively read.
- RankGPT, published EMNLP 2023: complete §3.1–3.2 and §4.1–4.3, Figures 2–3.
  The chapter uses the standard preferred-minus-rejected RankNet convention
  checked in Sentence Transformers, not the inconsistent printed sign of the
  paper's distillation equation. No source figure was edited to hide that issue.
- Guo et al., PMLR 2017: Figure 1, calibration definitions/reliability discussion
  in §2, binary calibration and temperature-scaling passages in §4.1–4.2.
  The classification figure is explicitly not described as a retrieval experiment.
  The complete PMLR publication agreement was read for the CC BY 4.0 basis.
- Complete Apache-2.0 license for Sentence Transformers was read; the unchanged
  public notice is reused. Published ACL permission evidence is the frozen ACL
  policy from the preceding retrieval pass, with the TACL paper also carrying
  its explicit license notice on the first page.
- Jurafsky and Martin, SLP3 frozen draft of 19 August 2026: the dense-retrieval,
  full-/bi-encoder and ColBERT explanation on PDF pages 14–16 in §11.3, including
  Figures 11.11–11.12 and the training-data discussion. Figures were read, not
  copied. Adjacent RAG pages seen during source selection are not presented as
  this chapter's newly ingested content.

## Historical images

The former comparison crop
`00 Учебник/Assets/Figures/curated/source-first-60-64/colbert-late-interaction.png`
included the beginning of the next paragraph. Its bytes were preserved as
historical evidence; it is no longer embedded in the Russian chapter. Registry
usage now points to this audit. The replacement `colbert-paradigms.png` is a
clean extraction of the same already-used Figure 2, including all panel labels.
Its inherited repository-owner reuse claim is preserved and explicitly separated
from independently verified open-license evidence in `legacy-figure-record.json`.
This pass does not establish new publication permission for that ACM figure.

The unchanged old Sentence Transformers bi-/cross-encoder PNG was verified
byte-identical to the pinned upstream source. Its previously missing raw source
path and owner-claim-only license metadata were replaced with the real frozen
PNG, hashes, commit and Apache-2.0 license evidence.

## CPU model verification limitation

On this host, the fixed MiniLM checkpoint returned NaNs in float32 with both
PyTorch 2.8.0 and 2.14.1, Transformers 4.57.6. Model parameters were finite;
the first nonfinite output appeared in the first encoder layer's query linear
projection from finite inputs. Eager versus SDPA and disabling MKLDNN did not
resolve it. Converting the same model to float64 produced the expected logits
`[8.607139745639522, -4.320078221938003]`, probabilities approximately
`[.9998172374, .0131243052]` and order `[0,1]`.

The chapter explicitly uses a float64 CPU reference calculation. This is a
verified mitigation for this example, not a diagnosed library-kernel fix or a
serving recommendation. No GPU behavior, model-training outcome or throughput
claim is inferred. Temporary dependencies are outside the repository; the site's
dependency files were not changed.

## Explanation acceptance

The expanded chapter was read in successive complete ranges, then reread after
the prose and mobile-formula edits. A truncated repeated middle range was read
again in a bounded range rather than counted as complete. All seven original
numbered section anchors, the MaxSim example anchor and source footer remain.

| Mechanism | Worked explanation and limitation |
| --- | --- |
| Retrieve/rerank | Hypothetical version/SSO candidates; candidate cutoff versus final context; original English pipeline explanation retained with attribution, not replaced by a translation. |
| Cross-encoder | Joint BERT sequence, CLS head, pair-dependent representation, million-versus-hundred pair count and explicit 512-token pair budget. |
| Real API | Pinned model, two exact English source passages, raw logits versus sigmoid, ID mapping and a disclosed float64 CPU reference, not a quality benchmark. |
| Data | Domain hard negatives, graded versus binary labels, unjudged documents, retriever/version provenance and teacher errors. |
| BCE | Positive and negative logits, numeric losses and gradient directions; no double sigmoid before BCEWithLogitsLoss. |
| RankNet | Preferred-minus-rejected difference, correct/inverted order, shared-offset invariance, equal labels and explicit log-base convention. |
| ListNet | Label and score softmax distributions, reversed-list loss 1.982816347 and minimum cross-entropy .832395582; entropy/KL distinction and no identity with nDCG optimization. |
| Calibration | Same ranking with different threshold results; .9 predicted versus .6 observed, one-group temperature 5.419, Brier .33 to .24, calibration split and limits for RAG answer confidence. |
| Late interaction | Four original matching paradigms, shared encoder versus joint input, token projection/normalization and parameter-free MaxSim. |
| MaxSim | Two explicit 2×3 matrices, winning indices, 1.6 versus 1.4, changed row giving 1.2, non-bijective matches and limits of token-match explanations. |
| Masks | Negative similarities with a zero padding column; mask before max, all-invalid rejection, and active query-augmentation MASK tokens versus ordinary padding. |
| Indexing | Reranking complete document matrices versus token-level nearest-neighbor candidate generation and refinement; recall can be lost before full scoring. |
| Residual compression | Explicit center, residual, four quantization levels, restored coordinates and re-normalization; 36 versus 256 bytes per vector, not a whole-service compression claim. |
| Supervision variants | ColBERT-QA answer-string heuristic and question splits distinguished from ColBERTv2 cross-encoder teacher/KL training. |
| Cost | Attention's quadratic length term and linear projections/FFN; 1048576 versus 526336 padded attention positions, not an asserted wall-clock speedup or p95 formula. |
| LLM ranking | Query likelihood, relevance labels, permutations, dependency on list position and the difference between inference input format and training loss format. |
| RankGPT windows | Full original three-step figure, exact ID trace, three local calls, validated permutations and a counterexample to global sorting after one perfect local pass. |
| Metrics | Fixed candidates, full-gold IDCG, candidate oracle .765360637 versus baseline .530721274; unchanged Recall 2/3 and nDCG headroom .234639363. |
| Failure analysis | Visible versus invisible rank information, pair-side swaps/cycles, length/windows/source shortcuts, ACL before external scoring, untrusted document instructions and dated labels. |

## Editorial review

No-ai-slop SKILL and eval were applied with the project's academic qualifications.
The authorized expansion includes clearly labeled new teaching examples, not
invented source experiments. The source quotation and English model-card passages
remain in their original language; only API-link formatting in the quotation was
adapted and disclosed. Full text is saved in the canonical Markdown file instead
of pasted into chat.

Editing-principle checks 1–11 pass under that scope: explanations, intermediate
arithmetic, caveats and source terminology were preserved. Generic process
sentences were removed, ambiguous ListNet/nDCG and length-cost claims corrected,
and several unnecessary English nouns replaced in authored Russian prose.
Words/pattern checks pass with the local exceptions for useful mathematical
contrasts, method names, code and normal Russian punctuation. Necessary cautions
about candidate ceilings and source-versus-model evidence were not deleted merely
because they contain a contrast. The style checklist is not a quality or
authorship detector; mathematical and source review are separate evidence.

## Figure acceptance

Eight original illustrations are embedded: one unchanged Sentence Transformers
comparison and seven entries in
`00 Учебник/Assets/Figures/curated/reranking-review-2026-10/assets.json`.
The pipeline is a byte-identical pinned upstream PNG. Six PDF extractions record
the PDF/page, full-page render/hash/dimensions, exact crop box, output size,
source authors, license evidence and modifications. All six crop outputs match
their recorded source rectangles pixel-for-pixel.

The first comparison crop ended too close to the example labels. The final box
`[215,320,2345,940]` includes all four panels and all parenthetical example labels,
without the paper's caption or next paragraph. RankGPT panels preserve their
original English prompts/IDs and complete step labels. The calibration figure
preserves all four panels, titles, axes, values and legend; the caption identifies
the original CIFAR-100 domain. Nothing was redrawn, recolored or generated.

## Current verification

- `publishing/tools/check_reranking_examples.py`: passes 1185 counted grouped
  parameter cases on the chapter hash above. It executes seven blocks both
  independently and in one namespace; checks finite-difference loss gradients,
  entropy, extreme values, exact token budget, MaxSim against independent dot
  products, permutations/masks, residual/padding arithmetic, sliding-window
  coverage and invalid replies, fixed-candidate metrics, source hashes, literal
  excerpts, figure pixels and built anchors.
- Real CPU inference: PyTorch 2.14.1, Transformers 4.57.6, float64, pinned model
  and two source pairs. The verified float32 limitation remains disclosed above.
- `pnpm --dir publishing test:unit`: 251 tests across 18 files pass. This includes
  49 asset-registry tests, not 251 plus another 49 tests. New adversarial metadata
  cases reject incorrect figure hashes, source URLs/pages, crop boxes and rights
  identities instead of accepting a source link alone.
- `pnpm --dir site build`: 1961 pages built; final log completed at 18:56:03.
- `pnpm --dir publishing test:output`: 13 tests pass.
- `pnpm --dir publishing check:links`: no broken internal route, fragment or file.
- `pnpm --dir publishing check:source-coverage`: passes; its course/artifact counts
  are structural and do not prove full-course teaching coverage.

The first asset check correctly failed for a new source location and a not-yet
saved audit usage target. The audit now exists and the specific pinned image has
a narrow source/byte/license validator. Generic source checks were not relaxed.
The passing results above are subsequent runs on the corrected metadata.

## Actual rendered inspection

In-app browser, desktop 1365×900 and mobile 390×844:

- All eight figures opened with correct original file, caption and source links;
  no horizontal body overflow. All extracted panels were separately viewed
  against their full PDF pages before embedding.
- Tall calibration and prompt panels remain available through viewport scrolling;
  wide four-panel ColBERT comparison labels are readable at mobile 4× with
  horizontal scrolling. Escape closes the viewer and returns focus to the source.
- The initial mobile rendering had three overwide displays: BCE 404 pixels,
  the paired MaxSim matrices 438, and IDCG 354 in a 343-pixel article. BCE/IDCG
  are now multiline and the two matrices stacked. Final DOM inspection shows all
  seven display formulas at 343-pixel width without internal overflow, no KaTeX
  errors and no horizontal body overflow. This was visually checked, not inferred
  merely from a successful build.

Proof screenshots under `/tmp/`:
`bookvar-reranking-{pipeline,bi-cross,calibration,paradigms,maxsim,supervision,instructions,windows}-{desktop,mobile}-2026-10-20.jpg`,
`bookvar-reranking-paradigms-mobile-zoom-2026-10-20.jpg` and
`bookvar-reranking-maxsim-math-mobile-final-2026-10-20.jpg`.
Logs use `bookvar-reranking-…-2026-10-20.log`.

The local chapter review is complete. General float32 behavior of this host and
independent publication permission for the inherited ACM figure are not claimed
resolved. No model-training run, corpus benchmark, translation or live release
was performed. The next chapter is `63 RAG — полный конвейер.md`.
