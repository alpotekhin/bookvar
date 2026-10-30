# Chapter 41: corpus collection, cleaning, and mixtures

Date: 2026-10-13.

Scope: `00 Учебник/11 Pre-training и Scaling/41 Сбор, очистка и смеси данных.md`.
This audit records the chapter revision, source checks, and numerical checks.
Site build, publication, and rendered-page acceptance belong to the parent
review; they are not established by this record.

## Reading and source checks

The chapter was read in full before and after the substantive revision.
CS336 lectures 13 and 14 were read in full from the existing local snapshots.
Assignment 4 was checked against its source PDF, including extraction,
language identification, quality filtering, and corpus construction sections.
Assignment links in the chapter now use the snapshot commit
`0555bea66369872d912652debf10b115ca0688c8`, rather than an unversioned `main`.

Additional primary sources consulted:

- [Common Crawl: WARC, WAT, and WET](https://commoncrawl.org/blog/web-archiving-file-formats-explained).
- [DCLM, version 4](https://arxiv.org/html/2406.11794v4): extraction comparison,
  filtering stages, quality-filter comparison, and the experimental scale.
- [fastText language identification documentation](https://fasttext.cc/docs/en/language-identification.html).
- [CCNet](https://arxiv.org/abs/1911.00359).
- [Docling, version 1](https://arxiv.org/html/2501.17887v1): document conversion
  stages and original Figure 1.
- [FinePDFs author material](https://huggingfacefw-finepdfsblog.hf.space/).
- [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean
  Crawled Corpus](https://aclanthology.org/2021.emnlp-main.98/).
- [UniMax](https://arxiv.org/html/2304.09151): allocation subject to a maximum
  number of repetitions. The paper's original unit is characters; the
  chapter labels its token-based numerical example as an adaptation.
- [RegMix](https://arxiv.org/abs/2407.01492): small-run mixture search and
  regression-based prediction, distinguished from measured final-run loss.
- [Olmix](https://arxiv.org/html/2602.12237v1): comparison of mixture methods.
- [Data Mixing Laws](https://arxiv.org/html/2501.11747v1): simulated epoching
  and the difference between a sampled source and the full source.

The Pile and Dolma composition tables were transcribed from the existing
source figures and checked row by row. Their units are preserved: storage
size or corpus weight must not silently become the fraction of training
tokens under an arbitrary tokenizer.

## Substantive changes

The opening now follows one concrete problem: retaining useful article text
while removing menus, template fragments, and extraction damage. It then
develops the corpus pipeline rather than listing its stages.

- WARC, WAT, and WET are distinguished from an archived URL and a parsed
  document. The text explains what information is already lost in WET,
  why extraction versions matter, and why truncation flags must survive.
- A small HTML example shows what extraction keeps or discards. PDF reading
  order, OCR, layout detection, and table structure are explained separately.
- A JSON record gives an illustrative, versioned document representation.
  The example is labelled as an example, not as an executed production job.
- Language identification distinguishes the model's score from an
  empirically calibrated probability. The chapter treats mixed-language and
  uncertain documents explicitly and works through precision and recall.
- Quality rules include the exact Assignment 4 Gopher-style subset in a
  native table. Words, characters, and tokenizer tokens are not conflated.
- Classifier-based filtering explains the construction of positive and
  negative examples, the danger of learning source style, domain-level
  validation splits, and manual inspection of rejected as well as accepted
  documents. Perplexity filtering includes a worked calculation.
- DCLM extraction and filtering results are presented with the experimental
  scale and metric meaning. The filtering percentages retain their original
  denominator; a within-stage retention fraction is calculated separately.
- The small CS336 corpus example explains why changing filtering also
  changes repetition under a fixed training-token budget.
- C4, The Pile, and Dolma demonstrate different questions about corpus
  composition. Full Pile and Dolma tables replace tiny image tables.
- Mixtures now define whether the sampling unit is a document, a fixed-size
  token block, or a token budget. Document fractions and token fractions
  have separate formulas and a two-source numerical example.
- Expected repetitions use `e_i = p_i T / n_i`, with a three-source example.
  Proportional, equal, smoothed, and repetition-capped allocation are
  explained using the same quantities.
- The UniMax constraint is corrected to `p_i T <= E_max n_i`. A worked
  capped allocation also checks that the requested total budget is feasible.
- RegMix and simulated epoching have concrete mechanisms and examples.
  The comparison of seven mixture-selection methods is a native table.
- The closing Python example reports allocated tokens and source
  repetitions, validates its inputs, and is exercised by the check script.

Deduplication and contamination remain a short bridge to chapter 41a;
the chapter does not duplicate its algorithms. The preceding general
pretraining chapter and the data-loader/packing chapter are linked where
their explanations become relevant.

## Figures and readable tables

All old files remain on disk. The following old embeds were removed from
this chapter, not deleted from the repository. Paths below are relative to
`00 Учебник/Assets/Figures/curated/stanford-cs336-2026/data-corpus/`.

| Old embed | Replacement |
|---|---|
| `dclm-wet.png` | Native three-row extraction-results table with experiment scale |
| `finepdf-layout.webp` | Original Docling Figure 1 and a PDF-conversion explanation |
| `dclm-quality.png` | Native eight-row quality-filter comparison |
| `the-pile-composition.png` | Complete native 22-row composition table |
| `dolma-composition.png` | Complete native seven-source composition table |
| `data-mixing-methods.png` | Native transposed comparison of seven mixture methods |
| `dclm-filter.png` | Original DCLM SVG, added and checked by the parent reviewer |
| `c4-domains.png` | Separate, higher-resolution C4 domain and website panels, prepared by the parent reviewer |

The existing `raw-target-schema.png`, `data-filtering-scale.png`, and
`regmix.png` remain. The chapter currently contains seven figure embeds.
All original chapter figures were visually inspected before the changes.
The Docling raster and the two replacement C4 panels were also inspected
by this editor. The final DCLM SVG visual check belongs to the parent
reviewer, who selected that asset and changed its embed.

New assets reside in
`00 Учебник/Assets/Figures/curated/corpus-collection-review-2026-10/`.

For `docling-pdf-pipeline.png`:

- Authors: Nikolaos Livathinos et al.; the complete author list is in
  `NOTICE.md` beside the asset.
- Work: *Docling: An Efficient Open-Source Toolkit for AI-driven Document
  Conversion*, arXiv:2501.17887v1, Figure 1.
- Pinned source: <https://arxiv.org/html/2501.17887v1#S1.F1>.
- Exact raster URL: <https://arxiv.org/html/2501.17887v1/docling_arch_v2.png>.
- License: CC BY 4.0, shown on the versioned paper page.
- Dimensions: 1020 × 480.
- SHA-256: `1e6a4b15237301560be98d0acbc8a8bc95d4653a0a3a032108eac22c98c00b0c`.
- Changes: none; byte-identical original image, renamed locally.
- Retrieval and personal visual inspection: 2026-10-13.

The parent reviewer owns the provenance entries for
`dclm-filter-original.svg`, `c4-top-level-domains.png`, and
`c4-websites.png`, as well as the central figure-registry updates.
The chapter caption for the retained RegMix figure now identifies the
verified CC BY 4.0 license. Public availability alone was not used as a
license claim.

## Executed checks

The reproducible check was created at
`/tmp/corpus-collection-validation.py`; the parent reviewer has also added
a portable repository copy at
`publishing/tools/check_corpus_collection_examples.py`.
This audit's executed evidence is from the temporary script against the
edited chapter:

```text
PASS: actual Markdown Python example and invalid-input cases
PASS: illustrative JSON record parses
PASS: LID precision/recall, token-weighted fraction, perplexity, retention, epochs
PASS: document-vs-token weighting and inverse conversion
PASS: capped equal-budget allocation is feasible and matches the text
PASS: simulated-epoching equality and 50-pass example
PASS: source units, explicit anchors, code fences, seven image paths
```

The script extracts and executes the actual Markdown Python example instead
of checking an independently retyped copy. It also checks invalid input
cases, the JSON record, exact-fraction mixture calculations, water filling
under repetition caps, and preservation of existing `source_unit_id`
values and explicit anchors against `HEAD`.

`git diff --check` passed for the chapter. Existing heading anchors were
kept; two renamed headings retain explicit compatibility anchors.

## Acceptance boundaries

- No full corpus was downloaded, filtered, or used to train a model.
- Numerical examples are illustrative calculations, not benchmark results.
- No extractor throughput, model quality improvement, or acceleration was
  measured by this revision.
- No English pages, raw course snapshots, navigation, CSS, or shared
  registries were edited by this subtask.
- No build, commit, push, or deployment was run by this subtask.
- The parent reviewer must complete the final rendered-page review and
  common repository checks after integrating the chapter and figure
  registry changes.
