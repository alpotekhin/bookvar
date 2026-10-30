# Higher-resolution originals selected during the main review

These replace earlier lecture screenshot copies in the Russian chapter only.
Earlier assets and English pages are preserved. Checked 2026-10-13.

## DCLM filtering

Jeffrey Li et al., *DataComp-LM*, arXiv:2406.11794v4, Figure 4.
https://arxiv.org/html/2406.11794v4#S4.F4
License CC BY 4.0, as declared by that version of the paper.

`dclm-filter-original.svg` is an unchanged download of
https://arxiv.org/html/2406.11794v4/dclm_sankey_fig.svg
SHA-256: `edc3cda632521ffade66f4747d466c3b2a900c2f34d5da2af1a4d38c6e1354a5`.
The source image excludes the caption overlaid separately by the paper's
typesetting. The Russian caption retains the crucial original-document denominator.

## C4 composition

Jesse Dodge et al., *Documenting Large Webtext Corpora*, EMNLP 2021, Figure 2.
https://aclanthology.org/2021.emnlp-main.98/
ACL license: CC BY 4.0; https://aclanthology.org/faq/copyright/ .
The same two panels used by Stanford CS324/CS336 are rendered from the
coauthor-hosted paper: https://maartensap.com/pdfs/dodge2021documentingC4.pdf .
Local source: `05 Источники/Papers/Source PDFs/dodge-2021-documenting-c4-author.pdf`.
Source SHA-256: `79d45e53ab0c1c9f81f8082b7c294020536f97298c9a629ca8e20ef73c1c6a3a`.

Poppler commands use PDF page 3, `-scale-to 3600 -png -singlefile`:

- `c4-top-level-domains.png`: `-x 290 -y 290 -W 860 -H 810`.
  SHA-256 `fc97e4e3abfabeb8d167e49ce7159b366781e3788c18c13d98f882d7fd4d99ce`.
- `c4-websites.png`: `-x 1180 -y 290 -W 1100 -H 810`.
  SHA-256 `f5840deae1fbadd61ab6130fa45ebca44d138eec1b906f6a143f7d6f7d7d4f9c`.

Only the two separate panel rectangles are extracted; captions are outside
the crops and attributed next to each image. No plotted data, labels, ticks,
colors or scales were modified. The right crop was widened after inspection
to retain the complete rightmost tick label. Counts are spaCy word tokens in
C4.EN, not the tokens of an arbitrary LLM tokenizer.
