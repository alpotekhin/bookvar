# Corpus deduplication illustrations

Verified 2026-10-13. These are original documentation/paper figures, not newly
drawn explanations. English labels are preserved. Three earlier CS336 code
screenshots were removed from chapter 41a only; their original files remain.

## Presidio: analyze and anonymize

- File: `presidio-analyze-anonymize.png` (1737 x 500).
- Author / copyright: Presidio Contributors.
- Source page: https://microsoft.github.io/presidio/text_anonymization/
- Pinned source: https://github.com/microsoft/presidio/blob/2523c7b74a469270c5c78bb253f140eafca21e31/docs/assets/analyze-anonymize.png
- Direct original: https://raw.githubusercontent.com/microsoft/presidio/2523c7b74a469270c5c78bb253f140eafca21e31/docs/assets/analyze-anonymize.png
- License: MIT; full notice retained in `PRESIDIO-LICENSE.txt`.
- License URL: https://github.com/microsoft/presidio/blob/2523c7b74a469270c5c78bb253f140eafca21e31/LICENSE
- SHA256: `bddfcb2ed5085b94f94ff2ed7f45d383d1e7a1fd85597f3ca557e5d4f2c4e5a7`.
- Changes: none; source bytes retained exactly.
- Review: full original inspected; text/caption correctly distinguishes entity
  detection from applying a replacement. The diagram's component name
  `Anonymizer` is not treated as a guarantee of anonymization.

## Lee et al.: Deduplicating Training Data Makes Language Models Better

Authors: Katherine Lee, Daphne Ippolito, Andrew Nystrom, Chiyuan Zhang,
Douglas Eck, Chris Callison-Burch, Nicholas Carlini.

- Publication: ACL 2022, pp. 8424-8445, DOI 10.18653/v1/2022.acl-long.577.
- Publication page: https://aclanthology.org/2022.acl-long.577/
- **Actual extraction source:** https://aclanthology.org/2022.acl-long.577.pdf
- Source PDF retained as `05 Источники/Papers/Source PDFs/lee-deduplicating-acl2022.pdf`.
- PDF length: 634490 bytes, 22 pages.
- PDF SHA256: `f282bf06ab3564b363a4df5ab184cebb729c23a3be3ecb9aa7c5edc0a9f5e045`.
- License: CC BY 4.0 for this ACL publication.
- Verified publisher policy: https://aclanthology.org/faq/#what-is-the-copyright-for-materials-in-the-acl-anthology
- License terms: https://creativecommons.org/licenses/by/4.0/
- Rights scope: the publisher policy explicitly covers ACL materials since
  2016 and excludes third-party publications. This is an ACL 2022 proceedings
  paper. Its figures 1 and 2 present the paper authors' own experiments.
- Full ACL PDF pages 5 and 6 were rendered and inspected before cropping.
  Their plots were also visually compared with arXiv v2. Final assets were
  rendered again from the **ACL PDF**, not from arXiv. The separate arXiv
  distribution license is not used as permission for reproduction.

### Figure 1

- File: `lee-dedup-cluster-sizes-figure1.png` (940 x 505).
- Source: PDF page 5, printed page 8428, Figure 1.
- SHA256: `3bb63b7c1d27806e56f42e5c306664ce71c76ad692c7e8cb045cd14bd89106bb`.
- Changes: extracted only the original plot; surrounding article text removed.
  Axis labels, logarithmic scale, legend and every original count are retained.
- Reproduction command (Poppler):

```sh
pdftoppm -f 5 -l 5 -r 300 -x 280 -y 1080 -W 940 -H 505 -png -singlefile \
  lee-deduplicating-acl2022.pdf lee-dedup-cluster-sizes-figure1
```

### Figure 2

- File: `lee-dedup-perplexity-figure2.png` (960 x 480).
- Source: PDF page 6, printed page 8429, Figure 2.
- SHA256: `65fcd935279cee348ffbeee9d203abf559b7cc6f4cd610c412e50da066744376`.
- Changes: extracted only the original plot; surrounding article text removed.
  All five evaluation rows, all three training-data series, legend and axes
  are retained. No relabeling or graph redraw.
- Reproduction command (Poppler):

```sh
pdftoppm -f 6 -l 6 -r 300 -x 1260 -y 280 -W 960 -H 480 -png -singlefile \
  lee-deduplicating-acl2022.pdf lee-dedup-perplexity-figure2
```

Both crops were inspected at their final native resolution. Caption context
is supplied in chapter 41a with attribution and direct publication links.
