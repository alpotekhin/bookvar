# Chapter 44e: source figures and licenses

Verified 2026-10-14. Registry handoff is in assets.json. No custom diagrams
were created. Three complete source illustrations are reproduced; the
chapter explains their meaning and preserves their original labels.
Full license texts are retained alongside these files. Publication must
retain attribution, source links, applicable license and notices.
No author endorsement is asserted.

## PyTorch FSDP, Figures 1 and 4

Authors: Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang,
Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer,
Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen,
Geeta Chauhan, Yuchen Hao, Ajit Mathews, Shen Li.

Title: PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.
PVLDB 16(12), 3848–3860, 2023; DOI 10.14778/3611540.3611569.
Version-pinned extraction source: https://arxiv.org/pdf/2304.11277v2
Source page: https://arxiv.org/abs/2304.11277v2

Retained source:
`05 Источники/Papers/Source PDFs/pytorch-fsdp-2304.11277v2.pdf`
SHA-256: 7ee0f202b567627d668deae8b786e2a7a48920fbed22cc125a060f4c6da27cab

The first page of this exact PDF explicitly states:
“This work is licensed under the Creative Commons BY-NC-ND 4.0 International
License.” It also states that copyright is held by the owner/author(s)
and publication rights are licensed to the VLDB Endowment.

License: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0.
https://creativecommons.org/licenses/by-nc-nd/4.0/
Complete license: LICENSE-CC-BY-NC-ND-4.0.txt
Downloaded from https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt
SHA-256: 38762e3777f4ec00a6f769062a7c3f704fb78ce08303ecff88558da4c49cf9ea

Each figure, INCLUDING its full original caption, is reproduced unchanged.
Only surrounding article content is omitted. The PDF is technically
rasterized without translating, recoloring, relabeling, redrawing,
compositing or cropping internal figure content. The source license permits
noncommercial reproduction in whole or in part and technical format changes;
it does not authorize sharing adapted material. These figures are not
relicensed under the book's license and must not be translated or edited
under this reuse record. Commercial reuse requires separate review/permission.
The license includes its original warranty and liability disclaimers.

Figure 1:
- PDF page 3; source Figure 1, FSDP Algorithm Overview.
- File: pytorch-fsdp-figure1-overview.png, 1722 × 1057 pixels.
- SHA-256: 1944e36c2196172ae200bfad3a25932c488b5cf0660307f398a498fd8bad55fc
- PDF-point clip: (315, 82, 561, 233), scale 7.

Figure 4:
- PDF page 5; source Figure 4, hybrid sharding on 16 GPUs.
- File: pytorch-fsdp-figure4-hybrid-groups.png, 1722 × 1099 pixels.
- SHA-256: 020f1f62d5c013edfa97e6b2e2d1dc1125cb6ef3fdb297df68cea7432828c124
- PDF-point clip: (315, 84, 561, 241), scale 7.

Both complete original PDF pages and final crops were personally viewed.
All arrows, group boundaries, rank labels, legends and captions are present.
The chapter explicitly distinguishes historical FSDP1/FlatParameter from
FSDP2. The figures explain grouping/lifetimes, not measured performance.

## EDLS DTensor layouts

Authors and copyright: Copyright (c) 2021 course authors.
Efficient Deep Learning Systems, Week 5 seminar, PDF page 10.
Commit: e632aa89ca9e6638d52e1b686095e7442faffbb0

Source page:
https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/seminar.pdf
Original PDF:
https://raw.githubusercontent.com/mryab/efficient-dl-systems/e632aa89ca9e6638d52e1b686095e7442faffbb0/week05_fsdp/seminar.pdf

Retained source:
`05 Источники/Papers/Source PDFs/edls-week05-seminar-e632aa89.pdf`
SHA-256: 76e3d309d5c2279a95783a8d04930f4cf0e6d1b8c3f31ca6fe580f1b0c020b97

License: MIT. The complete unchanged upstream notice is LICENSE-EDLS-MIT.txt.
https://github.com/mryab/efficient-dl-systems/blob/e632aa89ca9e6638d52e1b686095e7442faffbb0/LICENSE
License SHA-256: 01908feead9275d5043e082e9fa37e415a2972dca45fd62b8489e1fd2a058a13

File: edls-dtensor-layouts.png, 1980 × 924 pixels.
SHA-256: 1799545cb33e70ff5befe244c4c3774ed85d75aedce6832e5c6f7a32ab230c32
PDF-point clip: (45, 72, 705, 380), scale 3.
Only slide heading/margins removed; original values, all three layouts and
both mesh expressions are intact. No translation or redraw.
Personally viewed in the original page and final crop. The right-hand
layout is reproduced by actual four-process chapter code, not a mock.

## Reproducing the rasterization

With PyMuPDF installed, run from repository root:

~~~python
from pathlib import Path
import pymupdf

source = Path("05 Источники/Papers/Source PDFs")
output = Path("00 Учебник/Assets/Figures/curated/fsdp-review-2026-10")
cases = [
    ("pytorch-fsdp-2304.11277v2.pdf", 3, (315, 82, 561, 233), 7,
     "pytorch-fsdp-figure1-overview.png"),
    ("pytorch-fsdp-2304.11277v2.pdf", 5, (315, 84, 561, 241), 7,
     "pytorch-fsdp-figure4-hybrid-groups.png"),
    ("edls-week05-seminar-e632aa89.pdf", 10, (45, 72, 705, 380), 3,
     "edls-dtensor-layouts.png"),
]
for filename, page, bounds, scale, image in cases:
    with pymupdf.open(source / filename) as document:
        document[page - 1].get_pixmap(
            clip=pymupdf.Rect(bounds),
            matrix=pymupdf.Matrix(scale, scale),
            alpha=False,
        ).save(output / image)
~~~

## Removed from this chapter, retained on disk

CS336 original screenshots l08-p17.png, l08-p22.png and l08-p25.png remain
unchanged. Their used_in references to chapter 44e should be replaced by the
editorial audit. The existing course rights record is not upgraded merely
because the course is publicly accessible.

Harvard's zero-memory-partitioning.svg also remains unchanged but is not
used in the chapter: its bar heights do not agree with the printed GB values.
The chapter now gives exact values as a Markdown table and worked calculation.
See assets.json for the four explicit used_in handoffs.
