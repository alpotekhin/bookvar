# Quantization figure provenance

All four figures are original source material, not generated diagrams.

- hf-weight-packing.png: Hugging Face Team, documentation-images dataset,
  pinned revision 2dd75e791509da38222ecbd3bfaa2b44ac658ef4.
  CC BY-NC-SA4.0, as stated in that dataset's README.
  Copied unchanged; the image remains under its original license.
- smoothquant-intuition.png and smoothquant-migrate.jpg: Guangxuan Xiao,
  Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth and Song Han;
  MIT HAN Lab SmoothQuant, revision c61476d728e42ae0d8a35e7e78494edcac3237b5.
  MIT, Copyright2022 MIT HAN Lab. The complete license is retained at
  site/public/licenses/smoothquant-c61476d7-MIT.txt.
- qlora-training-original.png: Dettmers, Pagnoni, Holtzman and Zettlemoyer,
  QLoRA Figure1, arXiv:2305.14314v1, CC BY4.0.
  The author-supplied standalone vector figure, figures/qlora.pdf, is retained
  unchanged, with its source archive. Poppler renders its entire single page
  at250 DPI. No labels, axes or other figure content were cropped or redrawn.

assets.json and the main asset registry retain direct source URLs, license
evidence, version pins, dimensions and hashes. Figure captions contain attribution.

The two older chapter figures, HF absmax rounding and AWQ Figure2, retain their
existing registry entries and permission status. No new license is assigned
to them. The numerical text explains the rounding and diagnostic mixed precision
shown in those original figures.

Regenerate the QLoRA PNG from the retained vector source:

```sh
pdftoppm -r 250 -png -singlefile \
  '05 Источники/Papers/Source PDFs/qlora-2305.14314v1-figure1.pdf' \
  /tmp/qlora-training-original
```

The chapter checker validates source hashes, reproduction scope, and full-frame
rendering. It does not claim that reduced on-page figures have readable fine
labels at every phone width; clicking opens the original.
