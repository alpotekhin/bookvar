# Source figures for the word-embedding chapter

Verified 2026-10-21. No figure was redrawn, cropped or translated.
The five existing Voita PNGs and existing Alammar PNG retain their complete
original bytes; only their captions and provenance metadata were corrected.

## D2L hierarchical softmax

Authors: Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola.
Page: <https://d2l.ai/chapter_natural-language-processing-pretraining/approx-training.html#hierarchical-softmax>.
Image: <https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/hi-softmax.svg>.
Pinned revision: `23d7a5aecceee57d1292c56e90cce307f183bb0a`.
License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
SHA-256: `2d8c76e194b90130d364694d0429ba84df3927f6c14ba0ebe9d5e442a08a152c`.
The complete SVG has viewBox `0 0 390 199`.
The pinned LICENSE is retained at
`05 Источники/Originals/neural-language-model-2026-10-21/d2l-LICENSE.txt`.
The bold path to w3 is left, right, left; every internal node, including
the root, has an output parameter. No node/leaf was removed.

## Existing Lena Voita illustrations

Page: <https://lena-voita.github.io/nlp_course/word_embeddings.html>.
Five complete PNGs remain under `../voita-word-embeddings/`:
`count-based-idea.png`, `cbow-skipgram.png`, `negative-sampling.png`,
`glove-idea.png`, `glove-loss.png`.
Official image URLs and fixed hashes are recorded in asset-registry.yml.
Fresh official copies are retained under
`05 Источники/Originals/word-embeddings-2026-10-21/voita-*.png`.
All five byte comparisons passed.

No named redistribution license was independently established for these
existing images. The user's permission record is retained as such; it is
not relabelled as Creative Commons. Source identity is verified separately
from third-party rights. No additional Voita image was introduced.

The chapter explains the notation changes between the original CBOW image
and its equations, and distinguishes the sum in the image from its mean
example. The negative-sampling arrows describe score derivatives rather
than a universal finite-step guarantee. The GloVe graph is schematic:
the alpha=3/4 function below its cap is curved, not linear.

## Existing Jay Alammar illustration

Page: <https://jalammar.github.io/illustrated-bert/#bert-for-feature-extraction>.
Image: <https://jalammar.github.io/images/bert-feature-extraction-contextualized-embeddings.png>.
License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/),
as stated in the saved author page footer.
SHA-256: `ab1792bd6e14283e015f2bc60647de48df1e11c6fe7b270382800c0ca5865ebc`.
The full existing PNG remains under `../bert/` and matches the author's
image byte-for-byte. Its subject is layer extraction for CoNLL-2003 NER,
not a static-vs-contextual embedding comparison. The chapter caption is
corrected accordingly; no figure content or reported F1 value was changed.

Source files, page snapshots and registry entries preserve the authors'
licenses independently of the license for Bookvar code. No endorsement is
implied by including the figures.
