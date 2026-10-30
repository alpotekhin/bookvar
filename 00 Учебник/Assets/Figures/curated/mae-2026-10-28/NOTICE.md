# MAE original figures

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick
and Meta Platforms contributors. Source project:
https://github.com/facebookresearch/mae/tree/efb2a8062c206524e35e47d04501ed4f544c0ae8.

The project's CC BY-NC 4.0 LICENSE is preserved in the source packet and
in `site/public/licenses/mae-efb2a80-CC-BY-NC-4.0.txt`. The license applies
to the authors' released project material; this is not a claim to relicense
the full paper or the ImageNet dataset. The notebook credits its source
fox photograph as ILSVRC2012_val_00046145, retained beside both demo figures.

`architecture.png` is the complete README diagram, copied byte-for-byte.
It was visually compared with the paper's Figure 1. No labels were changed.

`pixel-reconstruction.png` is the complete image/png output of notebook
cell 10. `gan-comparison.png` is the complete image/png output of cell 14,
which uses a different checkpoint with an extra GAN loss. Both are decoded
directly from the pinned notebook's saved base64 data, not clipped from
the README's Colab screenshot. All four panels and their original labels
are retained. There is no crop, resizing, repainting or new model run.

In both outputs the last panel pastes visible input pixels into the raw
reconstruction. The reader explains that composition and keeps the GAN
result separate from the ordinary pixel-loss result. Hashes, cell indices,
source snapshot and license evidence are recorded in assets.json and the
central registry; tests compare the decoded notebook output byte-for-byte.
