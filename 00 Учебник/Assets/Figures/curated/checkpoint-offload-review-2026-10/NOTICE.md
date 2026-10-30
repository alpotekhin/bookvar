# Chapter 44b: illustration provenance

Verified 2026-10-13. Machine-readable registry handoff: assets.json.
This directory contains one new active image. No custom drawings were created.

## DeepSpeed optimizer placement

File: deepspeed-optimizer-offload.png
Dimensions: 546 x 364 pixels.
SHA-256: 3e31b0a326d662ada6f409025696df8c96f6680bb648b694847a1fa2afda8a8f

Authors: DeepSpeed Team; Guanhua Wang, Masahiro Tanaka, Xiaoxia Wu,
Lok Chand Koppaka, Samyam Rajbhandari, Olatunji Ruwase, Yuxiong He.

Source project and commit:
https://github.com/deepspeedai/DeepSpeed/tree/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c

Original image:
https://raw.githubusercontent.com/deepspeedai/DeepSpeed/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/blogs/deepspeed-offloadpp/images/cpu-offload.png

Author's explanation:
https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/blogs/deepspeed-offloadpp/README.md#twin-flow

License: Apache License, Version 2.0.
Exact upstream LICENSE copied to DeepSpeed-LICENSE.txt, SHA-256:
c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4
https://github.com/deepspeedai/DeepSpeed/blob/dcd8595c234155a4c70ca9f3cd8c9cbef462cc3c/LICENSE

Changes: none. The PNG is copied byte-for-byte; no crop, redraw, relabeling,
upscale or compositing. Its original CPUAdam/FusedAdam placement is explained
in the Russian caption; no performance number from the article is imported.

The complete license is also published as
`site/public/licenses/deepspeed-Apache-2.0.txt`.

The pinned repository's THIRD_PARTY_NOTICES.md was checked. It identifies
TorchTitan-derived source code in listed Python files, not this illustration.
No separate notice overriding the repository license was found in this image
directory. The attribution above is retained and the full license accompanies
the copied file. Publication must retain the license copy and attribution.

## Existing PyTorch originals reused, not copied again

Files already in curated/arithmetic-review-2026-09:
- pytorch-checkpoint.png, SHA-256:
  13bc1fead2c8b4976effff97a3f6dff14eb81788e27adca200c1e77e972aab74
- pytorch-selective-checkpoint.png, SHA-256:
  14955736f9fe3a8b870e0ebf94c32fde67fd514f0e6090fbfa6763b4f73551c0

PyTorch contributors; pytorch/pytorch.github.io commit
563f92c6e87782196b81894c4e671428cd8d34af, images fg5.png and fg8.png under
assets/images/activation-checkpointing-techniques/.
https://github.com/pytorch/pytorch.github.io/blob/563f92c6e87782196b81894c4e671428cd8d34af/LICENSE

BSD-3-Clause, Copyright 2018 Facebook Inc.
The existing full notice is retained at:
site/public/licenses/arithmetic-visuals-NOTICE.txt
No changes to source bytes or existing license copy. Registry used_in gains
chapter44b; old audit provenance remains.

## Rejected Stanford PDF crops

Existing full slides l05-p35.png and l05-p36.png remain untouched in
curated/stanford-cs336-2026/systems/. Their legacy rights record is NOT a
verified source license. Public course access or user permission was not
treated as permission from the author.

Two candidate crops were rendered directly from the existing original PDF
(lecture_05.pdf, commit 8b59b50730766695c2ffedd1a79c50cd09b9eb91, SHA-256
a2d5664a3b5a1f246df2fe58b43189215f29931ec05bde4d8a6af4664f55cafe)
and personally reviewed. The embedded diagrams were blurry; page35 already
contains a truncated line at the right edge of its frame in the full source.
Cropping cannot repair that source artwork without redrawing it.

Candidates were removed from Assets and kept only temporarily in /tmp:
cs336-saved-sigmoids-rejected.png, cs336-recomputed-sigmoids-rejected.png.
They are not publication assets and need no active registry entry.
The source-backed eight-versus-five tensor-access comparison is preserved
as prose and a normal Markdown table, with the original page citation.
