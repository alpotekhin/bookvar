---
title: "PaddleOCR-VL v1: page layout and reading order"
type: source-note
status: reviewed
source_type: paper
locale: en
translation_of: "05 Источники/Papers/PaddleOCR-VL v1.md"
last_verified: 2026-11-11
primary_sources:
  - https://arxiv.org/abs/2510.14528v1
---

# PaddleOCR-VL v1

Cheng Cui et al., PaddlePaddle Team, Baidu. [Report v1, 16 October 2025](https://arxiv.org/html/2510.14528v1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

[Section 2.1 and Figure 2](https://arxiv.org/html/2510.14528v1#S2.F2)
separate page layout from recognition. PP-DocLayoutV2 detects regions,
classifies them, and predicts reading order. PaddleOCR-VL-0.9B recognizes
individual crops under task-specific instructions. A postprocessor
assembles Markdown or JSON.

- [§2.1.1, Figure 3](https://arxiv.org/html/2510.14528v1#S2.F3):
  RT-DETR supplies boxes and classes; the ordering network scores pairs of regions.
- [§2.1.2, Figure 4](https://arxiv.org/html/2510.14528v1#S2.F4):
  a NaViT-style encoder, a projector with patch merging, and ERNIE-4.5-0.3B
  recognize crops.
- [§4](https://arxiv.org/html/2510.14528v1#S4) reports page parsing and
  element recognition separately. Neither score measures exact invoice
  field accuracy on its own.

[Transformers v5.14.0](https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/pp_doclayout_v2/image_processing_pp_doclayout_v2.py#L85-L111)
sums sigmoid probabilities rather than rounded decisions. The binary
values in Figure 3 illustrate how those totals give a reading order.

[[en/00 Textbook/16 Multimodal Models/64d Documents, OCR, and visual grounding#doc-paddleocr-reading-order|Worked example: reading the regions of an invoice]].
