# Serving engines: source-text rebuild, 2026-10-14

## Request and scope

The user rejected the prose of the open serving-engines page and explicitly
asked to use existing explanations instead. This supersedes the prose
acceptance in 2026-10-14-serving-engines.md. A successful build of that earlier
draft did not establish satisfactory writing.

Deliverable: canonical chapter 55c and its existing local route
http://127.0.0.1:4330/textbook/inference/serving-engines/.
No EN edits, commit, push or deployment. Existing unrelated work is preserved.
This is not completion of the whole textbook or acceptance of chapter 56.

## What changed

- Removed the custom five-layer introduction and most synthetic product prose.
- Integrated English original passages directly into the chapter, not as
  iframes, separate course embeds or a list of external links.
- vLLM: complete V1 Process Architecture, Worker/Model Runner/Model, and the
  full explanation of sharding/quantization at initialization.
- SGLang: original RadixAttention mechanism and overlap-scheduler explanation,
  explicitly dated 2024. Current server API is linked separately.
- TensorRT-LLM: complete Architecture Overview body, including runtime
  optimizations, both code excerpts and the original architecture image.
- FlashInfer: NHD/HND, ragged tensors, mask layout, page table, MLA and FAQ.
  Legacy mask API subsection using begin_forward is explicitly excluded; the
  plan/run names are linked to the pinned implementation. Cascade inference
  is not claimed as included.
- Preserved two clearly labeled editorial examples. Only the NumPy example is
  executed; original GPU snippets are reproduced as source examples.
- Seven original figures, four newly added unchanged files. Existing original
  images retained. Source author, link and license are adjacent to each figure.
- Pinned original files, license notices and the exact excerpt manifest:
  05 Источники/Originals/serving-engines-2026-10-14/.

The main agent read all selected source sections and personally viewed all
four new image files. It inspected the resulting page, including the narrow
screen rendering and native-resolution opening of the wide SGLang diagram.

## Executed evidence

- Source reproduction checker: all 10 excerpts match their saved sources;
  3097 whitespace-delimited text/code tokens unchanged after markup
  normalization. The five original code blocks also compare exactly
  (apart from RST indentation and final whitespace). Separate added captions
  are explicitly excluded, not mistaken for source prose.
- CPU example checker: actual marked Markdown block, independent scalar
  reference, request isolation, bad-mask control, NHD/HND permutation and
  malformed offsets PASS. Maximum reference error 2.220e-16.
- Unit tests: 236/236 PASS, 15:19 MSK.
- Astro check: 0 errors, 0 warnings, 0 hints.
- Final reader build: 1961 pages, completed 15:17:21 MSK.
- Built-output tests: 12/12 PASS.
- Built links: 0 broken internal routes, fragments or files.
- git diff --check PASS; no tracked en/ changes.

Final chapter SHA256:
1125281504c3d757d7c5951325b37ce108accf66979d68cded83949d6750e5e8

Source checker SHA256:
ac202ada35887b9cfafeec78caddbb6c4fbfbe45c744b973629a2a8593e5d011

CPU checker SHA256:
7c563e9a85ecdfbd43c70eafd9f27e2b15fc432e266809277d6885a3db6b9e60

## Browser inspection

1280x900 and 390x844:
- Seven images loaded at original aspect ratios, on white backgrounds.
- No document-width overflow; long code scrolls within the code container.
- No KaTeX errors; six fenced Python blocks render as code.
- Visually inspected original vLLM prose, FlashInfer page table, SGLang timeline
  and mobile mask diagram. No internal cropping of figures.
- The SGLang timeline labels are too small in the mobile inline preview.
  Opening the original and clicking again displays its native 3314x334 pixels,
  with horizontal panning. This is not claimed as readable at 343px width.
- Temporary viewport reset; user tab left at the revised chapter.

## Existing-batch bookkeeping and outstanding work

The previous batch left two FA3 images unregistered. Their existing metadata
was integrated, including a strict pinned source-archive member/hash check and
negative controls. This records provenance; it is not editorial acceptance of
FlashAttention.

The coverage check exposed four obsolete Harvard Performance Engineering rows
still pointing to chapter 56. They are now explicitly re-review /
integration-withdrawn-2026-10-14, with the proposed destination preserved.
No source content was deleted and no token source marker was inserted to
pretend coverage. The full chapters, slides and labs remain available.

Coverage after reconciliation: 210 matrix rows, 58 linked destinations,
215 Scikit-learn MOOC pages / 212 figures / 7 linked canonical pages,
34 Harvard labs, 35 slide decks, 67 HSE artifacts. This is a structural
connectivity check, not proof that all course knowledge has been integrated.

The rest of the textbook has NOT been rebuilt into this source-based format
in this turn. Chapter 56 and the pending Harvard placement remain to review.
No real serving engine, GPU inference, HTTP generation or performance
benchmark was run. Upstream speedup claims are attributed, not reproduced.
