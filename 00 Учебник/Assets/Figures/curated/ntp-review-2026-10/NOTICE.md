# Original D2L illustrations for next-token prediction

Verified 2026-10-13; complete renders checked again on 2026-10-21.
Used in:

- `00 Учебник/11 Pre-training и Scaling/42 Next-token prediction.md`.
- `00 Учебник/03 Языковое моделирование/01 Вероятность текста и next-token prediction.md`.

Authors: Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola.
Work: *Dive into Deep Learning*.
Upstream repository: <https://github.com/d2l-ai/d2l-en>.
Pinned revision: `23d7a5aecceee57d1292c56e90cce307f183bb0a`.

License: **CC BY-SA 4.0**. Evidence is the repository's
[LICENSE at the pinned revision](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/LICENSE).
Its complete bytes are retained in `LICENSE-D2L-CC-BY-SA-4.0.txt`.
The local SVG originals and PNG format conversions retain this license;
they are not covered by any more permissive license for Bookvar code.
Bookvar and these files are not endorsed by D2L's authors.

No diagrams were redrawn, translated, cropped, or partially reconstructed.
Each PNG is a complete rasterization of the adjacent original SVG at 2000 px
width with a white background. Local SVG originals are byte-identical to the
pinned upstream files. The chapter contains a separate Russian explanation
and caption, with author, source-page, and license links.

## d2l-language-model-data.png

- Source page: <https://d2l.ai/chapter_recurrent-neural-networks/language-model.html#fig-lang-model-data>.
- Original source: <https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/lang-model-data.svg>.
- Saved source: `d2l-language-model-data-original.svg`.
- Original SHA-256: `b22c0c38c2d987ed2b4f7c8b394f8d8442e592e42c321f63775fa1d388840373`.
- PNG SHA-256: `96122632db141d7aed7457171a234e07b0d0cf944e6e5c4c0b0f169db04d0491`.
- Dimensions: 2000 × 292 px.
- Figure identity: D2L Figure 9.3.1, character-level input/target shift.
- Caption check: five-character segments, not word tokens; first paired
  segments are `e tim` and ` time`. The shift is one character.
- No external asset dependencies are required by the saved SVG.

## d2l-gpt-decoder.png

- Source page: <https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html#gpt-and-gpt-2>.
- Original source: <https://raw.githubusercontent.com/d2l-ai/d2l-en/23d7a5aecceee57d1292c56e90cce307f183bb0a/img/gpt-decoder-only.svg>.
- Saved source: `d2l-gpt-decoder-original.svg`.
- Original SHA-256: `c586784e0c1404f7fe0ffcda1354f02cfe64f708f80eb9c6e5fc1d32248ea964`.
- PNG SHA-256: `ab061b4721eac955985c274afaf476d9a4ee00aca2d9f6769bf9afdaa2c56d91`.
- Dimensions: 2000 × 1056 px.
- Figure identity: GPT language-model pretraining and causal dependencies.
- Caption check: attention output rows are labelled by the predicted target,
  and arranged from bottom to top. `like` has access to `<bos>` and `I`, not
  to the input `like`; the apparent triangular orientation is explained.
- No external asset dependencies are required by the saved SVG.

## Reproduction

With CairoSVG 2.9.1 and Cairo available, run from this directory:

```python
import cairosvg

for stem in ("d2l-language-model-data", "d2l-gpt-decoder"):
    cairosvg.svg2png(
        url=f"{stem}-original.svg",
        write_to=f"{stem}.png",
        output_width=2000,
        background_color="white",
    )
```

On the review machine, Cairo is in `/opt/homebrew/lib`; the command environment
used `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib`. PNG encoding can differ
with library versions; source SVG hashes are the authoritative byte identity.
Both final PNGs were personally inspected at readable size, including edges
and all labels. This does not establish their mobile/site rendering; the main
editor performs the publication UI check separately.

## Replaced illustrations

Remove only this chapter's `used_in` entries for:

- `curated/training-systems-courses/42-slp-chain-rule.png`
- `curated/training-systems-courses/42-slp-perplexity.png`

The first had a probability expression clipped at its right edge, and the
second had an expression clipped at its lower edge. Their image files and
other consumers are retained. Their mathematical information is now typeset
in the chapter, not lost with the screenshots.

The existing `curated/course-follow-through/mtp-overview.png` is retained,
with its existing CC BY 4.0 attribution and registry entry.
