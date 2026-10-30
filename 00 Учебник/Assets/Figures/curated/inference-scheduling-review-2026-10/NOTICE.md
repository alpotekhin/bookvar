# Original illustrations for chapter 55b

Verified 2026-10-14. All three image files are unchanged upstream raster files:
no redraw, translation, recoloring, rescaling, or crop has been applied. Original
transparency (vLLM) and white margins (LMSYS) remain.

## vLLM contributors — Apache License 2.0

Pinned repository: [vllm-project/vllm, 23fbc2ec2f7219888bfc76cf987381f501151971](https://github.com/vllm-project/vllm/tree/23fbc2ec2f7219888bfc76cf987381f501151971).

The root [LICENSE](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/LICENSE)
is retained in full as [LICENSE-vllm-Apache-2.0.txt](LICENSE-vllm-Apache-2.0.txt).
The pinned repository tree contains no closer LICENSE or NOTICE for these
documentation assets. The accompanying explanatory documentation is
[Prefix Caching](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/design/prefix_caching.md).

- `vllm-prefix-cache-overview.png` (743 × 331): original
  [overview.png](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/assets/design/prefix_caching/overview.png).
  SHA-256: `44f3000fcd73ec08ad66183ca8b3e0901e4b52044cf501f19fa8839391ac4e6a`.
- `vllm-prefix-release-shared-blocks.png` (844 × 301): original
  [example-time-5.png](https://github.com/vllm-project/vllm/blob/23fbc2ec2f7219888bfc76cf987381f501151971/docs/assets/design/prefix_caching/example-time-5.png).
  SHA-256: `12043b8bd2b1bf7a03d6b5a2723c66276f6e18ae45aa29b3d7824e31c7a8035e`.

The second filename is the upstream file name, not a newly assigned simulation
step: the documentation introduces it after its fourth event, Request 0 finishing.
Chapter caption follows the event and block numbers rather than guessing from
the filename. Both PNGs use black labels over transparency and require a light
display background.

Full license file SHA-256:
`c95bae1d1ce0235ecccd3560b772ec1efb97f348a79f0fbe0a634f0c2ccefe2c`.

## LMSYS Org / SGLang authors — MIT License

Pinned source: official website-and-blog repository
[lm-sys/lm-sys.github.io, 88c036dd1a78c8f2525782f98ee4493a63d7a532](https://github.com/lm-sys/lm-sys.github.io/tree/88c036dd1a78c8f2525782f98ee4493a63d7a532).
The root [LICENSE](https://github.com/lm-sys/lm-sys.github.io/blob/88c036dd1a78c8f2525782f98ee4493a63d7a532/LICENSE)
is MIT and contains two notices:

- Copyright (c) 2023 LMSYS Org.
- Copyright (c) 2022 LAION AI.

Both notices and both full grants are retained in
[LICENSE-LMSYS-MIT.txt](LICENSE-LMSYS-MIT.txt). This provenance refers to the
website's own image, not to a presumed license for the paper PDF or to the
separate SGLang runtime repository.

`sglang-radix-tree-original.jpg` (3212 × 3208) is the unchanged
[radix_attn_preview.jpg](https://github.com/lm-sys/lm-sys.github.io/blob/88c036dd1a78c8f2525782f98ee4493a63d7a532/public/images/blog/sglang/radix_attn_preview.jpg).
It is the author's compact tree preview accompanying
[Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/).
Full original branch labels, orange evictions and outer white margins retained.
It is not a crop made by Bookvar.

Image SHA-256:
`408a78d48a1efcd4cec9fb6b10969730adaff8b4672821d10f8bdec5b62b2b80`.

Full license SHA-256:
`324daad511da8c35aed6a4cffb7fbc0cae10295eca9f42086b5bbc806e017014`.

## Reproduction and integration

Raw download URLs, native dimensions, exact hashes, license copies and
central-registry proposals are in [assets.json](assets.json). Images are direct
downloads; no render or crop command is required. The CPU checker verifies
image bytes, native image headers and license-copy hashes.

Existing files removed from the reader chapter have not been deleted. Five
`used_in` updates are proposed in `assets.json`; explanatory links live in
`docs/audit/2026-10-14-inference-scheduling.md`. The main editor owns registry,
public-license integration, site rendering and deployment.
