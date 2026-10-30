# Existing scaling illustrations: chart-only crops

Prepared and visually reviewed 2026-10-13 for the local Russian editorial pass.
No redraw, recoloring, smoothing, OCR replacement, or invented data.
Each PNG is a crop of a fresh 2400-pixel-long-edge render of the pinned
original course PDF, not an upscale of the old 1600-pixel screenshot.
Axes, plotted observations, legends, and experimental annotations are retained.
Slide titles and prose duplicated in the chapter are removed.

## Rights and publication boundary

The existing figure registry records these source slides as
`User permission record (not a named license)`. That record is retained;
it is not independent permission from each third-party figure's rightsholder.
This pass only reformats illustrations already used in chapter 43.
It introduces no additional third-party figures with unknown conditions.

The MIT license verified for the pinned CS336 Assignment 3 software and
documentation is not applied to lecture PDFs or the papers shown inside
them. The pinned lectures repository does not expose a `LICENSE` file at
the checked URL. Do not describe all of these charts as MIT-licensed or
`source-license-verified`.

MiniCPM's paper, arXiv:2404.06395v1, explicitly carries CC BY 4.0; this
supports the three MiniCPM WSD/joint-fit figures listed below. The
`minicpm-cosine-campaign-cost` filename is legacy naming: its underlying
figure is from Chinchilla, not MiniCPM. Chinchilla, DeepSeek LLM, and the
other third-party figures are not promoted to an independently verified
reuse license by this revision. The parent reviewer owns the central
registry and publication review.

## Shared source provenance

- Course author: Tatsunori Hashimoto and Stanford CS336 course staff.
- Repository commit: `8b59b50730766695c2ffedd1a79c50cd09b9eb91`.
- Lecture 9: https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_09.pdf
- Lecture 9 SHA-256: `92a5d9e26e65bb0a9d8c79b12ed7ea26d14530bf960dcf44f9ed5fd372f519fa`.
- Lecture 11: https://github.com/stanford-cs336/lectures/blob/8b59b50730766695c2ffedd1a79c50cd09b9eb91/lecture_11.pdf
- Lecture 11 SHA-256: `b2148e683b4020f3fa95f5c95f37be55ebf6b2f57382f4c148517e84bf1df890`.
- Render command: `pdftoppm -f PAGE -l PAGE -singlefile -scale-to 2400 -png PDF PREFIX`.
- All pages rendered to 2400 × 1352 pixels.
- Crop coordinates below are `(left, top, right, bottom)` in that render.
- Used in: `00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md`.

## Underlying works

- Kaplan et al., *Scaling Laws for Neural Language Models*, https://arxiv.org/abs/2001.08361: `data-power-law`, `parameter-count-convention`, `critical-batch-vs-loss`.
- Tatsunori Hashimoto, *Model Performance Scaling with Multiple Data Sources*, https://proceedings.mlr.press/v139/hashimoto21a.html: `distribution-shift-offset`.
- Muennighoff et al., *Scaling Data-Constrained Language Models*, https://arxiv.org/html/2305.16264v1: `repeated-data-effective-tokens`.
- Kim, Kotha, Liang, Hashimoto, *Pre-training under infinite compute*, https://arxiv.org/abs/2509.14786: `finite-data-breakdown`.
- Hoffmann et al., *Training Compute-Optimal Large Language Models*, https://arxiv.org/html/2203.15556v1: the three Chinchilla method charts and `minicpm-cosine-campaign-cost`.
- Besiroglu et al., *Chinchilla Scaling: A replication attempt*, https://arxiv.org/html/2404.10102v1: `chinchilla-residuals`.
- Yang et al., *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, https://arxiv.org/abs/2203.03466: `learning-rate-scaling` (the adjacent table from Yao et al. is not part of this crop).
- Hu et al., *MiniCPM*, https://arxiv.org/html/2404.06395v1, CC BY 4.0: `minicpm-wsd-branching`, `minicpm-wsd-loss-dynamics`, `minicpm-joint-scaling-fit`.
- DeepSeek-AI, *DeepSeek LLM*, https://arxiv.org/html/2401.02954v1: the three DeepSeek crops.
- StepFun and collaborators, *Predictable Scale: Part I*, https://step-law.github.io/: `optimizer-search-surface`.
- Wen, Hall, Ma, Liang, *Fantastic Pretraining Optimizers and Where to Find Them*, https://arxiv.org/abs/2509.02046: `optimizer-unfair-tuning`, `optimizer-compute-ratio-confound`.

## Per-file evidence

| File | Lecture / page | Crop pixels | Dimensions | SHA-256 |
|---|---|---|---|---|
| `data-power-law-chart.png` | 9 / 15 | `636, 473, 1524, 1169` | 888 × 696 | `f1deb03b93052b6cbf0ca4ab2807ef8c97737992f672962e71fe1e9431d7c19a` |
| `distribution-shift-offset-chart.png` | 9 / 22 | `276, 723, 2256, 1183` | 1980 × 460 | `f42535169371c55efef876c2b0c2d0f7cf184ea32c4811dd08b9879d56622fcc` |
| `repeated-data-effective-tokens-chart.png` | 9 / 24 | `276, 379, 2316, 1129` | 2040 × 750 | `7865d1d22654c41304f524cd0b5cf5238c3dc5a6cd5f5da3b7941a29653e8a81` |
| `finite-data-breakdown-chart.png` | 9 / 25 | `108, 500, 2364, 1217` | 2256 × 717 | `809739c21bcdcf7c71f8bb169852ff5f2142e68fd69df722dbfed7d9e3015f57` |
| `parameter-count-convention-chart.png` | 9 / 35 | `300, 426, 2352, 1055` | 2052 × 629 | `3fe9a90fa7bb2ff302f99a16a3351d2db7da6cb9439980f8baf028728637129d` |
| `chinchilla-lower-envelope-chart.png` | 9 / 47 | `348, 460, 2184, 946` | 1836 × 486 | `3a727fbd7b1e6954385f18062082d4570dac1352117c37bb4dc58a2a87039285` |
| `chinchilla-isoflops-chart.png` | 9 / 48 | `180, 399, 2220, 994` | 2040 × 595 | `d1d3c4ea710ff85545450f6230dbc24f9c0dce418b3f5c3f440541c1df974367` |
| `chinchilla-joint-fit-chart.png` | 9 / 49 | `312, 433, 2304, 1088` | 1992 × 655 | `6e848739c66ce799a81a29a12084db9bc9b84527b2baf09ad4337eedd5b7ef28` |
| `chinchilla-residuals-chart.png` | 9 / 53 | `216, 487, 2256, 1197` | 2040 × 710 | `82bec92ccc8efa37be469093ba6c6e92996f3d352650967ef234a2e78bf1aa8d` |
| `critical-batch-vs-loss-chart.png` | 9 / 39 | `444, 223, 1896, 1075` | 1452 × 852 | `696e5e1e4756e7fe214b4094ba30619d31f9fcea1b17b200b883f92f14bb501c` |
| `learning-rate-scaling-chart.png` | 9 / 40 | `29, 311, 1296, 919` | 1267 × 608 | `93cdc081f261fd46168acb4ef2a7ea239d41a060be03eae9756fa8f38fd8f853` |
| `minicpm-cosine-campaign-cost-chart.png` | 11 / 13 | `468, 331, 1920, 1203` | 1452 × 872 | `65868ec00a2ecfa32ba3036decbb99c2973f887dc2f16de9994ae5ad8e0041af` |
| `minicpm-wsd-branching-chart.png` | 11 / 14 | `240, 365, 1032, 886` | 792 × 521 | `ec3c96ffef95d1f99ce26784aac7e6a9b666aefb69ce5f538b7b5d252ca59eee` |
| `minicpm-wsd-loss-dynamics-chart.png` | 11 / 15 | `564, 338, 2076, 1351` | 1512 × 1013 | `053a3e2550767ad8ca53ff33b55044775f78423972f7eda16828ba2aeb8be0d7` |
| `minicpm-joint-scaling-fit-chart.png` | 11 / 18 | `588, 358, 2088, 1332` | 1500 × 974 | `fe24af9ea7d1643f7273dc149766efd7693b28f41907b92985a0b7101072c149` |
| `deepseek-lr-batch-grid-chart.png` | 11 / 20 | `552, 608, 2004, 1149` | 1452 × 541 | `d824366c0f34e6b3137ce3d3fd2fc9e0a9a833d574d40a4476b8138cefc880f2` |
| `deepseek-isoflop-curves-chart.png` | 11 / 23 | `204, 372, 2316, 1203` | 2112 × 831 | `01eab0c997dfb130b249b4a7a613dc77eaea77836557ae0a4e9a821974d4a880` |
| `deepseek-loss-prediction-chart.png` | 11 / 24 | `840, 588, 1764, 1142` | 924 × 554 | `77433ad6b7027c88ce7ab5f164073fd8f609866583eebd2ad7fec5097fdad7ba` |
| `optimizer-search-surface-chart.png` | 11 / 34 | `360, 324, 1512, 1284` | 1152 × 960 | `ba95f5c732ce8ce02aad6f428f474adc073846854fc204bc1fea7d060976980c` |
| `optimizer-unfair-tuning-chart.png` | 11 / 39 | `216, 291, 2352, 1068` | 2136 × 777 | `45554ed617ff9e2d280899c775e07249e51bf40965ab0a9d69f6fffb9b5bbcfa` |
| `optimizer-compute-ratio-confound-chart.png` | 11 / 40 | `96, 250, 2364, 1000` | 2268 × 750 | `b1627cc8a50a31f283a0652e388114d256cf399a623bc75134c864c603dff446` |
