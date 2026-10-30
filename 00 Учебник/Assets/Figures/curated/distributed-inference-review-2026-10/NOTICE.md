# Harvard pipeline request-flow panels

Author: Vijay Janapa Reddi and Harvard ML Systems Book contributors.
Source: `pipeline-parallel-routing.svg`, revision
`45ecc8d82fcae70c149cdce550d3b3d3411df913`:
[pinned original](https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/inference/images/svg/pipeline-parallel-routing.svg).

License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
This adapted viewport retains the same license and attribution.

Original SHA-256:
`489af6b438c5de04e4bb76efe7ab58010aa6985378300936a7fef4ef92bb90b5`.
Displayed SVG SHA-256:
`ad059d3ee911f5a22a9afe79034f664b4b5d637fbcfa340a96567195505408ea`.

The only change is `viewBox="0 0 680 480"` to `viewBox="0 0 680 365"`.
Both complete panels (a) and (b), their titles, devices, layer ranges, arrows,
requests, idle slots and pipeline-fill annotation remain visible. The footer
below y=365 is excluded. No text, path, color, geometry or label was rewritten.
The full source file remains unchanged in `curated/ml-systems/harvard/inference/`.

Reason for the excerpt: the original footer states four times single-stage
throughput for a four-stage pipeline. With equal stage time tau the steady-state
rate is 1/tau, or four times the unpipelined full-model rate 1/(4*tau), not four
times the rate of one stage. Chapter 58a provides an explicit finite schedule,
its fill/drain cost, and the autoregressive dependency instead of using that claim.

`publishing/tools/check_distributed_inference_examples.py` checks that the SVG
is exactly the recorded viewport-only transformation of the pinned source.
