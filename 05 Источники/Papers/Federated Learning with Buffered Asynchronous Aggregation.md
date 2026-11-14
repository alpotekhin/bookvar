---
title: Federated Learning with Buffered Asynchronous Aggregation
type: source-note
status: canonical
last_verified: 2026-11-09
source_url: https://proceedings.mlr.press/v151/nguyen22b.html
authors: [John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour, Mike Rabbat, Mani Malek, Dzmitry Huba]
year: 2022
license: CC BY 4.0
source_language: en
---

# Federated Learning with Buffered Asynchronous Aggregation

John Nguyen et al. AISTATS 2022, PMLR 151, pp. 3581–3607.
[Publication and PDF](https://proceedings.mlr.press/v151/nguyen22b.html).

- [Figure 2, PDF p. 3](https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf#page=3):
  the number of active clients falls at the end of a synchronous round,
  while clients can join independently in an asynchronous system.
- [Section 3 and Algorithms 1–2, PDF p. 4](https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf#page=4):
  clients start from the server model available when they join. The server
  takes a step after collecting a buffer of $K$ updates. Buffer size and
  the number of concurrently training clients are separate parameters.
- [Section 5, PDF p. 6](https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf#page=6):
  the authors describe an optional weight that decreases with the number
  of server steps since the client received its starting model.

The [[00 Учебник/19 Deployment, Reliability и MLOps/03 Edge и federated deployment#edge-fedbuff|Bookvar chapter]]
walks through two server steps and shows why they combine deltas rather
than final local weights. Its arithmetic is a teaching example, not a
reproduction of the paper's benchmarks.

Figure 2 is cropped from the published PDF without redrawing. The source
hash and crop coordinates are in the asset registry. The
[PMLR publication agreement](https://proceedings.mlr.press/pmlr-license-agreement.html)
specifies CC BY 4.0.
