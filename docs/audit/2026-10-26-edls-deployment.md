# EDLS Week 7 deployment walkthrough, 26 October 2026

## Current revision and acceptance boundary

`06 Практика/17 Развернуть наблюдаемый ML сервис.md`: 645 lines, SHA-256
`87f6c48330ce879804236099e5f248e61cdf101505b8e94523af5f7035a913a0`.
The previous complete 26-line assignment was read and is retained inside a
labeled disclosure. The new chapter was read in full in three consecutive
sections; its final inline-code changes preserve CLI hyphens and JSON quotes
against typographic substitution. The original route/index are preserved.

This review delivers a source-backed walkthrough and source-logic checks,
not a completed container/SLO experiment. The chapter intentionally remains
`in-review`, and E-07-P1–P7 are `partial-reviewed-2026-10-26`, not complete.

## Read sources and explanation check

61 pinned EDLS files were downloaded: the Week 7 tree excluding illustrative
dataset JPEGs, plus the full root MIT notice. Every byte sequence was verified
against its Git blob at `e632aa89ca9e6638d52e1b686095e7442faffbb0`.
`source-manifest.json` records Git SHA-1 and SHA-256. The instructional Python,
Dockerfiles, Compose files, protobuf, supervisor/monitoring configurations,
tests and README files were read in full. Labels and uv.lock are retained as
data/dependency metadata, not claimed as line-by-line prose reading.

All 15 slides' extracted text was read; pages 3, 5, 6, 8, 10, 12 and 14 were
inspected visually. The remaining playground/list pages did not supply a useful
mechanism figure. Logos and a low-resolution third-party HTTP screenshot were
not inserted as if they explained deployment. The external linked homework
repository was not read or integrated in this unit.

Nine complete original files are reproduced unchanged in English: application,
model, bad/good Dockerfiles, tracing script, metrics server, scrape config,
protobuf schema and supervisor config. A fidelity test caught a changed quote
in the initially hand-entered trace excerpt; the original double quotes were
restored. All nine final fences now match their source files exactly.

The explanation review covers:

- smoke-test shape versus valid classification; neutral-text randomness and
  swallowed malformed-input errors in the anti-pattern example;
- model loading, validation, readiness body versus status code, optional Redis
  and the daemon-thread failure boundary;
- seed 7's first jitter, including the arithmetic and erroneous source expected
  value, without silently modifying the source test;
- dependency-layer inputs, cache invalidation after COPY, tag/digest distinction
  and the actual frozen/locked difference checked in complete selected uv docs;
- client-side normalization, shorter-edge Resize, missing explicit RGB/shape
  bounds, tracing versus training and eval versus gradient disabling;
- success-only prediction counts, attempted invocation counts, in-progress
  cleanup after failure, direct scraping, Grafana's query address and ports;
- the absent nginx target and missing supervisor bind paths, container-visible
  Telegraf resources, label cardinality and per-stage measurement still absent;
- cumulative histogram values 1/2/4/4, sum 1.45, count 4 and SLO fraction 2/4,
  aggregate-before-quantile, bucket resolution and counter-rate distinction;
- protobuf shape/data, the 150,528-element double/float32 payload cost, process
  separation and remaining admission/deadline/input limits;
- the incorrect sibling-container port, missing imgs directory and protobuf
  destination-directory issue, distinguished from an executed Docker failure;
- separate model copies under supervision, startup priority versus routing and
  the lack of a load balancer to the second process.

The complete Docker cache/invalidation and Prometheus overview pages were read.
The histogram document's complete Count/sum, Bucketing and Quantiles/error
sections were read. Selected torchvision Resize and PyTorch eval documentation
was checked. No actual VGG tracing or pretrained inference is claimed. The
source-language code, reasoning, examples and limitations survived no-ai-slop
review; prose does not reduce the topic to API names or a link list.

## Figures, rights and actual page

Two complete originals were selected rather than replacement diagrams:

- Docker's 2742×855 cache-invalidation PNG, unchanged, SHA
  `7b1702be0b426d6cf2e7e4435ef9a8e65da6a80123044a6bcc273017980e7e07`,
  documentation commit `419ecb47185bf05c83f3cfb7e995b82266046c03`;
- Prometheus's 1351×811 architecture SVG, unchanged, SHA
  `c67cdad4edc76999d415a5321d1f8909ec9e8378b26a71e3f185e47c765a4a73`,
  documentation commit `3576591af32864770ad6ff73a14fc485f70afb73`.

Both complete Apache-2.0 licenses were read, retained and served. Prometheus's
full NOTICE, including SoundCloud attribution, is also retained and linked.
Docker's pinned tree has no root NOTICE. The course's full MIT notice is served.
Adjacent assets.json and global registry bind author, original bytes, source
page, commit, rights, license and dimensions. No crop, redraw or recompression.

Actual RU page:
`http://127.0.0.1:4330/practice/17-razvernut-nablyudaemyy-ml-servis/`.
At desktop 1280 px, Docker displays whole at 617×192 and Prometheus at 617×370.
At mobile 360 px they are 313×98 and 313×188; small labels require the viewer.
2× views were inspected and saved. Prometheus at 4× reached the right/bottom
extremes (x 987, y about 172 px); navigation back toward the left/top was also
exercised, but no exact 0/0-extreme measurement is claimed for that pass.
All eleven code fences scroll internally in 313 px containers, rather than
expanding the document. Document scroll width is 345 inside viewport 360;
no KaTeX errors. The SVG's foreignObject labels render correctly in the browser.

The actual EN route displays the explicit missing-translation notice and the
new source-language chapter/figures as fallback, with previous/next links to
EN practices 16/18. No EN body was edited. Screenshots:
`screenshots/2026-10-26-edls-deployment-prometheus-desktop.png`,
`screenshots/2026-10-26-edls-deployment-prometheus-mobile-zoom.png`,
`screenshots/2026-10-26-edls-deployment-docker-mobile-zoom.png`.

## Executed checks and exact gaps

The original pytest suite ran unchanged in a selected compatible Python 3.14
environment: four pass, one fails on the source's inconsistent expected jitter.
The complete diagnostic is retained. This is not a reproduction of uv.lock.
The new checker also executes in-process ASGI readiness/validation and bad-input
paths, then the actual Flask exporter with a stub VGG model. Three successful
predictions plus one injected failure produce +3 predictions, +4 invocations
and an in-progress gauge of zero. The histogram fixture also ran.

Canonical recording:
`06 Практика/Results/edls-deployment-2026-10-26/source-checks.json`, SHA
`b90d36ac62d7c95935ea8e2c613bf46d3e8c890f41e9d0d8a9eca7f9aee492cc`.
Checker SHA:
`7f505b0b0e44c9efc42f16e0dd51f2c5fa564d3c11022fa2fdcf9f3df992e057`.
The public recording is byte-identical. No weights were downloaded.

Final full unit run: 498 tests in 55 files passed. Build: 1967 pages. Final
built-output checks: 135 passed. Built links: zero broken routes/fragments/files.
Source coverage passed: 210 matrix rows and 56 direct-linked destinations;
that structural counter is not proof that all course knowledge is integrated.
Scoped tracked-file whitespace checks pass. Logs:
`/var/tmp/bookvar-edls-deployment-{unit,build,output,links,coverage}-final-2026-10-26.log`
and `/var/tmp/bookvar-edls-deployment-source-checks-2026-10-26.log`.

Docker info confirms no running daemon. Container build, real scrapes/dashboard,
gRPC transport, actual VGG predictions and the fixed-arrival load/SLO experiment
remain unexecuted. No Docker daemon or application stack was started. Those
checks are not replaced by the in-process fixture. Full university-course
integration and source/rights reconciliation also remain open. Accepted RL,
translations, staging, commit, push and deployment were not changed.
