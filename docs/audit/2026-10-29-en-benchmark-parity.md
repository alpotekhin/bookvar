# English benchmarking chapter

The parent editor read the 914-line canonical chapter, the previous 291-line English chapter, and the complete 924-line replacement. The English version now contains the request-record example, token/chunk distinction, joint goodput denominators, percentile estimators and censoring, generator-side queueing, correlated lengths and prefix-cache regimes, phase accounting, energy per acceptable result, sustainable-load curves, and multiwindow SLO alerts.

All eight Python blocks and six canonical figures are retained. The full existing English energy section, including its licensed Harvard quotation, is byte-identical. The former cropped DistServe figure is replaced by the accepted complete canonical figure with its historical experimental conditions.

`publishing/tools/check_english_benchmark_parity.py` passed all eight actual Markdown blocks and 600 independent timestamp, context-work, and energy-integration cases. These are CPU calculations, not an executed serving or power benchmark. The frozen paths and hashes are in [the JSON journal](2026-10-29-en-benchmark-parity.json). Integrated rendering is recorded separately.
