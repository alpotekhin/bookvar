# Harvard checkpoint trade-off and fragmentation

Both PNGs are complete, unchanged originals from the rendered Harvard ML
Systems Book. SHA-256 matches both the earlier import manifest and the
live source downloaded on 2026-10-23. No crop, redraw, resampling or
translation was performed. Existing imported PNGs remain the source assets.

The pinned commit 45ecc8d82fcae70c149cdce550d3b3d3411df913 identifies the QMD and
generating Python, not an immutable URL for the rendered PNG. Both full QMDs
and the pinned CC BY-NC-SA 4.0 notice are retained in
Originals/fault-tolerance-2026-10-23. The public license notice is
licenses/harvard-45ecc8d8-LICENSE.md.

- Checkpoint plot: source code uses write time 2.5 minutes, MTBF 2 hours,
  and C/T + T/(2M). The original vertical-axis wording is approximate;
  the adjacent caption explains that these are the chapter's first-order
  costs, not an exact fraction of elapsed wall time. This is a Young plot,
  not the higher-order Daly curve. The earlier misleading hand-drawn SVG
  is not restored.
- Fragmentation: source row_patterns has 64 slots, 45 occupied, 19 free;
  no node has 8 free slots. The full-node placement restriction is explicit,
  not assumed for every distributed job.

Author: Harvard Edge ML Systems Book contributors. CC BY-NC-SA 4.0.
Direct page/image links and exact hashes are in assets.json.
