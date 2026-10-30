# Fault-tolerance illustration

`gang-deadlock.svg` is an unmodified illustration by Harvard Edge ML Systems
Book contributors, from Fleet Orchestration.

- Original: https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/fleet_orchestration/images/svg/gang-deadlock.svg
- Commit: `45ecc8d82fcae70c149cdce550d3b3d3411df913`.
- Licence: CC BY-NC-SA 4.0; https://creativecommons.org/licenses/by-nc-sa/4.0/
- Pinned declaration: https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/LICENSE.md
- SHA-256: `11eb48297257ed819ca48e2c31bf8097a5eef3e3a441fbed7e88e123b99c32cd`.
- Retrieved and declaration checked 2026-10-13. No changes to labels or geometry.
- Used in 44h Fault tolerance и fleet orchestration.md. The all-or-nothing
  admission example assumes a fixed-size worker group, not elastic resizing.

`failure-types.svg` is an unmodified illustration by the same contributors,
from Fault Tolerance, at the same commit and under the same licence.

- Original: https://github.com/harvard-edge/cs249r_book/blob/45ecc8d82fcae70c149cdce550d3b3d3411df913/book/quarto/contents/vol2/fault_tolerance/images/svg/failure-types.svg
- SHA-256: `4e4c167ee73d53a7e8f3fd12c28a2d279e24626d3e928f262c62049fa93586f2`.
- Retrieved 2026-10-13. No modifications. Chapter caption explains the
  centralized drawing is not a DDP topology and the limits of timeout/checksum.

Two alternative originals were rejected after visual inspection:
young-daly-optimization.svg draws the supposed sum below one component;
checkpoint-recovery-timeline.svg marks a checkpoint saved before its save.
Neither is embedded in the revised RU 44h. The pre-existing Young-Daly
source file remains available for provenance/other consumers.
