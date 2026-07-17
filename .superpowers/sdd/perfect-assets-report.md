# Perfect assets report

Date: 2026-07-17

## Delivered

- Copied the three images emitted by the publication spike from the ignored
  `raw/papers/attention-is-all-you-need/images/` tree to the tracked canonical
  `00 Учебник/Assets/Figures/attention/` directory.
- Updated only the published self-attention chapter's three embeds.
- Added `05 Источники/asset-registry.yml`, covering all 32 figures tracked at
  the start of the work plus the three newly packaged attention figures.
- Recorded Jay Alammar, the exact *The Illustrated Transformer* page URL,
  CC BY-NC-SA 4.0, its license URL, copy status, and use for all three attention
  figures. Existing figures with incomplete evidence are explicitly marked
  `needs-confirmation`; missing exact values are `null`, not inferred.
- Added a manifest policy that rejects publication sources located in ignored
  `raw/`, plus a registry test that compares every figure file with the
  per-asset inventory.

## Verification evidence

- `npm test` in `publishing/`: 8 files, 47 tests passed.
- Byte comparison with `cmp`: all three tracked copies match their raw source.
- `git diff --check`: passed.
- Detached clean worktree: all 47 unit tests passed; publication generator
  passed; `astro check` passed with 0 errors, warnings, or hints.
- Detached clean-worktree `astro build` reached static entrypoint compilation,
  then failed because reusing `site/node_modules` through a symlink made Astro
  resolve the Starlight module from the original worktree while virtual-module
  metadata belonged to the clean worktree (`No cached compile metadata`). This
  is a dependency-reuse/path blocker, not a missing-content or asset error.

## Remaining metadata gaps

The 32 pre-existing figures have inventory coverage, but 32 entries remain
`needs-confirmation` (the two repository-authored SVGs are included). Their
exact author, direct asset URL, redistribution license/license URL, and/or
extraction history must be confirmed before treating the inventory as a
whole-repository redistribution clearance. The three Jay Alammar entries are
`verified`.

## Clean-worktree status

A genuine detached clean worktree at this asset commit passed tests, generation,
and Astro checking. A full Astro production build still needs either a local
dependency install in that worktree or a dependency layout that does not cross
worktree paths; the available cache could only be reused by symlink and Astro's
virtual-module cache rejects that split path. The working tree's concurrent
manifest expansion is intentionally outside this scoped commit.
