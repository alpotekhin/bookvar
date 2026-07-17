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
- Generator attempt reached the expanded concurrent manifest and stopped on
  unrelated missing frontmatter: `Concepts/Architectures/DeepSeek-R1.md` lacks
  required `status`.

## Remaining metadata gaps

The 32 pre-existing figures have inventory coverage, but 32 entries remain
`needs-confirmation` (the two repository-authored SVGs are included). Their
exact author, direct asset URL, redistribution license/license URL, and/or
extraction history must be confirmed before treating the inventory as a
whole-repository redistribution clearance. The three Jay Alammar entries are
`verified`.

## Clean-worktree status

A genuine clean-worktree build is required after this scoped commit. The live
workspace contains concurrent navigation/site changes outside this work; those
changes currently block the generator on unrelated frontmatter before the site
build begins. The clean worktree should be created from this commit so it uses
the original 10-page spike manifest and tests the packaged attention assets in
isolation.
