# Task 4 report: wikilinks and Obsidian embeds

## Status

Implemented and committed from base `10dbc4a0b107104047ed71f0734858464a9649ef`.

Implementation commit: `73c53de7ecc7d57b57334f12045095f9169f90db`

## Scope

- Added pure `convertWikiSyntax(markdown, registry)` in `publishing/adapter/links.ts`.
- Added focused coverage in `publishing/tests/links.test.ts` for the five exact brief inputs.
- Source Markdown files were not modified.
- Unresolved page embeds remain visible as plain text and are returned in `unresolved`.
- Image handling is limited to `.png`, `.jpg`, `.jpeg`, `.webp`, `.svg`, and `.gif`.

## TDD evidence

### RED

The prescribed command was attempted first:

```text
pnpm --dir publishing test -- links.test.ts
zsh:1: command not found: pnpm
```

Because this did not exercise the test, the installed workspace binary was run directly:

```text
cd publishing
./node_modules/.bin/vitest run links.test.ts
FAIL tests/links.test.ts
Error: Cannot find module '../adapter/links.js'
Test Files 1 failed (1)
```

This is the expected RED: `convertWikiSyntax` did not exist.

### GREEN

After the minimal implementation:

```text
./node_modules/.bin/vitest run links.test.ts
Test Files 1 passed (1)
Tests 2 passed (2)
```

## Final verification

Commands run from `publishing/`:

```text
./node_modules/.bin/vitest run
Test Files 2 passed (2)
Tests 4 passed (4)

./node_modules/.bin/tsc --noEmit
exit 0

git diff --check
exit 0
```

## Self-review

- The function only transforms its string input and a local result array; it performs no file I/O and does not mutate the registry.
- Embeds are processed before regular links so the inner wikilink is not converted twice.
- Heading resolution queries the registry without the heading and appends a lower-case Unicode-aware slug; Cyrillic letters are preserved, whitespace becomes hyphens, and punctuation is removed.
- Link labels default to the target basename, while aliases are preserved exactly.
- Unsupported embed extensions are not treated as images; they follow ordinary link resolution and remain visible if unresolved.
- No source Markdown or unrelated file was changed in the implementation commit.

## Concerns

- `pnpm` is unavailable in this environment's `PATH`; verification used the existing local binaries under `publishing/node_modules/.bin` and required no network access.
- The brief does not require deduplication of repeated unresolved targets, so the implementation reports occurrences in encounter order.
