---
type: editorial
status: editorial
last_updated: 2026-09-15
---

# Scaling-laws render repair

Status: canonical-source P1 repaired and targeted renderer checks PASS; primary owns full-site regeneration and the global HTML math scan.

## Ownership and reading

Changed only 00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md plus this report and its snapshot. All 825 original lines were read before editing. The full revised chapter was reread afterward in contiguous ranges; two final sentence corrections were checked separately.

No existing authored EN counterpart is declared in publishing/navigation.yml or present under en/. The generated English route is a Russian fallback; no separate source translation was invented or edited.

Exact pre-edit snapshot:
.superpowers/sdd/2026-09-15-close-editorial-findings/snapshots/scaling-render-before/00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md

No shared adapter, registry, navigation, source map, raw archive or generated-site file was edited. No full build, git operation or new agent was used.

## Root cause and red–green evidence

The DeepSeek learning-rate/batch fit opened a display block with two dollar signs but closed it with one. The next genuine display delimiter therefore closed this block instead of opening the next formula. Following prose and images became math, and later formulas alternated with prose treated as math through the end of the chapter.

Observed before editing:

- site/dist/textbook/pretraining/scaling-laws/index.html had 6 katex-error elements.
- The first error included the DeepSeek fit, the lone dollar sign and subsequent explanatory paragraphs.
- Further errors contained μP prose, image Markdown and the final source section.
- A direct rendering of canonical Markdown using the configured unified/remark-math/rehype-katex stack failed the no-katex-error assertion, reporting 7 errors and only 15 headings. This count differs from built HTML because the canonical source still contains Obsidian embeds rather than the adapter's Markdown-image output.

A one-character in-memory repair, before file modification, produced 0 errors and restored all 22 headings. The same minimal change was then applied to the snapshotted source. Language work followed as a separate pass, with a fresh renderer test afterward.

Diagnostic implementation note: JavaScript string.replace with a replacement string interprets a pair of dollar signs as one literal dollar sign. The repair must use a replacement callback or apply_patch. This was verified during the in-memory test and communicated to primary; no parser workaround was added.

## Editorial follow-through

Reworked mixed Russian-English prose in the optimizer and μP sections and their surrounding WSD, DeepSeek, experimental-design and concluding discussion, preserving mechanisms and qualifications.

Notable clarifications:

- A small model trained longer than the compute-optimal allocation is not necessarily statistically overfit.
- Optimizer comparisons need method-specific tuning and a disclosed search budget.
- μP transfers a base learning-rate multiplier with parameter-group rules, not identical raw updates for every tensor.
- The rank-one SGD update is explicitly for one example.
- Width-transfer failures involving RMSNorm gains, Lion and weight decay are scoped to the cited experiments.
- Kept worst-case assumptions of the simplified spectral-norm derivation and the distinct SGD/Adam scaling rules.
- Preserved all empirical coefficients, examples and mathematical derivations.
- Corrected the Lingle citation title to the verified v1 title, A Large-Scale Exploration of μ-Transfer.

Primary sources checked for these clarifications:
[Lingle v1](https://arxiv.org/html/2404.05728v1) and
[DeepSeek LLM v1 Figure 3](https://arxiv.org/html/2401.02954v1#S3.F3).
This task does not claim to have independently reproduced the training campaigns or proven the general μP theorem.

## Final checks

Targeted real renderer used the same Markdown stack as site/astro.config.mjs:
@astrojs/markdown-remark unified processor, remark-math and rehype-katex.
For image visibility checking, canonical image embeds were converted in memory to Markdown images; no generated file or asset was written.

Final output:

- katex-error elements: 0.
- Rendered headings: 22, including μP and Первоисточники.
- Rendered images: 37.
- Original figure embeds preserved: 37/37.
- Explicit anchors preserved: 15/15.
- Original headings preserved: 22/22.
- Display-formula bodies preserved: 17/17, compared with the baseline after only the delimiter correction.
- No figure Markdown or final source heading leaked into KaTeX annotations.
- site command npm exec vitest run tests/markdown-rendering.test.ts: 1 test file / 3 tests PASS.

CPU numerical checks, /private/tmp/bookvar-editorial-venv/bin/python with PyTorch 2.8.0:

- DeepSeek C=1e20: eta=0.0009859981744405008; batch=1017144.9599051544 tokens; 248.32640622684434 sequences at length4096.
- Worked extrapolation: N=18e9; D=277777777777.7778; D/N=15.432098765432098.
- Same-budget N interval14–24b gives357142857142.8571–208333333333.33334 tokens.
- Autograd single-example linear-layer gradient equals the analytic outer product and has rank1.

These are scalar/CPU checks, not training, performance measurements or a native GPU run.

## Reproducible renderer check

Run from site with Node in module mode. This calls the real renderer, not a regex-only delimiter counter.

    import fs from 'node:fs';
    import assert from 'node:assert/strict';
    import {unified} from '@astrojs/markdown-remark';
    import remarkMath from 'remark-math';
    import rehypeKatex from 'rehype-katex';

    const file = '../00 Учебник/11 Pre-training и Scaling/43 Scaling laws.md';
    const source = fs.readFileSync(file, 'utf8')
      .replace(/^---\n[\s\S]*?\n---\n/, '')
      .replace(/!\[\[([^\]\n]+)\]\]/g,
        (_, target) => '![Original figure](/assets/' + encodeURI(target) + ')');
    const renderer = await unified({
      remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex]
    }).createRenderer({});
    const {code} = await renderer.render(source, {
      fileURL: new URL(file, import.meta.url)
    });
    assert(!code.includes('katex-error'));
    assert.equal((code.match(/<h[1-6]\b/g) || []).length, 22);
    assert.equal((code.match(/<img\b/g) || []).length, 37);
    assert.match(code, /<h2[^>]*>μP: перенос по ширине и его границы<\/h2>/);
    assert.match(code, /<h2[^>]*>Первоисточники<\/h2>/);

## Hashes and remaining gate

- Before snapshot SHA256: c9c536acb67bbcb21eb79b463a548e3d2e7462fac8a748944b33e06354f546f7.
- Current chapter SHA256: 082fcc2b81f7d505f4207e5e81c17a7d280e36abbe2ed649da8afb8dac98567d.

The source is ready for primary regeneration and independent rendered review. Existing dist was diagnostic evidence, not an output modified by this task. The systematic-debugging and test-driven-development workflows caused a failing real-render check before the repair; verification-before-completion caused the final render, preservation and numerical checks before readiness reporting.
