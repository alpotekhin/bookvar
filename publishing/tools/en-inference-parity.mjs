import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { beforeTopicConsolidation } from './topic-consolidation-baseline.mjs';

const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const fencePattern = /^(```|~~~)[^\n]*\n[\s\S]*?^\1\s*$/gm;
const withoutCode = (text) => text.replace(fencePattern, '');
const normalizeSpace = (text) => text.replace(/\s+/g, ' ').trim();
const normalizePath = (path) => path.replace(/^02 Areas\/ML & DL\//, '');
const figures = (text) => [...text.matchAll(/^!\[\[([^\]|]+).*$/gm)]
  .map((m) => normalizePath(m[1]));
const headings = (text) => [...withoutCode(text).matchAll(/^(#{1,6}) (.+)$/gm)]
  .map((m) => ({ level: m[1].length, title: m[2] }));
const urls = (text) => [...new Set([...text.matchAll(/https?:\/\/[^\s<>\])]+/g)].map((m) => m[0]))];
const wikiTargets = (text) => [...new Set([...text.matchAll(/(?<!!)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)]
  .map((m) => normalizePath(m[1])))];
const maths = (text) => [...withoutCode(text).matchAll(/\$\$([\s\S]*?)\$\$/g)].map((m) => m[1]
  .replaceAll('бит/вес', 'bits/weight').replaceAll('на заданной сетке', 'on the chosen grid')
  .replaceAll('мс', 'ms').replaceAll('{,}', '.').replace(/\s+/g, ''));
const code = (text) => [...text.matchAll(fencePattern)].map((m) => m[0].trimEnd());

// The existing English packed-attention example already translated these
// two non-computational strings. Do not normalize other code differences.
const normalizeCode = (text) => text
  .replace('Разрешает читать предыдущие чужие запросы.', 'Incorrectly allows reading earlier requests.')
  .replace('Границы запросов сохранены; общая причинная маска даёт другой ответ.',
    'Request boundaries preserved; a global causal mask gives different outputs.');

function checkOriginalExcerpts(canonical, english) {
  let excerpts = 0;
  for (const match of canonical.matchAll(/<!-- source-excerpt:\s*([^\n]*?):start -->([\s\S]*?)<!-- source-excerpt:\s*\1:end -->/g)) {
    const [, id, body] = match;
    const enMatch = [...english.matchAll(/<!-- source-excerpt:\s*([^\n]*?):start -->([\s\S]*?)<!-- source-excerpt:\s*\1:end -->/g)]
      .find((m) => m[1] === id);
    if (enMatch) {
      assert.equal(enMatch[2], body, `Modified licensed excerpt: ${id}`);
    } else {
      // The pre-existing EN serving originals lack boundary comments. Verify
      // every complete original paragraph in order, excluding figure captions
      // (translated editorial material) and the independently checked embeds.
      const paragraphs = body
        .replace(/<!-- source-caption:start -->[\s\S]*?<!-- source-caption:end -->/g, '')
        .replace(/<!--[\s\S]*?-->/g, '').replace(/^!\[\[.*\]\]$/gm, '')
        .split(/\n\s*\n/).map(normalizeSpace).filter(Boolean);
      const target = normalizeSpace(english);
      let cursor = 0;
      for (const paragraph of paragraphs) {
        const found = target.indexOf(paragraph, cursor);
        assert.ok(found >= 0, `Missing/reordered original paragraph in ${id}`);
        cursor = found + paragraph.length;
      }
    }
    excerpts += 1;
  }
  for (const match of canonical.matchAll(/<!-- source-excerpt: ([^\n:]+) -->([\s\S]*?)<!-- \/source-excerpt -->/g)) {
    assert.ok(english.includes(match[0]), `Modified original excerpt: ${match[1]}`);
    excerpts += 1;
  }
  return excerpts;
}

export function inspectEnInferenceParity(root) {
  const report = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-en-inference-parity.json'), 'utf8'));
  assert.equal(report.assigned, 7);
  assert.equal(report.completed, 7);
  assert.equal(report.chapters.length, 7);
  const totals = { chapters: 7, figures: 0, displayMath: 0, codeFences: 0, originalExcerpts: 0,
    canonicalSections: 0, aliases: 0 };
  const revisions = [];
  for (const chapter of report.chapters) {
    const canonical = readFileSync(resolve(root, chapter.source), 'utf8');
    const english = readFileSync(resolve(root, chapter.path), 'utf8');
    // Keep the receipt's exact reviewed hashes. Reverse only the recorded
    // prerequisite and practice-navigation deltas; compare current bodies below.
    const receiptRevision = (text, source) => beforeTopicConsolidation(text, source, root,
      '2026-10-30-reader-provenance-additions.json');
    const reviewedCanonical = receiptRevision(canonical, chapter.source);
    const reviewedEnglish = receiptRevision(english, chapter.path);
    assert.equal(sha(reviewedCanonical), chapter.source_sha256, chapter.source);
    assert.equal(sha(reviewedEnglish), chapter.after_sha256, chapter.path);
    assert.ok(reviewedEnglish.includes(`canonical_revision: "${chapter.source_sha256}"`));
    assert.ok(english.includes(`canonical_revision: "${sha(canonical)}"`),
      `Current canonical revision: ${chapter.path}`);
    revisions.push({ source: chapter.source, path: chapter.path,
      canonicalSha256: sha(canonical), englishSha256: sha(english),
      receiptCanonicalSha256: chapter.source_sha256, receiptEnglishSha256: chapter.after_sha256 });
    assert.ok(english.includes('translation_status: current'));
    assert.deepEqual(headings(canonical), chapter.source_sections.map(({ level, title }) => ({ level, title })));
    assert.deepEqual(headings(english), chapter.english_sections.map(({ level, title }) => ({ level, title })));
    assert.deepEqual(maths(english), maths(canonical), `Display mathematics: ${chapter.path}`);
    assert.deepEqual(code(english).map(normalizeCode), code(canonical).map(normalizeCode),
      `Protected code: ${chapter.path}`);
    assert.deepEqual(figures(english), figures(canonical), `Source figures: ${chapter.path}`);
    for (const figure of chapter.figures) assert.equal(sha(readFileSync(resolve(root, figure.path))), figure.sha256);
    for (const url of urls(canonical)) assert.ok(urls(english).includes(url), `Missing source URL: ${url}`);
    for (const target of wikiTargets(canonical)) assert.ok(wikiTargets(english).includes(target), `Missing wiki target: ${target}`);
    const aliases = [...withoutCode(english).matchAll(/<(?:a|span) id="([^"]+)"/g)].map((m) => m[1]);
    assert.deepEqual(aliases, chapter.retained_aliases);
    const ids = [...aliases, ...chapter.english_sections.map((h) => h.anchor)];
    assert.equal(new Set(ids).size, ids.length, `Duplicate IDs: ${chapter.path}`);
    for (const id of [...chapter.source_sections.map((h) => h.anchor), ...chapter.source_explicit_aliases]) {
      assert.ok(ids.includes(id), `Missing canonical fragment: ${id}`);
    }
    assert.ok(!/^\s*\$\s*$/m.test(withoutCode(english)), `Broken display delimiter: ${chapter.path}`);
    assert.ok(!/@@(?:CODE|MATH|FIG|EXCERPT)/.test(english), `Unexpanded editing placeholder: ${chapter.path}`);
    totals.figures += chapter.figures.length;
    totals.displayMath += maths(english).length;
    totals.codeFences += code(english).length;
    totals.originalExcerpts += checkOriginalExcerpts(canonical, english);
    totals.canonicalSections += chapter.source_sections.length;
    totals.aliases += aliases.length;
  }
  return { report, totals, revisions };
}

if (process.argv[1] && resolve(process.argv[1]) === import.meta.filename) {
  console.log(JSON.stringify(inspectEnInferenceParity(resolve(import.meta.dirname, '../..')).totals, null, 2));
}
