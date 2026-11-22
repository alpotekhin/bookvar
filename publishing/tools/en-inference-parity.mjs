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
const explicitIds = (text) => [...withoutCode(text).matchAll(/<(?:a|span) id="([^"]+)"/g)].map((m) => m[1]);
const sectionKeys = (rows) => rows.map(({ level, title }) => ({ level, title }));

// The Chen figure replacement renamed one heading in each edition. The
// historical receipt still describes the old headings; both old URLs must
// survive as explicit aliases where they used to be generated.
const speculativeRenames = {
  ru: {
    oldAnchor: 'полный-алгоритм-на-исходном-слайде',
    title: 'Алгоритм 2: черновик, проверка и замена',
    anchor: 'алгоритм-2-черновик-проверка-и-замена',
  },
  en: {
    oldAnchor: 'the-full-algorithm-in-the-original-slide',
    title: 'Algorithm 2: drafting, verification and correction',
    anchor: 'algorithm-2-drafting-verification-and-correction',
  },
};
const decodingRenames = {
  ru: [
    { oldAnchor: 'temperature-меняет-контраст-вероятностей', title: 'Температура', anchor: 'температура' },
    { oldAnchor: 'top-p-переменный-набор-заданной-массы', title: 'Top-p: порог суммарной вероятности', anchor: 'top-p-порог-суммарной-вероятности' },
    { oldAnchor: 'почему-порядок-преобразований-меняет-результат', title: 'Порядок фильтров', anchor: 'порядок-фильтров' },
    { oldAnchor: 'остановка-тоже-относится-к-декодированию', title: 'EOS, стоп-строки и лимит токенов', anchor: 'eos-стоп-строки-и-лимит-токенов' },
    { oldAnchor: 'как-выбирать-стратегию', title: 'Сравнение стратегий', anchor: 'сравнение-стратегий' },
    { oldAnchor: 'практика-и-первоисточники', title: 'Источники', anchor: 'источники' },
  ],
  en: [
    { oldAnchor: 'temperature-changes-probability-contrast', title: 'Temperature', anchor: 'temperature' },
    { oldAnchor: 'top-p-a-variable-set-with-a-mass-threshold', title: 'Top-p: a cumulative probability threshold', anchor: 'top-p-a-cumulative-probability-threshold' },
    { oldAnchor: 'transformation-order-changes-the-result', title: 'Filter order', anchor: 'filter-order' },
    { oldAnchor: 'stopping-is-part-of-decoding', title: 'EOS, stop strings, and token limits', anchor: 'eos-stop-strings-and-token-limits' },
    { oldAnchor: 'choosing-a-strategy', title: 'Comparing strategies', anchor: 'comparing-strategies' },
    { oldAnchor: 'practice-and-primary-sources', title: 'References', anchor: 'references' },
  ],
};
const currentBaseline = (rows, renames = []) => rows.map(row => {
  const rename = renames.find(candidate => row.anchor === candidate.oldAnchor);
  return rename ? { ...row, title: rename.title, anchor: rename.anchor } : row;
});
const preservedAnchors = (ids, renames = [], path) => renames.map(rename => {
  assert.equal(ids.filter(id => id === rename.oldAnchor).length, 1,
    `Missing or duplicate legacy fragment: ${path}#${rename.oldAnchor}`);
  return rename.oldAnchor;
});
const addedLevels = (current, baseline, path) => {
  let next = 0;
  const added = [];
  for (const section of current) {
    const previous = baseline[next];
    if (previous && previous.title === section.title && previous.level === section.level) next += 1;
    else added.push(section.level);
  }
  assert.equal(next, baseline.length,
    `A reviewed section was renamed, removed or reordered: ${path} (${baseline[next]?.title})`);
  return added;
};
const urls = (text) => [...new Set([...text.matchAll(/https?:\/\/[^\s<>\])]+/g)].map((m) => m[0]))];
const wikiTargets = (text) => [...new Set([...text.matchAll(/(?<!!)\[\[([^\]|]+)(?:\|[^\]]+)?\]\]/g)]
  .map((m) => normalizePath(m[1])))];
const maths = (text) => [...withoutCode(text).matchAll(/\$\$([\s\S]*?)\$\$/g)].map((m) => m[1]
  .replaceAll('бит/вес', 'bits/weight').replaceAll('на заданной сетке', 'on the chosen grid')
  .replaceAll('мс', 'ms').replaceAll('{,}', '.').replace(/\s+/g, '')
  // A full stop at the end of a display is sentence punctuation.
  .replace(/\.$/, ''));
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
    // The receipt describes a historical revision. Reverse the exact journals
    // for that revision, then check that the published editions still agree.
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
    assert.deepEqual(headings(reviewedCanonical), sectionKeys(chapter.source_sections));
    assert.deepEqual(headings(reviewedEnglish), sectionKeys(chapter.english_sections));
    const renames = chapter.path.endsWith('/58 Speculative decoding.md')
      ? { ru: [speculativeRenames.ru], en: [speculativeRenames.en] }
      : chapter.path.endsWith('/54 Decoding and next-token selection.md') ? decodingRenames : {};
    const canonicalBaseline = currentBaseline(chapter.source_sections, renames.ru);
    const englishBaseline = currentBaseline(chapter.english_sections, renames.en);
    assert.deepEqual(addedLevels(headings(english), englishBaseline, chapter.path),
      addedLevels(headings(canonical), canonicalBaseline, chapter.source), `New section hierarchy: ${chapter.path}`);
    assert.deepEqual(maths(english), maths(canonical), `Display mathematics: ${chapter.path}`);
    assert.deepEqual(code(english).map(normalizeCode), code(canonical).map(normalizeCode),
      `Protected code: ${chapter.path}`);
    assert.deepEqual(figures(english), figures(canonical), `Source figures: ${chapter.path}`);
    for (const figure of chapter.figures) assert.equal(sha(readFileSync(resolve(root, figure.path))), figure.sha256);
    for (const url of urls(canonical)) assert.ok(urls(english).includes(url), `Missing source URL: ${url}`);
    for (const target of wikiTargets(canonical)) assert.ok(wikiTargets(english).includes(target), `Missing wiki target: ${target}`);
    assert.deepEqual(explicitIds(reviewedEnglish), chapter.retained_aliases);
    assert.deepEqual(explicitIds(reviewedCanonical), chapter.source_explicit_aliases);
    const aliases = explicitIds(english);
    const canonicalAliases = explicitIds(canonical);
    assert.deepEqual(aliases.filter(id => chapter.retained_aliases.includes(id)), chapter.retained_aliases);
    assert.deepEqual(canonicalAliases.filter(id => chapter.source_explicit_aliases.includes(id)), chapter.source_explicit_aliases);
    const englishPreserved = [...chapter.retained_aliases, ...preservedAnchors(aliases, renames.en, chapter.path)];
    const canonicalPreserved = [...chapter.source_explicit_aliases, ...preservedAnchors(canonicalAliases, renames.ru, chapter.source)];
    assert.deepEqual(aliases.filter(id => !englishPreserved.includes(id)),
      canonicalAliases.filter(id => !canonicalPreserved.includes(id)), `New shared anchors: ${chapter.path}`);
    const ids = [...aliases, ...englishBaseline.map((h) => h.anchor)];
    assert.equal(new Set(ids).size, ids.length, `Duplicate IDs: ${chapter.path}`);
    for (const id of [...chapter.source_sections.map((h) => h.anchor), ...chapter.source_explicit_aliases]) {
      assert.ok(ids.includes(id), `Missing canonical fragment: ${id}`);
    }
    assert.ok(!/^\s*\$\s*$/m.test(withoutCode(english)), `Broken display delimiter: ${chapter.path}`);
    assert.ok(!/@@(?:CODE|MATH|FIG|EXCERPT)/.test(english), `Unexpanded editing placeholder: ${chapter.path}`);
    totals.figures += figures(english).length;
    totals.displayMath += maths(english).length;
    totals.codeFences += code(english).length;
    totals.originalExcerpts += checkOriginalExcerpts(canonical, english);
    totals.canonicalSections += headings(canonical).length;
    totals.aliases += aliases.length;
  }
  return { report, totals, revisions };
}

if (process.argv[1] && resolve(process.argv[1]) === import.meta.filename) {
  console.log(JSON.stringify(inspectEnInferenceParity(resolve(import.meta.dirname, '../..')).totals, null, 2));
}
