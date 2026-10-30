import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import matter from 'gray-matter';
import { loadManifest } from '../adapter/manifest.js';
import { wikiHeadingSlug } from '../adapter/links.js';
import { beforeTopicConsolidation } from './topic-consolidation-baseline.mjs';

const root = resolve(import.meta.dirname, '../..');
const read = (path: string) => readFileSync(resolve(root, path), 'utf8');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const json = (path: string) => JSON.parse(read(path));
const snapshotRoot = 'docs/audit/snapshots/2026-10-29-before-topic-consolidation';
function update(path: string, value: unknown) {
  const snapshot = resolve(root, snapshotRoot, path.replace(/^docs\/audit\//, ''));
  if (!existsSync(snapshot)) {
    mkdirSync(dirname(snapshot), { recursive: true });
    writeFileSync(snapshot, read(path));
  }
  writeFileSync(resolve(root, path), JSON.stringify(value, null, 2) + '\n');
}
const stripCode = (text: string) => text.replace(/^(```|~~~)[^\n]*\n[\s\S]*?^\1\s*$/gm, '');
const headings = (text: string) => [...stripCode(text).matchAll(/^(#{1,6}) (.+)$/gm)]
  .map(match => ({ level: match[1].length, title: match[2], anchor: wikiHeadingSlug(match[2]) }));
const aliases = (text: string) => [...stripCode(text).matchAll(/<(?:a|span) id="([^"]+)"/g)].map(match => match[1]);
const figures = (text: string) => [...text.matchAll(/!\[\[([^\]|]+)/g)]
  .map(match => match[1].replace(/^02 Areas\/ML & DL\//, ''));

const enPath = 'docs/audit/2026-10-29-english-reconciliation.json';
const en = json(enPath);
for (const row of en.entries) {
  const canonical = read(row.canonical), english = read(row.path);
  const meta = matter(english).data;
  if (meta.canonical_revision !== hash(canonical))
    throw new Error('Unreconciled English metadata: ' + row.path);
  if (row.after_sha256 !== hash(english) || row.canonical_sha256 !== hash(canonical)) {
    // Exact deltas must restore the previously reviewed body, not merely yield
    // a plausible new SHA. Reader guide prose has its own full-read review.
    if (hash(beforeTopicConsolidation(english, row.path, root)) !== row.after_sha256)
      throw new Error('Unrecorded English change: ' + row.path);
    if (hash(beforeTopicConsolidation(canonical, row.canonical, root)) !== row.canonical_sha256)
      throw new Error('Unrecorded canonical change: ' + row.canonical);
    row.previous_consolidation_sha256 = row.after_sha256;
    row.previous_canonical_sha256 = row.canonical_sha256;
    row.after_sha256 = hash(english);
    row.canonical_sha256 = hash(canonical);
    row.after_lines = english.split('\n').length;
    row.figures = figures(english);
    row.figure_count = row.figures.length;
    row.consolidation_review = 'Exact recorded additions reconciled in both locales; historical full-body review retained with its limits.';
    row.render_review = 'docs/audit/2026-10-29-publication-consolidation-browser.json';
  }
}
en.inventory.figure_embeds_after = en.entries.reduce((sum: number, row: any) => sum + row.figure_count, 0);
en.consolidation_report = 'docs/audit/2026-10-29-publication-consolidation.md';
en.checks.render = 'Earlier render checks remain in the pre-consolidation snapshot. Final changed-revision browser evidence is recorded separately in publication-consolidation-browser.json.';
update(enPath, en);

const inferencePath = 'docs/audit/2026-10-29-en-inference-parity.json';
const inference = json(inferencePath);
for (const row of inference.chapters) {
  const canonical = read(row.source), english = read(row.path);
  if (hash(canonical) === row.source_sha256 && hash(english) === row.after_sha256) continue;
  if (hash(beforeTopicConsolidation(canonical, row.source, root)) !== row.source_sha256
    || hash(beforeTopicConsolidation(english, row.path, root)) !== row.after_sha256)
    throw new Error('Unrecorded inference parity change: ' + row.path);
  row.previous_source_sha256 = row.source_sha256;
  row.previous_english_sha256 = row.after_sha256;
  row.source_sha256 = hash(canonical);
  row.after_sha256 = hash(english);
  row.source_sections = headings(canonical);
  row.english_sections = headings(english);
  row.source_explicit_aliases = aliases(canonical);
  row.retained_aliases = aliases(english);
  row.source_lines = canonical.split('\n').length;
  row.source_read_lines = [1, row.source_lines];
  row.consolidation_review = 'Paired exact additions and source terminology reviewed; code/figures preserved, current CPU receipt required.';
}
update(inferencePath, inference);

const corpusPath = 'docs/audit/2026-10-29-corpus-revisions.json';
const corpus = json(corpusPath);
const book = loadManifest(resolve(root, 'publishing/navigation.yml')).sections.find(section => section.id === 'textbook')!;
for (const page of book.pages) {
  const text = read(page.source), current = hash(text);
  let row = corpus.pages.find((row: any) => row.source === page.source);
  if (!row) {
    row = { source: page.source, route: page.route, sha256: current,
      evidence: { kind: 'consolidated-full-chapter', reports: ['docs/audit/2026-10-29-publication-consolidation.md'] } };
    corpus.pages.push(row);
    continue;
  }
  if (row.sha256 === current) continue;
  if (page.route !== 'textbook/how-to-use' && hash(beforeTopicConsolidation(text, page.source, root)) !== row.sha256)
    throw new Error('Canonical change lacks reversible consolidation evidence: ' + page.source);
  row.pre_topic_consolidation_sha256 = row.sha256;
  row.sha256 = current;
  row.evidence.consolidation = {
    scope: page.route === 'textbook/how-to-use' ? 'Reader guide read in full and updated for the unified structure.'
      : 'Recorded consolidation deltas preserve the reviewed preceding revision; new explanations have separate source/example review.',
    report: 'docs/audit/2026-10-29-publication-consolidation.md',
  };
}
corpus.consolidation = {
  pages: book.pages.length, previousPages: 149,
  report: 'docs/audit/2026-10-29-publication-consolidation.md',
  limit: 'This is revision lineage, not a new whole-book primary-source or visual acceptance.',
};
update(corpusPath, corpus);
console.log(JSON.stringify({ english: en.entries.length, figures: en.inventory.figure_embeds_after, canonical: corpus.pages.length }));
