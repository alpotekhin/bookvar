import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import matter from 'gray-matter';
import { loadManifest } from '../adapter/manifest.js';
import { parseMarkdownHeadings, normalizeObsidianHeading } from '../adapter/build.js';
import { wikiHeadingSlug } from '../adapter/links.js';

const root = resolve(import.meta.dirname, '../..');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const pages = loadManifest(resolve(root, 'publishing/navigation.yml')).sections.flatMap(section => section.pages);
const published = new Map(pages.map(page => [page.source, page]));
const maps = process.argv.slice(2);
if (!maps.length) throw new Error('Provide reviewed consolidation JSON maps.');
const applied: object[] = [];
for (const path of maps) {
  const audit = JSON.parse(readFileSync(resolve(root, path), 'utf8'));
  for (const entry of audit.entries) {
    if (entry.decision !== 'alias') continue;
    const source = published.get(entry.source);
    const target = published.get(entry.canonicalSource);
    // Unpublished legacy notes remain untouched until their own migration.
    if (!source) continue;
    if (!target) throw new Error('Unpublished canonical target: ' + entry.canonicalSource);
    const absolute = resolve(root, entry.source);
    const before = readFileSync(absolute, 'utf8');
    const parsed = matter(before);
    const canonical = readFileSync(resolve(root, target.source), 'utf8');
    if (matter(canonical).data.status === 'redirect') throw new Error('Alias chain: ' + entry.source);
    if (entry.canonicalHeading) {
      const matches = parseMarkdownHeadings(matter(canonical).content, true).filter(heading =>
        heading.explicitAnchor === entry.canonicalHeading
        || heading.normalized === normalizeObsidianHeading(entry.canonicalHeading));
      const fragments = new Set(matches.map(heading => heading.explicitAnchor ?? wikiHeadingSlug(heading.text)));
      if (fragments.size !== 1) throw new Error('Missing/ambiguous heading: ' + entry.source + ': ' + entry.canonicalHeading);
    }
    if (parsed.data.status === 'redirect' && parsed.data.canonical_topic === target.source) continue;
    if (entry.sourceSha256 && hash(before) !== entry.sourceSha256)
      throw new Error('Source changed after review: ' + entry.source);
    if (entry.canonicalSha256 && hash(canonical) !== entry.canonicalSha256)
      throw new Error('Canonical changed after review: ' + target.source);
    const snapshot = 'docs/audit/snapshots/2026-10-29-reference-before-consolidation/' + source.source;
    const archive = resolve(root, snapshot);
    if (existsSync(archive) && readFileSync(archive, 'utf8') !== before)
      throw new Error('Refusing to overwrite an existing reference snapshot: ' + snapshot);
    mkdirSync(dirname(archive), { recursive: true });
    writeFileSync(archive, before);
    const title = matter(canonical).data.title;
    const link = target.source.replace(/\.md$/, '') + (entry.canonicalHeading ? '#' + entry.canonicalHeading : '');
    const anchors = [...new Set(parseMarkdownHeadings(parsed.content, true)
      .map(heading => heading.explicitAnchor ?? wikiHeadingSlug(heading.text)))];
    const data = {
      ...parsed.data,
      status: 'redirect',
      last_updated: '2026-10-29',
      canonical_topic: target.source,
      ...(entry.canonicalHeading ? { canonical_heading: entry.canonicalHeading } : {}),
    };
    const body = '# ' + parsed.data.title + '\n\n'
      + 'Полное объяснение: [[' + link + '|' + title + ']].\n\n'
      + '<!-- Compatibility anchors for existing Obsidian and web links. -->\n'
      + anchors.map(anchor => '<span id="' + anchor + '" aria-hidden="true"></span>').join('\n') + '\n';
    const after = matter.stringify(body, data);
    writeFileSync(absolute, after);
    applied.push({
      source: source.source, route: source.route, canonicalSource: target.source,
      canonicalHeading: entry.canonicalHeading, review: path, snapshot,
      beforeSha256: hash(before), afterSha256: hash(after), canonicalSha256: hash(canonical),
      retainedFragmentAnchors: anchors,
    });
  }
}
const journalPath = resolve(root, 'docs/audit/2026-10-29-topic-aliases.json');
const existing = existsSync(journalPath) ? JSON.parse(readFileSync(journalPath, 'utf8')).entries : [];
writeFileSync(journalPath, JSON.stringify({ date: '2026-10-29', entries: [...existing, ...applied] }, null, 2) + '\n');
console.log(JSON.stringify({ applied: applied.length, total: existing.length + applied.length }));
