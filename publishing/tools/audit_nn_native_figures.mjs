/** Render the complete existing Harvard native images for inspection only.
 * Never writes source assets or publication assets. PNG previews stay in --out.
 * Usage: node .../audit_nn_native_figures.mjs --out /var/tmp/... --sharp /path/to/sharp/dist/index.cjs
 */
import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const option = (name) => process.argv[process.argv.indexOf(name) + 1];
if (!process.argv.includes('--out') || !process.argv.includes('--sharp'))
  throw Error('Supply explicit --out and existing --sharp module paths');
const out = resolve(option('--out'));
if (!out.startsWith('/var/tmp/')) throw Error('QA output must remain outside the repository in /var/tmp');
const sharp = createRequire(import.meta.url)(option('--sharp'));
const page = '05 Источники/Courses/Harvard ML Systems/vol1/nn_architectures.md';
const source = await readFile(join(root, page), 'utf8');
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
if (hash(source) !== '52c77d7f6d386e97af65decf4f55523be785930571759fe5a906069ad13fcfac')
  throw Error('The previously read source revision changed; review before rendering');
const images = Array.from(source.matchAll(/!\[\[(Assets\/Sources\/Harvard ML Systems\/vol1\/nn_architectures\/[^\]|]+)(?:\|[^\]]*)?\]\]/g));
if (images.length !== 24 || new Set(images.map(m => m[1])).size !== 24)
  throw Error('Expected exactly the 24 distinct referenced source objects');
await mkdir(out, { recursive: true });
const plain = (text) => text.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim();
const rows = [];
for (const [i, match] of images.entries()) {
  const path = match[1];
  const bytes = await readFile(join(root, path));
  const metadata = await sharp(bytes).metadata();
  const preceding = source.slice(0, match.index);
  const headings = Array.from(preceding.matchAll(/^#{1,5} (.+)$/gm));
  const following = source.slice(match.index + match[0].length).trimStart();
  const paragraph = following.split(/\n\s*\n/, 1)[0];
  const caption = /^(?:\*|Figure\s)/.test(paragraph) ? plain(paragraph) : null;
  const destination = join(out, `${String(i + 1).padStart(2, '0')}-${basename(path)}.png`);
  const rendered = await sharp(bytes, { density: 180 })
    .resize({ width: 1600 }).flatten({ background: '#fff' }).png().toFile(destination);
  rows.push({ number: i + 1, asset: path, sha256: hash(bytes),
    source_line: preceding.split('\n').length,
    section: headings.at(-1)?.[1], caption,
    original_width: metadata.width, original_height: metadata.height,
    view_box: path.endsWith('.svg') ? bytes.toString().match(/viewBox="([^"]+)"/)?.[1] : null,
    preview: destination, preview_width: rendered.width, preview_height: rendered.height });
  console.log(`${String(i + 1).padStart(2, '0')} ${basename(path)} | ${rows.at(-1).section}\n${caption || '(no standalone caption)'}\n`);
}
await writeFile(join(out, 'inventory.json'), JSON.stringify({ source: page,
  source_sha256: hash(source), figures: rows }, null, 2) + '\n');
console.log(`Rendered ${rows.length} complete QA previews; source/publication files unchanged. ${join(out, 'inventory.json')}`);
