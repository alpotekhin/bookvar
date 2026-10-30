/** Compare the chapter's retained figures with pinned files or exact PDF crops. */
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import assert from 'node:assert/strict';
import YAML from 'yaml';

const root = resolve(import.meta.dirname, '../..');
const chapter = '00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md';
const text = readFileSync(resolve(root, chapter), 'utf8');
const paths = [...text.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m => m[1]);
const rows = YAML.parse(readFileSync(resolve(root, '05 Источники/asset-registry.yml'), 'utf8'), { merge: true }).assets;
assert.equal(paths.length, 15);
const sha = (bytes: Buffer) => createHash('sha256').update(bytes).digest('hex');
const tmp = mkdtempSync(join(tmpdir(), 'bookvar-gpu-figures-'));
let remoteCount = 0, cropCount = 0;

try {
  for (const [index, path] of paths.entries()) {
    const entry = rows.find((r: { asset: string }) => r.asset === path);
    assert.ok(entry, path);
    assert.ok(entry.used_in.includes(chapter), path);
    const local = readFileSync(resolve(root, path));
    assert.equal(sha(local), entry.sha256, path);
    if (entry.derivation === 'pdf-page-render-crop') {
      const pdf = resolve(root, entry.source_asset);
      assert.equal(sha(readFileSync(pdf)), entry.source_asset_sha256);
      const [x, y, right, bottom] = entry.crop_box_px;
      const [width, height] = entry.render_page_size_px;
      const prefix = join(tmp, String(index));
      execFileSync('pdftoppm', ['-f', String(entry.source_page), '-l', String(entry.source_page),
        '-singlefile', '-scale-to-x', String(width), '-scale-to-y', String(height),
        '-x', String(x), '-y', String(y), '-W', String(right - x), '-H', String(bottom - y),
        '-png', pdf, prefix], { stdio: 'pipe' });
      execFileSync(process.env.BOOKVAR_PYTHON ?? 'python3', ['-c',
        'from PIL import Image; import sys; a=Image.open(sys.argv[1]).convert("RGB"); b=Image.open(sys.argv[2]).convert("RGB"); assert a.size==b.size and a.tobytes()==b.tobytes(), sys.argv[1]',
        resolve(root, path), prefix + '.png'], { stdio: 'pipe' });
      cropCount++;
      console.log('PIXELS', path);
    } else {
      const url = entry.source_image_url?.includes(entry.commit)
        ? entry.source_image_url
        : entry.source_url.replace('https://github.com/', 'https://raw.githubusercontent.com/').replace('/blob/', '/');
      assert.ok(entry.commit && url.includes(entry.commit), 'Unpinned figure: ' + path);
      const source = execFileSync('curl', ['--fail', '--silent', '--show-error', '--location',
        '--max-time', '30', url], { maxBuffer: 16 * 1024 * 1024 });
      assert.ok(local.equals(source), 'Upstream bytes differ: ' + path);
      remoteCount++;
      console.log('BYTES', path);
    }
  }
  assert.equal(cropCount, 6);
  assert.equal(remoteCount, 9);
  console.log('PASS 15 source figures: 9 pinned byte matches and 6 pixel-identical PDF crops');
} finally {
  // Only this invocation's temporary render directory, never source assets.
  rmSync(tmp, { recursive: true, force: true });
}
