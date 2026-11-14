import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Copy saved original rasters only. Never execute plotting code or resize images.
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const folder = '00 Учебник/Assets/Figures/curated/dimensionality-2026-11-03/';
const source = '05 Источники/Originals/pca-tsne-umap-2026-10-25/jake-pca.ipynb';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const notebookBytes = readFileSync(resolve(root, source));
if (digest(notebookBytes) !== 'fa00993593486ca2d2445bf4781eba2ea79bb07f3360fa9456d67f573b9da9f2')
  throw new Error('Source notebook revision changed');
const notebook = JSON.parse(notebookBytes);
const jake = 'https://raw.githubusercontent.com/jakevdp/PythonDataScienceHandbook/d66231454ef753818dc9213c9b5942e067266966/';
const umap = 'https://raw.githubusercontent.com/lmcinnes/umap/e82ed0d457b566b043ef44f4007a7149b0daca74/';
const distill = 'https://raw.githubusercontent.com/distillpub/post--misread-tsne/f2492b469bec8aaa7aa83cc828081e14a295414d/';
const cases = [
  ['jake-pixel-basis.png', jake + 'notebooks/figures/05.09-digits-pixel-components.png'],
  ['jake-pca-basis.png', jake + 'notebooks/figures/05.09-digits-pca-components.png'],
  ['jake-cumulative-variance.png', 34],
  ['jake-noisy-digits.png', 40],
  ['jake-filtered-digits.png', 44],
  ['umap-neighbors-2.png', umap + 'doc/images/parameters_13_1.png'],
  ['umap-neighbors-200.png', umap + 'doc/images/parameters_13_7.png'],
  ['umap-min-dist-0.png', umap + 'doc/images/parameters_16_1.png'],
  ['umap-min-dist-099.png', umap + 'doc/images/parameters_16_6.png'],
  ['distill-global-150-input.png', distill + 'public/assets/figure_04_original.png'],
  ['distill-global-150-p50.png', distill + 'public/assets/figure_04_04.png'],
  ['distill-global-600-input.png', distill + 'public/assets/figure_05_original.png'],
  ['distill-global-600-p50.png', distill + 'public/assets/figure_05_04.png'],
  ['distill-noise-input.png', distill + 'public/assets/figure_06_original.png'],
  ['distill-noise-p2.png', distill + 'public/assets/figure_06_01.png'],
];
const check = process.argv.includes('--check');
if (!check) mkdirSync(resolve(root, folder), {recursive:true});
for (const [name, origin] of cases) {
  let bytes;
  if (typeof origin === 'number') {
    const saved = notebook.cells[origin].outputs[0].data['image/png'];
    bytes = Buffer.from(Array.isArray(saved) ? saved.join('') : saved, 'base64');
  } else {
    bytes = execFileSync('curl', ['--fail', '--silent', '--show-error', '--location', origin],
      {maxBuffer:10_000_000});
  }
  if (!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))
    throw new Error('Not a PNG: ' + name);
  const path = resolve(root, folder, name);
  if (check || existsSync(path)) {
    if (!readFileSync(path).equals(bytes)) throw new Error('Original bytes differ: ' + name);
  } else writeFileSync(path, bytes);
  console.log(JSON.stringify({
    name, source:origin, sha256:digest(bytes), bytes:bytes.length,
    dimensions:[bytes.readUInt32BE(16), bytes.readUInt32BE(20)],
  }));
}

