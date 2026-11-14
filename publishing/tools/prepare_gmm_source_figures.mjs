import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const folder = '00 Учебник/Assets/Figures/curated/gmm-2026-11-03/';
const notebookPath = '05 Источники/Originals/gmm-2026-10-25/jake-gmm.ipynb';
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const notebookBytes = readFileSync(resolve(root, notebookPath));
if (digest(notebookBytes) !== '9438d483e156738ba8d581e54f150590061ff6fbe6147cb38ab56ae0449c07e6')
  throw new Error('Original notebook changed');
const notebook = JSON.parse(notebookBytes);
const native = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/199d116ab37beed506c63fc9f6a74f56e228de43/1.9/_images/sphx_glr_plot_gmm_init_001.png';
const sourceCode = 'https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/examples/mixture/plot_gmm_init.py';
const cases = [
  ['jake-moons-generated.png', 34],
  ['jake-training-digits.png', 42],
  ['jake-digit-aic.png', 46],
  ['jake-generated-digits.png', 52],
  ['sklearn-gmm-initialization.png', native],
];
const check = process.argv.includes('--check');
function preserve(path, bytes) {
  const absolute = resolve(root, path);
  if (check || existsSync(absolute)) {
    if (!readFileSync(absolute).equals(bytes)) throw new Error('Original bytes differ: ' + path);
  } else {
    mkdirSync(dirname(absolute), {recursive:true});
    writeFileSync(absolute, bytes);
  }
}
const sourceBytes = execFileSync('curl', ['--fail','--silent','--show-error','--location',sourceCode]);
const sourcePath = '05 Источники/Originals/gmm-2026-11-03/sklearn-gmm-init-1.9.1.py';
preserve(sourcePath, sourceBytes);
console.log(JSON.stringify({sourcePath,sourceCode,sha256:digest(sourceBytes)}));
for (const [name, origin] of cases) {
  let bytes;
  if (typeof origin === 'number') {
    const data = notebook.cells[origin].outputs[0].data['image/png'];
    bytes = Buffer.from(Array.isArray(data) ? data.join('') : data, 'base64');
  } else bytes = execFileSync('curl', ['--fail','--silent','--show-error','--location',origin],{maxBuffer:10_000_000});
  if (!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) throw new Error('Not a PNG');
  preserve(folder+name,bytes);
  console.log(JSON.stringify({name,origin,sha256:digest(bytes),bytes:bytes.length,
    dimensions:[bytes.readUInt32BE(16),bytes.readUInt32BE(20)]}));
}
