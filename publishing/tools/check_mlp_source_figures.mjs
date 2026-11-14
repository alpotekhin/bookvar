import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const rows=JSON.parse(readFileSync(resolve(root,'00 Учебник/Assets/Figures/curated/mlp-2026-11-03/assets.json'),'utf8'));
const hash=b=>createHash('sha256').update(b).digest('hex');
const fetch=url=>execFileSync('curl',['--fail','--silent','--show-error','--location',url],{maxBuffer:10_000_000});
for(const row of rows){
  const actual=readFileSync(resolve(root,row.asset)),upstream=fetch(row.source_image_url);
  if(!actual.equals(upstream)||hash(actual)!==row.sha256)throw Error('Original differs: '+row.asset);
  console.log(JSON.stringify({asset:row.asset,sha256:hash(actual),bytes:actual.length}));
}
const source=readFileSync(resolve(root,rows[0].source_page_snapshot));
if(!source.equals(fetch('https://raw.githubusercontent.com/cs231n/cs231n.github.io/8ec57ccfc32379ae4b37c45de949af359673f92f/neural-networks-1.md')))
  throw Error('Saved CS231n chapter differs from pinned source');
const license=readFileSync(resolve(root,'site/public/licenses/cs231n-MIT.txt'));
if(!license.equals(fetch('https://raw.githubusercontent.com/cs231n/cs231n.github.io/8ec57ccfc32379ae4b37c45de949af359673f92f/LICENSE')))
  throw Error('Public MIT notice differs from pinned repository license');
const notebook=JSON.parse(readFileSync(resolve(root,'05 Источники/Originals/neuron-mlp-2026-10-21/micrograd-demo.ipynb'),'utf8'));
const data=notebook.cells[8].outputs.find(o=>o.data?.['image/png']).data['image/png'];
const original=Buffer.from(Array.isArray(data)?data.join(''):data,'base64');
const image=readFileSync(resolve(root,'00 Учебник/Assets/Figures/curated/micrograd/moon-mlp.png'));
if(!image.equals(original))throw Error('micrograd image differs from saved output');
console.log(JSON.stringify({licenseSha256:hash(license),sourceSha256:hash(source),microgradSha256:hash(image),originals:4}));
