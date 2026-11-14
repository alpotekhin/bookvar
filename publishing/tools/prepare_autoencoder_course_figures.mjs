import {createHash} from 'node:crypto';
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const notebookPath='05 Источники/Originals/autoencoder-2026-10-26/uva-ae.ipynb';
const bytes=readFileSync(resolve(root,notebookPath));
const sha=b=>createHash('sha256').update(b).digest('hex');
if(sha(bytes)!=='ae9cb1d0c65d447744eadca5aeccec52597b9e9191692d7873344e215439a6be')
  throw Error('Pinned UvA notebook changed');
const notebook=JSON.parse(bytes);
const folder='00 Учебник/Assets/Figures/curated/autoencoder-course-2026-11-03/';
const cases=[
  ['uva-latent-width.svg',26,'472ec692b4afb5b51012b78d61ac8cd8ae827303b105fb7cd3a3172ad188bee3'],
  ['uva-noise-reconstruction.svg',32,'5fefe3cd0a33955b020f7343acd2d3da9dffa004df4f773f1c657d0ac089cc13'],
  ['uva-pattern-reconstruction.svg',34,'f0f5ef6a8decb6a29fd2975f70cd5d7918d0b483431486cd1ccba4140a06d63e'],
  ['uva-random-codes.svg',37,'94bd5c7d9898dbc7483f254e77a2b7d032e7efbe995a0cac7803aeefa1968722'],
];
for(const[name,cell,expected]of cases){
  const raw=notebook.cells[cell].outputs[0].data['image/svg+xml'];
  const data=Buffer.from(Array.isArray(raw)?raw.join(''):raw,'utf8');
  if(sha(data)!==expected)throw Error('Notebook output changed: '+cell);
  const source=data.toString();
  const viewBox=source.match(/viewBox="([^"]+)"/)?.[1];
  if(!viewBox||source.includes('<script')||/<image[^>]+(?:href|xlink:href)="https?:/i.test(source))
    throw Error('Incomplete or externally dependent SVG');
  const target=resolve(root,folder+name);
  if(process.argv.includes('--check')||existsSync(target)){
    if(!readFileSync(target).equals(data))throw Error('Published SVG differs: '+name);
  }else{
    mkdirSync(dirname(target),{recursive:true});writeFileSync(target,data);
  }
  console.log(JSON.stringify({name,cell,output:0,bytes:data.length,sha256:sha(data),viewBox}));
}

