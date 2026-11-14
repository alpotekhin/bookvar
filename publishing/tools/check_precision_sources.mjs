import {readFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';

const root=resolve(import.meta.dirname,'../..');
const manifest=JSON.parse(readFileSync(resolve(root,'publishing/tools/precision-accumulation-sources.json'),'utf8'));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const sources=[];
for(const row of manifest.sources){
  let bytes;
  if(process.argv.includes('--upstream')||!existsSync(resolve(root,row.path))){
    const response=await fetch(row.url);
    if(!response.ok)throw Error('Source HTTP '+response.status+': '+row.url);
    bytes=Buffer.from(await response.arrayBuffer());
  }else bytes=readFileSync(resolve(root,row.path));
  if(hash(bytes)!==row.sha256)throw Error('Source hash mismatch: '+row.path);
  if(existsSync(resolve(root,row.path))&&!readFileSync(resolve(root,row.path)).equals(bytes))
    throw Error('Local source differs from pinned upstream: '+row.path);
  sources.push(bytes.toString('utf8'));
}
const chapter=readFileSync(resolve(root,manifest.chapter),'utf8');
if(!sources[0].replace(/\s+/g,' ').includes(manifest.quote))throw Error('Quotation source mismatch');
if(!chapter.includes('> '+manifest.quote))throw Error('Quotation changed in chapter');
const notice='  <!-- Bookvar: unchanged upstream stylesheet embedded for standalone rendering; no geometry or labels changed. -->\n';
const bundle=notice+'  <style><![CDATA[\n'+sources[4]+']]></style>\n';
const figure=readFileSync(resolve(root,manifest.figure));
if(hash(figure)!==manifest.figure_sha256
  ||figure.toString('utf8')!==sources[3].replace('  <defs>\n',bundle+'  <defs>\n'))
  throw Error('Figure changed beyond the exact stylesheet bundle');
if(!readFileSync(resolve(root,'site/public/licenses/pytorch-v2.14.1-LICENSE.txt'),'utf8')
  .includes('Redistribution and use in source and binary forms'))throw Error('PyTorch notice missing');
console.log('PASS six pinned source/license files, exact quotation and complete source-SVG/style bundle'
  +(process.argv.includes('--upstream')?' against upstream':''));
