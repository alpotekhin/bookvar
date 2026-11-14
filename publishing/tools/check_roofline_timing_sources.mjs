import {createHash} from 'node:crypto';
import {existsSync,readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=resolve(fileURLToPath(new URL('../..',import.meta.url)));
const meta=JSON.parse(readFileSync(resolve(root,'publishing/tools/roofline-timing-sources.json'),'utf8'));
const texts={};
for(const file of meta.files){
  const path=resolve(root,file.path);
  const response=existsSync(path)?null:await fetch(file.url);
  if(response&&!response.ok)throw Error('Pinned source fetch failed: '+response.status);
  const bytes=existsSync(path)?readFileSync(path):Buffer.from(await response.arrayBuffer());
  if(bytes.length!==file.bytes||createHash('sha256').update(bytes).digest('hex')!==file.sha256)
    throw Error('Pinned source bytes differ: '+file.name);
  texts[file.name]=bytes.toString('utf8');
  console.log(file.name+' '+file.sha256);
}
const start='# While ``timeit.Timer.autorange``';
const raw=start+texts['benchmark.py'].split(start)[1].split('\n#\n\nm0 =')[0];
const original=raw.split('\n').map(s=>s.replace(/^# ?/,'')).join('\n').replaceAll('``','`');
const body=readFileSync(resolve(root,'00 Учебник/10 ML Systems/03 Измерение производительности и roofline.md'),'utf8');
const quote=body.split('<!-- timing-source: blocked-autorange:start -->')[1]
  .split('<!-- timing-source: blocked-autorange:end -->')[0]
  .trim().split('\n').map(s=>s.replace(/^> ?/,'')).join('\n');
if(quote!==original)throw Error('Complete original tutorial paragraph changed');
const tutorialLicense=readFileSync(resolve(root,meta.licensePaths[0]));
if(createHash('sha256').update(tutorialLicense).digest('hex')!==
  '5703fcbe2d5dfcc48d486d26e4b9d2e48255c24602e1bd0270b60b8978116f86')
  throw Error('Tutorial BSD notice differs');
if(!readFileSync(resolve(root,meta.licensePaths[1]),'utf8').includes('Redistribution and use'))
  throw Error('PyTorch full source license missing');
console.log('PASS exact complete tutorial paragraph,three pinned source files and published license notices');

