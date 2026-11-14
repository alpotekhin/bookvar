import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';

const root=resolve(import.meta.dirname,'../..');
const manifest=JSON.parse(readFileSync(resolve(root,'publishing/tools/dali-data-sources.json'),'utf8'));
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const sources=[];
for(const row of manifest.sources){
  const local=readFileSync(resolve(root,row.path));
  if(hash(local)!==row.sha256)throw Error('Local source hash mismatch: '+row.path);
  if(process.argv.includes('--upstream')){
    const response=await fetch(row.url);
    if(!response.ok)throw Error('Source HTTP '+response.status+': '+row.url);
    const remote=Buffer.from(await response.arrayBuffer());
    if(!local.equals(remote))throw Error('Local source differs from pinned upstream: '+row.path);
  }
  sources.push(local.toString('utf8'));
}
const figure=readFileSync(resolve(root,manifest.figure));
if(hash(figure)!==manifest.figure_sha256||figure.toString('utf8')!==sources[1])
  throw Error('Published DALI diagram differs from the original');
if(/<(script|foreignObject)\b|(?:href|src)=["']https?:/i.test(sources[1]))
  throw Error('DALI original has an unexpected external rendering dependency');
const sourceCode=sources[0].split('Example::\n\n')[1].split('\n\n\nThe resulting graph')[0]
  .split('\n').map(line=>line.startsWith('    ')?line.slice(4):line).join('\n');
if(sourceCode!==manifest.source_code)throw Error('Source example changed');
const chapter=readFileSync(resolve(root,manifest.chapter),'utf8');
if(!chapter.includes('```python\n'+sourceCode+'\n```'))
  throw Error('Original example is not preserved in chapter');
if(!sources[2].includes('Apache License')||!sources[2].includes('Version 2.0, January 2004'))
  throw Error('Incomplete license');
console.log('PASS three pinned DALI source/license files, complete original SVG and unchanged example'
  +(process.argv.includes('--upstream')?' against upstream':'')+'; no DALI execution or timing claim');
