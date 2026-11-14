import {createHash} from 'node:crypto';
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const meta=JSON.parse(readFileSync(resolve(root,'publishing/tools/gpu-copy-source-figures.json'),'utf8'));
const check=process.argv.includes('--check'),sha=b=>createHash('sha256').update(b).digest('hex');
function preserve(path,bytes){
  const p=resolve(root,path);
  if(check||existsSync(p)){
    if(!readFileSync(p).equals(bytes))throw Error('Source bytes differ: '+path);
  }else{mkdirSync(dirname(p),{recursive:true});writeFileSync(p,bytes);}
}
for(const file of meta.files){
  const p=resolve(root,meta.archive,file.name);
  const sourcePath=file.name==='LICENSE'?'LICENSE':
    file.name.endsWith('.py')?'intermediate_source/'+file.name:'_static/img/pinmem/'+file.name;
  const url='https://raw.githubusercontent.com/pytorch/tutorials/'+meta.pin+'/'+sourcePath;
  // Public checkouts can verify the same pinned bytes without private archives.
  const response=existsSync(p)?null:await fetch(url);
  if(response&&!response.ok)throw Error('Source fetch failed: '+url+' '+response.status);
  const bytes=existsSync(p)?readFileSync(p):Buffer.from(await response.arrayBuffer());
  if(bytes.length!==file.bytes||sha(bytes)!==file.sha256)throw Error('Pinned source changed: '+url);
  if(file.dimensions){
    if(!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))throw Error('Not PNG');
    if(JSON.stringify([bytes.readUInt32BE(16),bytes.readUInt32BE(20)])!==JSON.stringify(file.dimensions))
      throw Error('PNG dimensions changed: '+file.name);
    preserve(meta.folder+file.name,bytes);
  }else if(file.name==='LICENSE'){
    preserve('site/public/licenses/pytorch-tutorials-5672cd98-BSD-3-Clause.txt',bytes);
  }
  console.log(JSON.stringify({file:file.name,bytes:bytes.length,sha256:file.sha256,dimensions:file.dimensions}));
}
