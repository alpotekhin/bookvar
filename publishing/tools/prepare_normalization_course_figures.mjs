import {createHash} from 'node:crypto';
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const archive='05 Источники/Originals/normalization-course-2026-11-03/';
const folder='00 Учебник/Assets/Figures/curated/normalization-course-2026-11-03/';
const hash=b=>createHash('sha256').update(b).digest('hex');
const notebookBytes=readFileSync(resolve(root,archive,'makemore_part3_bn.ipynb'));
if(hash(notebookBytes)!=='464c959353b5e929bf0f922c0e7547823fb514fa2f672f30a065a9d403c641e4')throw Error('Notebook revision changed');
const notebook=JSON.parse(notebookBytes),check=process.argv.includes('--check');
function preserve(path,bytes){
  const target=resolve(root,path);
  if(check||existsSync(target)){
    if(!readFileSync(target).equals(bytes))throw Error('Source bytes differ: '+path);
  }else{
    mkdirSync(dirname(target),{recursive:true});writeFileSync(target,bytes);
  }
}
for(const [cell,name]of [[15,'karpathy-activation-distribution.png'],[16,'karpathy-gradient-distribution.png']]){
  const outputs=notebook.cells[cell].outputs;
  const figures=outputs.filter(o=>o.data?.['image/png']);
  if(figures.length!==1)throw Error('Ambiguous source output: '+cell);
  const encoded=figures[0].data['image/png'];
  const bytes=Buffer.from(Array.isArray(encoded)?encoded.join(''):encoded,'base64');
  if(!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))throw Error('Not a PNG');
  preserve(folder+name,bytes);
  console.log(JSON.stringify({name,cell,output:outputs.indexOf(figures[0]),sha256:hash(bytes),
    dimensions_px:[bytes.readUInt32BE(16),bytes.readUInt32BE(20)],bytes:bytes.length}));
}
const license=readFileSync(resolve(root,archive,'karpathy-LICENSE.txt'));
if(hash(license)!=='a59ec5cdb1c1e447e5266a014b3e7ded511f8d6e5a08931e931c87490ff821fe')throw Error('License changed');
preserve('site/public/licenses/karpathy-nn-zero-to-hero-73c3fcc7-MIT.txt',license);
