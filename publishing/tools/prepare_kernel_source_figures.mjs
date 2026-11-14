import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const folder='00 Учебник/Assets/Figures/curated/kernels-2026-11-03/';
const repository='https://raw.githubusercontent.com/scikit-learn/scikit-learn.github.io/199d116ab37beed506c63fc9f6a74f56e228de43/1.9/_images/';
const cases=[
  ['sklearn-rbf-decision-grid.png',repository+'sphx_glr_plot_rbf_parameters_001.png'],
  ['sklearn-rbf-validation-grid.png',repository+'sphx_glr_plot_rbf_parameters_002.png'],
];
const sourceUrl='https://raw.githubusercontent.com/scikit-learn/scikit-learn/866c0f51e7560ef0303cbcc5f159df5382ea9e3f/examples/svm/plot_rbf_parameters.py';
const check=process.argv.includes('--check');
const hash=b=>createHash('sha256').update(b).digest('hex');
const fetch=url=>execFileSync('curl',['--fail','--silent','--show-error','--location',url],{maxBuffer:10_000_000});
function preserve(path,bytes){
  const target=resolve(root,path);
  if(check||existsSync(target)){
    if(!readFileSync(target).equals(bytes))throw new Error('Original bytes differ: '+path);
  }else{
    mkdirSync(dirname(target),{recursive:true});
    writeFileSync(target,bytes);
  }
}
const source=fetch(sourceUrl);
const sourcePath='05 Источники/Originals/kernels-2026-11-03/sklearn-rbf-parameters-1.9.1.py';
preserve(sourcePath,source);
console.log(JSON.stringify({sourcePath,sourceUrl,sha256:hash(source)}));
for(const [name,url]of cases){
  const bytes=fetch(url);
  if(!bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))throw new Error('Not a PNG');
  preserve(folder+name,bytes);
  console.log(JSON.stringify({name,url,sha256:hash(bytes),bytes:bytes.length,
    dimensions:[bytes.readUInt32BE(16),bytes.readUInt32BE(20)]}));
}
