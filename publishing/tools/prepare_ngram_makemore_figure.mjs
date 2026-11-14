import {createHash} from 'node:crypto';
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const source='05 Источники/Originals/ngram-2026-10-21/makemore-part1-bigrams.ipynb';
const target='00 Учебник/Assets/Figures/curated/ngram-makemore-2026-11-03/karpathy-bigram-counts.png';
const hash=b=>createHash('sha256').update(b).digest('hex');
const nbBytes=readFileSync(resolve(root,source));
if(hash(nbBytes)!=='db2f139409bad2966baac06581b6ca8ec9a03668df50dd073577fd98b3be0045')throw Error('Notebook changed');
const nb=JSON.parse(nbBytes),outputs=nb.cells[11].outputs;
if(!nb.cells[11].source.join('').includes("plt.imshow(N, cmap='Blues')"))throw Error('Wrong cell');
const matches=outputs.filter(o=>o.data?.['image/png']);
if(matches.length!==1)throw Error('Ambiguous saved figure');
const encoded=matches[0].data['image/png'];
const bytes=Buffer.from(Array.isArray(encoded)?encoded.join(''):encoded,'base64');
if(hash(bytes)!=='0972425d7afc53f7d81c7da5aca31a4dd99cb664bf102f604b0e3c61cec57d39')throw Error('Saved output changed');
if(bytes.readUInt32BE(16)!==884||bytes.readUInt32BE(20)!==884)throw Error('Wrong dimensions');
const path=resolve(root,target);
if(process.argv.includes('--check')||existsSync(path)){
  if(!readFileSync(path).equals(bytes))throw Error('Published pixels differ');
}else{mkdirSync(dirname(path),{recursive:true});writeFileSync(path,bytes);}
console.log(JSON.stringify({asset:target,source_cell:11,source_output:outputs.indexOf(matches[0]),sha256:hash(bytes),dimensions_px:[884,884],bytes:bytes.length}));
