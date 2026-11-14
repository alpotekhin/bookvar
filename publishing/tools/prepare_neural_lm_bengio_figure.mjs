import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const folder='00 Учебник/Assets/Figures/curated/neural-lm-bengio-2026-11-03/';
const [asset]=JSON.parse(readFileSync(resolve(root,folder+'assets.json'),'utf8'));
const sha=b=>createHash('sha256').update(b).digest('hex');
if(sha(readFileSync(resolve(root,asset.source_asset)))!==asset.source_asset_sha256)throw Error('Source PDF changed');
const bytes=execFileSync(process.env.PDFTOPPM ?? 'pdftoppm',
  ['-f','6','-l','6','-singlefile','-r','240','-x','273','-y','283','-W','1507','-H','980',
    '-png',resolve(root,asset.source_asset)],
  {env:{...process.env,LC_ALL:'C',LANG:'C'},maxBuffer:4*1024*1024});
if(sha(bytes)!==asset.sha256)throw Error('Original figure render changed');
if(bytes.readUInt32BE(16)!==1507||bytes.readUInt32BE(20)!==980)throw Error('Unexpected crop geometry');
const output=resolve(root,asset.asset);
if(process.argv.includes('--check')||existsSync(output)){
  if(!readFileSync(output).equals(bytes))throw Error('Published figure differs from source crop');
}else{mkdirSync(dirname(output),{recursive:true});writeFileSync(output,bytes);}
console.log(JSON.stringify({asset:asset.asset,sha256:asset.sha256,page:6,dimensions_px:[1507,980],originalCaption:true}));
