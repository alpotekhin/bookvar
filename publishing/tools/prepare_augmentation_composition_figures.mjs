import {createHash} from 'node:crypto';
import {existsSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {dirname,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'../..');
const folder="00 Учебник/Assets/Figures/curated/augmentation-composition-2026-11-03/";
const rows=JSON.parse(readFileSync(resolve(root,folder,'assets.json'),'utf8'));
const sha=b=>createHash('sha256').update(b).digest('hex');
for(const row of rows){
  for(const [pathKey,hashKey] of [
    ['source_asset','source_asset_sha256'],['source_page_snapshot','source_page_sha256'],
    ['source_code_snapshot','source_code_sha256'],['license_evidence','license_evidence_sha256']]){
    if(sha(readFileSync(resolve(root,row[pathKey])))!==row[hashKey])throw Error('Source changed: '+row[pathKey]);
  }
  const bytes=readFileSync(resolve(root,row.source_asset)),svg=bytes.toString();
  if(sha(bytes)!==row.sha256||(svg.match(/<image\b/g)||[]).length!==8
    || /(?:href|src)="https?:/.test(svg)) throw Error('Invalid or externally dependent SVG');
  const page=readFileSync(resolve(root,row.source_page_snapshot),'utf8');
  const panel=page.split('id="pytorch-17-0"')[1]?.split('id="mxnet-17-1"')[0];
  if(!panel?.includes('output_image-augmentation_7d0887_81_0.svg'))
    throw Error('Figure does not match the archived PyTorch panel');
  const target=resolve(root,row.asset);
  if(process.argv.includes('--check')||existsSync(target)){
    if(!readFileSync(target).equals(bytes))throw Error('Original image bytes differ');
  }else{mkdirSync(dirname(target),{recursive:true});writeFileSync(target,bytes);}
  if(!readFileSync(resolve(root,row.public_license_copy)).equals(
    readFileSync(resolve(root,row.license_evidence)))) throw Error('Public license differs');
  console.log(JSON.stringify({asset:row.asset,sha256:row.sha256,embeddedPanels:8,sourcePanel:'pytorch-17-0'}));
}
