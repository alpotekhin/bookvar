import {readFileSync,existsSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const assets=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
const expected=[
 '00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-pre-post-norm.png',
 '00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-gradient-norms.png',
 '00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png',
 '00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-norm-qk-dropout.png',
 '00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-training-ablations.png'
];
describe('normalization and stability chapter',()=>{
 it('contains worked residual, pre/post, RMS derivative and epsilon counterexamples',()=>{
   for(const fact of ['1.0204','-0.0408','0.2041','-0.4082','3.1875',
      '0.256,-0.192','0.4,0,0,0','0.03162','1.66667','65 504','248','231'])
     expect(body).toContain(fact);
   expect(body).not.toContain('слегка меняет');
   expect(body).not.toContain('мистическую');
   expect(body.match(/~~~python\n/g)).toHaveLength(4);
   expect(body).toContain('input_dtype = x.dtype');
   expect(body).toContain('порядок округления различается');
 });
 it('explains QK, z-loss, capping and branch-norm independently rather than naming them',()=>{
   for(const fact of ['7071.07','0.8044','1000.3133','10.0063','0.0000172',
      '9.6403','0.8833','0.1230','4000 шагами warm-up','pRMSNorm',
      'LayerNorm\nдля Q/K','masked_fill(~allowed, -torch.inf)'])expect(body).toContain(fact);
   expect(body).not.toContain('def block(x):');
   expect(body).toContain('1.677');
   expect(body).toContain('524 288');
 });
 it('uses five exact licensed originals and withdraws the cropped legacy collage',()=>{
   const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
   expect(new Set(embeds)).toEqual(new Set(expected));
   expect(embeds).toHaveLength(5);
   expect(new Set(assets.filter((a:any)=>a.used_in?.includes(file)).map((a:any)=>a.asset)))
     .toEqual(new Set(expected));
   for(const path of expected){
     const row=assets.find((a:any)=>a.asset===path);
     expect(row.metadata_status).toBe('verified');
     expect(['CC BY 4.0','MIT']).toContain(row.license);
     expect(createHash('sha256').update(readFileSync(resolve(root,path))).digest('hex')).toBe(row.sha256);
   }
   const old=assets.find((a:any)=>a.asset.endsWith('/rmsnorm-paper-results.png'));
   expect(old.used_in).toEqual([]);
   expect(old.metadata_status).toBe('needs-confirmation');
   expect(existsSync(resolve(root,old.asset))).toBe(true);
 });
 it('preserves five source anchors and the sequential book route',()=>{
   for(const id of ['normalization-stability','rmsnorm-validation','ffn-variants',
     'stability-controls','normalization-evidence'])expect(body.split('id="'+id+'"')).toHaveLength(2);
   expect(fm.source_unit_id).toHaveLength(6);
   expect(fm.previous).toContain('02 Современный decoder block');
   expect(fm.next).toContain('04 RoPE');
   expect(body).toContain('**Назад:**');
   expect(body).toContain('**Далее:**');
 });
});
