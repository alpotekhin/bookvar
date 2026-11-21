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
 '00 Учебник/Assets/Figures/curated/norm-gating-2026-10-22/xiong-warmup.png',
 '00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png',
 '00 Учебник/Assets/Figures/curated/gpt-history-2026-11-09/tajima-figure1-ffn.svg',
 '00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-norm-qk-dropout.png',
 '00 Учебник/Assets/Figures/curated/chameleon-2026-10/chameleon-training-ablations.png',
 '00 Учебник/Assets/Figures/curated/gemma2-norm-2026-11-13/gemma-scope-figure1.svg'
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
 it('uses exact licensed originals and withdraws the cropped legacy collage',()=>{
   const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]|]+)(?:\|[^\]]*)?\]\]/g)].map(m=>m[1]);
   expect(new Set(embeds)).toEqual(new Set(expected));
   expect(embeds).toHaveLength(expected.length);
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
 it('connects the actual warm-up experiment and the two SwiGLU gradient branches to source figures',()=>{
   const warm=body.split('### Что показали эксперименты с warm-up')[1]?.split('id="rmsnorm-validation"')[0];
   for(const value of ['4000 шагов','8,45','31,16','2,77','На всех четырёх панелях архитектура post-LN',
     'https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf#page=4'])expect(warm).toContain(value);
   const gate=body.split('id="ffn-variants"')[1]?.split('id="norm-llama-order"')[0];
   for(const value of ['tajima-figure1-ffn.svg','1.8553','0.7311','SiLU'])expect(gate).toContain(value);
 });
 it('keeps the Gemma 2 output norms before residual addition and pins the exact author diagram',()=>{
   const section=body.split('id="gemma2-sandwich-norm"')[1]?.split('id="ffn-variants"')[0];
   expect(section).toBeDefined();
   for(const value of ['post_attention_layernorm', 'residual + hidden_states', 'GeGLU', 'W_O',
     '0.575,1.916','1.194,1.592,0.119,0.159','48.20','29.92','SAE','2408.05147v2#S3.F1'])
     expect(section).toContain(value);
   const row=assets.find((a:any)=>a.asset===expected.at(-1));
   expect(row).toMatchObject({
     source_image_url:'https://arxiv.org/html/2408.05147v2/sae_location_diagram.svg',
     license_evidence_url:'https://arxiv.org/html/2408.05147v2',
     rights_status:'source-license-verified',
     sha256:'a89a4f331b2aefab50191633add37e4e80691acf63cd325f3ca309653c12b424',
   });
   expect(fm.reference_terms).toContainEqual(expect.objectContaining({anchor:'gemma2-sandwich-norm'}));
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
