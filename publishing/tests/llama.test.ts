import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';
import {beforeTopicConsolidation} from '../tools/topic-consolidation-baseline.mjs';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const hash=(p:string)=>createHash('sha256').update(read(p)).digest('hex');
const chapter='00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура.md';
const english='en/00 Textbook/07 Anatomy of a modern LLM/01 LLaMA as a baseline architecture.md';
const body=read(chapter).toString();
const paper=read('Papers/LLaMA.md').toString();
const concept=read('Concepts/Architectures/LLaMA.md').toString();
const registry='05 Источники/asset-registry.yml';
const assets=YAML.parse(read(registry).toString(),{merge:true}).assets;
const fresh=JSON.parse(read('publishing/tools/llama-figures.json').toString());
const reused=[
 ['00 Учебник/Assets/Figures/curated/positions-2026-10-21/roformer-figure1.png',
  'b1a65eb5e30f1510dce571cedb548b98c1db5c5e57ca04112b856f931a0d2591','CC BY-NC-ND 4.0'],
 ['00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png',
  '59a942959793b641878bf65c6e83c6941e5153dc09a9893c6a5cb44f279f70e0','MIT'],
 ['00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png',
  '02c3d00ec9f339211daae34392aeed850b013457b64f8088f1c5c054a5e94ab6','CC BY 4.0'],
];
const oldIds=[
 'сначала-вся-модель-на-одной-странице','что-унаследовано-от-gpt',
 'изменение-1-нормализация-стоит-перед-подслоем','post-norm-в-исходном-transformer',
 'pre-norm-в-llama','изменение-2-rmsnorm-вместо-layernorm','численный-пример',
 'минимальная-реализация','изменение-3-rope-вместо-прибавления-position-embedding',
 'несколько-частот','чего-rope-не-обещает','изменение-4-swiglu-вместо-обычной-ffn',
 'почему-внутренняя-ширина-выглядит-необычно','изменение-5-меньше-необязательных-параметров',
 'один-токен-проходит-через-блок','проверка-форм-тензоров-по-официальному-коду',
 'llama-1-llama-2-и-llama-3--не-одна-конфигурация','что-llama-не-изобрела',
 'ошибки-при-чтении-схем','практика','1-сопоставить-формулу-и-код',
 '2-реализовать-rmsnorm-и-проверить-инвариант','3-увидеть-относительную-позицию-rope',
 '4-посчитать-параметры-ffn','после-главы-нужно-уметь',
 'литература-и-дополнительные-материалы','основные-статьи','код-и-хорошие-разборы','внутри-базы'
];

describe('LLaMA chapter contracts',()=>{
 it('keeps the current English edition complete and uses the same original assets',()=>{
   const translation=read(english).toString();
   const fm=YAML.parse(translation.split('---')[1]);
   expect(fm).toMatchObject({locale:'en',translation_status:'current',
     translation_of:chapter,canonical_revision:hash(chapter)});
   const extract=(text:string,pattern:RegExp)=>[...text.matchAll(pattern)].map(m=>m[0]);
   expect(extract(translation,/^~~~python\n[\s\S]*?^~~~$/gm))
     .toEqual(extract(body,/^~~~python\n[\s\S]*?^~~~$/gm));
   expect(extract(translation,/^\$\$\n[\s\S]*?^\$\$$/gm))
     .toEqual(extract(body.replace('\\text{байт}','\\text{bytes}'),/^\$\$\n[\s\S]*?^\$\$$/gm));
   const excerpts=/^> (?:Given a sequence|Training loss over train tokens).*(?:\n> .*)*/gm;
   expect(extract(translation,excerpts)).toEqual(extract(body,excerpts));
   const images=(text:string)=>[...text.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]|]+)/g)].map(m=>m[1]);
   expect(images(translation)).toEqual(images(body));
   expect(assets.filter((asset:any)=>asset.used_in?.includes(english)).map((asset:any)=>asset.asset).sort())
     .toEqual(images(body).sort());
   for(const id of body.matchAll(/<(?:a|span) id="([^"]+)"/g))
     expect(translation.split('id="'+id[1]+'"')).toHaveLength(2);
   expect(extract(translation,/^#{2,3} /gm)).toEqual(extract(body,/^#{2,3} /gm));
 });
 it('retains worked examples and version-specific source boundaries',()=>{
   expect(body.match(/~~~python\n/g)).toHaveLength(3);
   for(const fact of ['6 738 415 616','0.068866','2.405477','2.366481','norm_eps',
     '46 обучаемых','h[:, -1, :]','8192','34B','256','128K','Answer:',
     '1076b9c51c77ad06e9d7ba8a4c6df775741732bd'])
     expect(body).toContain(fact);
   expect(body).not.toContain('┌');
   expect(body).not.toContain('mermaid');
   expect(body).not.toMatch(/^### [1-4]\./m);
   expect(body).not.toContain('Взвешенная сумма V приносит контекст');
 });
 it('retains the seven original chapter figures and registers every current comparison figure',()=>{
   const stop='2026-11-09-attention-gpt1-pilot-additions.json';
   const previous=beforeTopicConsolidation(body,chapter,root,stop);
   const formerAssets=YAML.parse(beforeTopicConsolidation(read(registry).toString(),registry,root,stop),
     {merge:true}).assets;
   const images=(text:string)=>[...text.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]|]+)(?:\|[^\]]*)?\]\]/g)]
     .map(m=>m[1]);
   const embedded=images(previous);
   const expected=[...fresh.map((s:any)=>s.asset),...reused.map(s=>s[0])];
   expect(embedded).toHaveLength(7);
   expect(new Set(embedded)).toEqual(new Set(expected));
   expect(new Set(formerAssets.filter((r:any)=>r.used_in?.includes(chapter)).map((r:any)=>r.asset)))
     .toEqual(new Set(expected));
   const extra=[
     'norm-gating-2026-10-22/xiong-pre-post-norm.png',
     'gpt2-2026-10-22/gpt2-output-projection-2.png',
     'gpt2-2026-10-22/gpt2-input-embedding-positional-encoding-3.png',
     'gpt-continuity-2026-11-10/min-figure2-demonstrations.png',
     'long-context-review-2026-09/mistral-sliding-window.png',
     'gpt-30/gpt2-token-embeddings-wte-2.png',
     'gpt-history-2026-11-09/gulordava-figure1-ab.png',
     'gpt-history-2026-11-09/llama1-figure3-natural-questions.png',
     'gpt-history-2026-11-09/zhao21c-figure2.png',
     'gpt-history-2026-11-09/tajima-figure1-ffn.svg',
     'gpt-history-2026-11-09/pan-icl-figure1.png',
     'gpt-history-2026-11-09/pan-icl-figure2.png',
     'llama-scaling-2026-11-09/sardana24a-figure1.png',
     'llama-scaling-2026-11-09/sardana24a-figure2b.png',
     'llama-generations-2026-11-10/gao25-figure1-sft.png',
     'llama-generations-2026-11-10/gao25-figure2-data-mix.png',
     'gqa-2026-10-22/groups-speed.png',
   ].map(name=>'00 Учебник/Assets/Figures/curated/'+name);
   const current=images(body);
   expect(current).toHaveLength(expected.length+extra.length);
   expect(new Set(current)).toEqual(new Set([...expected,...extra]));
   expect(assets.filter((r:any)=>r.used_in?.includes(chapter)).map((r:any)=>r.asset).sort())
     .toEqual([...current].sort());
   for(const asset of extra){
     const row=assets.find((r:any)=>r.asset===asset);
     expect(hash(asset)).toBe(row.sha256);
     expect(row.source_url).toMatch(/^https:\/\//);
     expect(row.license).toMatch(/^CC BY/);
   }
   for(const f of fresh){
     const rows=assets.filter((r:any)=>r.asset===f.asset);
     expect(rows).toHaveLength(1);
     for(const [key,value] of Object.entries(f)){
       if(['name','width','height'].includes(key))continue;
       expect(rows[0][key]).toEqual(value);
     }
     expect(hash(f.asset)).toBe(f.sha256);
     expect(hash(f.source_asset)).toBe(f.source_asset_sha256);
     expect(hash(f.license_evidence)).toBe(f.license_evidence_sha256);
     const png=read(f.asset);
     expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([f.width,f.height]);
   }
   for(const [asset,sha,license] of reused){
     expect(hash(asset)).toBe(sha);
     const row=assets.find((r:any)=>r.asset===asset);
     expect(row.sha256).toBe(sha);
     expect(row.license).toBe(license);
     expect(row.used_in).toContain(chapter);
   }
   expect(hash('05 Источники/Originals/llama-2026-10-22/gqa-source.png'))
      .toBe(reused[2][1]);
   expect(body).toContain('Описание модели в CS336 (English, §3.1)');
   expect(body).toContain('Оригинальная подпись к Figure 1 LLaMA (English)');
 });
 it('preserves the source experiment and shows the same two prompt encodings in both editions',()=>{
   const sample=/```text title="Llama (?:2 Chat|3 Instruct)"\n([\s\S]+?)```/g;
   const prompts=[...body.matchAll(sample)].map(m=>m[1]);
   expect(prompts).toHaveLength(2);
   expect([...read(english).toString().matchAll(sample)].map(m=>m[1])).toEqual(prompts);
   expect(prompts[0]).toMatch(/^<s>\[INST\].+\[\/INST\]\n$/);
   expect(prompts[1]).toMatch(/<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\n\n$/);
   const chart=assets.find((r:any)=>r.id==='curated-gao25-long-context-sft-f4a36c325e98');
   expect(hash(chart.source_asset)).toBe(chart.source_asset_sha256);
   expect(chart.figure_url).toBe('https://aclanthology.org/2025.acl-long.366.pdf#page=2');
   const retired=assets.find((r:any)=>r.id==='curated-atlas-llama3-00000000002d');
   expect(retired).toMatchObject({publication_status:'retired',used_in:[]});
   expect(hash(retired.asset)).toBe(retired.sha256);
 });
 it('presents the historical changes in order without equating task prompts and training',()=>{
   for(const text of [body,read(english).toString()]){
     const start=text.indexOf('id="gpt-llama-comparison"');
     const transitions=['gpt1-to-gpt2','gpt2-to-gpt3','gpt3-to-llama1','gpt-llama-ffn']
       .map(id=>text.indexOf(`id="${id}"`,start));
     expect(transitions.every((offset,i)=>offset>(i===0?start:transitions[i-1]))).toBe(true);
     const [gpt2,gpt3,llama,ffn]=transitions;
     expect(text.slice(gpt2,gpt3)).toContain('xiong-pre-post-norm.png');
     expect(text.slice(gpt3,llama)).toContain('min-figure2-demonstrations.png');
     expect(text.slice(gpt3,llama)).toContain('mistral-sliding-window.png');
     const positions=text.slice(llama,ffn);
     expect(positions).toMatch(/id="gpt-llama-position-shift"[\s\S]*gpt2-input-embedding-positional-encoding-3\.png[\s\S]*roformer-figure1\.png/);
     expect(positions).toContain('-0.8415');
     expect(text.slice(ffn,text.indexOf('id="сначала-вся-модель-на-одной-странице"'))).toContain('tajima-figure1-ffn.svg');
     for(const count of [13e9,175e9]){
       const tokens=count===13e9?1e12:300e9;
       const flops=6*count*tokens;
       const [mantissa,exponent]=flops.toExponential(2).split('e+');
       expect(text).toContain(`${Number(mantissa)}\\cdot10^{${exponent}}`);
     }
     expect(text).toContain('2.34\\cdot10^{22}');
   }
 });
 it('preserves the shifted RoPE attention output but not the learned absolute input',()=>{
   type Pair=[number,number];
   const embed:Pair[]=[[1,0],[0,1]];
   const table:Pair[]=[[0,0],[1,0],[0,1]];
   const dot=([a,b]:Pair,[c,d]:Pair)=>a*c+b*d;
   const rotation=([x,y]:Pair,angle:number):Pair=>[
     x*Math.cos(angle)-y*Math.sin(angle),x*Math.sin(angle)+y*Math.cos(angle),
   ];
   const absolute=(shift:number)=>embed.map(([x,y],i):Pair=>[x+table[i+shift][0],y+table[i+shift][1]]);
   const absoluteScore=(shift:number)=>{
     const [key,query]=absolute(shift);
     return dot(query,key);
   };
   expect(absoluteScore(0)).toBe(1);
   expect(absoluteScore(1)).toBe(0);
   expect(absolute(0)).not.toEqual(absolute(1));
   const rotary=(shift:number)=>{
     const [key,query]=embed.map((e,i)=>rotation(e,i+shift));
     const scores=[dot(query,key),dot(query,query)];
     const [a,b]=scores.map(s=>Math.exp(s/Math.sqrt(2)));
     return {scores,output:[a/(a+b),b/(a+b)]};
   };
   expect(rotary(0).scores[0]).toBeCloseTo(-0.8415,4);
   for(const i of [0,1]){
     expect(rotary(1).scores[i]).toBeCloseTo(rotary(0).scores[i],12);
     expect(rotary(1).output[i]).toBeCloseTo(rotary(0).output[i],12);
   }
 });
 it('retains a complete published cache loop and its off-by-one boundary',()=>{
   for(const fact of ['def forward(ids, cache=None)', 'start_pos + x.shape[1]',
     'query_pos = old + torch.arange(t)', 'cache["k"].shape[1]',
     'processed += pending.shape[1]', 'torch.testing.assert_close(logits[:, -1], complete[:, -1])',
     '2+3+4=9', 'Последний выбранный ID', 'Маска имеет форму 2×5',
     'Когда можно использовать сохранённый префикс']) expect(body).toContain(fact);
   expect(body).toContain('llama1-downstream-training.png');
 });
 it('corrects legacy notes and registers their real original figures',()=>{
   expect(YAML.parse(paper.split('---')[1]).Date).toBe('2023-02-27');
   for(const fact of ['52.4 | 68.5','50.9 | 69.7','CrowS-Pairs','70.6','77.5',
     '27 февраля 2023','0.1 для pass@1','0.8 для pass@100'])expect(paper).toContain(fact);
   for(const text of [paper,concept]){
     expect(text).toContain('01 LLaMA как базовая архитектура');
     expect(text).not.toContain('Gender bias: 70.6% верных');
     expect(text).not.toContain('PaLM-Coder 62B | 36.0');
   }
   expect(concept).toContain('8K, более 15T');
   expect(concept).not.toContain('Apr 2024 | 8B/70B, 15T tokens, 128K');
   for(const [path,text] of [['Papers/LLaMA.md',paper],['Concepts/Architectures/LLaMA.md',concept]]){
     const embedded=[...text.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
     expect(embedded.length).toBe(path.startsWith('Papers') ? 1 : 2);
     expect(new Set(assets.filter((r:any)=>r.used_in?.includes(path)).map((r:any)=>r.asset)))
       .toEqual(new Set(embedded));
   }
 });
 it('preserves all previous H2/H3 targets and real chapter order',()=>{
   for(const id of oldIds)expect(body.split('id="'+id+'"')).toHaveLength(2);
   const fm=YAML.parse(body.split('---')[1]);
   expect(fm.previous).toContain('06 GPT-3');
   expect(fm.next).toContain('02 Современный decoder block');
   const footer=body.slice(body.lastIndexOf('**Назад:**'));
   expect(footer).toContain(fm.previous.slice(0,-2)+'|');
   expect(footer).toContain(fm.next.slice(0,-2)+'|');
 });
});
