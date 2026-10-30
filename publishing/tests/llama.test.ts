import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(p:string)=>readFileSync(resolve(root,p));
const hash=(p:string)=>createHash('sha256').update(read(p)).digest('hex');
const chapter='00 Учебник/07 Анатомия современной LLM/01 LLaMA как базовая архитектура.md';
const body=read(chapter).toString();
const paper=read('Papers/LLaMA.md').toString();
const concept=read('Concepts/Architectures/LLaMA.md').toString();
const assets=YAML.parse(read('05 Источники/asset-registry.yml').toString()).assets;
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
 it('embeds seven genuine source figures and registers precisely their current usages',()=>{
   const embedded=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
   const expected=[...fresh.map((s:any)=>s.asset),...reused.map(s=>s[0])];
   expect(embedded).toHaveLength(7);
   expect(new Set(embedded)).toEqual(new Set(expected));
   expect(new Set(assets.filter((r:any)=>r.used_in?.includes(chapter)).map((r:any)=>r.asset)))
     .toEqual(new Set(expected));
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
   expect(fm.previous).toContain('07 T5');
   expect(fm.next).toContain('02 Современный decoder block');
   const footer=body.slice(body.lastIndexOf('**Назад:**'));
   expect(footer).toContain(fm.previous.slice(0,-2)+'|');
   expect(footer).toContain(fm.next.slice(0,-2)+'|');
 });
});
