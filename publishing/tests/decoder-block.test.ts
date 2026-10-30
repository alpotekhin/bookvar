import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const path='00 Учебник/07 Анатомия современной LLM/02 Современный decoder block.md';
const body=readFileSync(resolve(root,path),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const registry=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
const slides=JSON.parse(readFileSync(resolve(root,'publishing/tools/decoder-block-figures.json'),'utf8'));
const images=[
 '00 Учебник/Assets/Figures/curated/llama-2026-10-22/cs336-prenorm-block.png',
 '00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png',
 '00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png',
 ...slides.map((s:any)=>s.asset)
];
const hash=(p:string)=>createHash('sha256').update(readFileSync(resolve(root,p))).digest('hex');
const old=[
 ['два-последовательных-уточнения','Два последовательных уточнения'],
 ['что-происходит-внутри-attention','Что происходит внутри attention'],
 ['что-именно-экономят-gqa-mqa-и-разрежённое-attention','Что именно экономят GQA, MQA и разрежённое attention'],
 ['что-происходит-внутри-swiglu','Что происходит внутри SwiGLU'],
 ['прямой-проход-без-скрытых-шагов','Прямой проход без скрытых шагов'],
 ['от-gpt-2-к-llama-что-сохранилось-а-что-заменили','От GPT-2 к LLaMA: что сохранилось, а что заменили'],
 ['обучение-prefill-и-decode--три-разных-режима-чтения-блока','Обучение, prefill и decode — три разных режима чтения блока'],
 ['как-выбирают-размеры-блока','Как выбирают размеры блока'],
 ['ширина-ffn','Ширина FFN'],['число-голов-и-размер-головы','Число голов и размер головы'],
 ['глубина-и-ширина','Глубина и ширина'],['размер-словаря','Размер словаря'],
 ['dropout-и-weight-decay','Dropout и weight decay'],
 ['что-считать-подтверждением-корректной-сборки','Что считать подтверждением корректной сборки'],
 ['где-реализация-чаще-всего-расходится-с-формулой','Где реализация чаще всего расходится с формулой'],
 ['источники-и-реализации','Источники и реализации']
];
describe('modern decoder chapter contracts',()=>{
 it('has a fully specified worked block, cache comparison and version-specific boundaries',()=>{
   expect(body.match(/~~~python\n/g)).toHaveLength(2);
   for(const fact of ['128 его','0.0685','0.5189','2.1306','2.0984','96 байт','42.7',
       'start_pos=1','window','152064','dropout_rate=0.1','Параллельные подслои PaLM']){
     expect(body).toContain(fact);
   }
   expect(body).not.toContain('def decoder_block(x, positions, causal_mask):');
   expect(body).not.toContain('это главный инвариант decoder-only');
   expect(body).not.toContain('мермайд');
   expect(fm.source_unit_id).toEqual([
     'lecture-03-hyperparameter-evidence','lecture-03-kv-cache-attention','lecture-03-sparse-attention']);
 });
 it('works through SwiGLU products, per-layer cache and the backward boundary',()=>{
   for(const fact of ['−1.5672','1.8936','−0.5123','1.2456',
     'Два блока и два кэша','copy.deepcopy(model)','1.4761','0.0090',
     '192 байта','3.1494','0.9441','detach()','градиент по первой входной',
     'строке станет нулевым','@torch.inference_mode()'])expect(body).toContain(fact);
   expect(body).toContain('"product": product');
   expect(body).toContain('torch.cat((prefix, suffix), dim=1)');
 });
 it('preserves six original images, exact registry usages and honest slide provenance',()=>{
   const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^\]]+)\]\]/g)].map(m=>m[1]);
   expect(embeds).toHaveLength(6);
   expect(new Set(embeds)).toEqual(new Set(images));
   expect(new Set(registry.filter((r:any)=>r.used_in?.includes(path)).map((r:any)=>r.asset)))
     .toEqual(new Set(images));
   for(const s of slides){
     const row=registry.find((r:any)=>r.asset===s.asset);
     expect(row).toBeDefined();
     for(const [key,value] of Object.entries(s))if(key!=='name')expect(row[key]).toEqual(value);
     expect(hash(s.asset)).toBe(s.sha256);
     expect(hash(s.source_asset)).toBe(s.source_asset_sha256);
     expect(row.license).not.toBe('MIT');
     expect(row.metadata_status).toBe('needs-confirmation');
   }
 });
 it('preserves previous headings, four source anchors and adjacent chapter order',()=>{
   for(const [id,text] of old)
     expect(body.includes('id="'+id+'"') || body.includes('## '+text)).toBe(true);
   for(const id of ['modern-decoder-architecture','efficient-attention-boundary',
     'decoder-hyperparameter-evidence','decoder-evidence'])
     expect(body.split('id="'+id+'"')).toHaveLength(2);
   expect(fm.previous).toContain('01 LLaMA');
   expect(fm.next).toContain('03 Pre-norm');
   const footer=body.slice(body.lastIndexOf('**Назад:**'));
   expect(footer).toContain(fm.previous.slice(0,-2)+'|');
   expect(footer).toContain(fm.next.slice(0,-2)+'|');
 });
});
