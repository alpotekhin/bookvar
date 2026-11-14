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
const originals='00 Учебник/Assets/Figures/curated/decoder-design-2026-11-10/';
const gemma='00 Учебник/Assets/Figures/curated/gemma/gemma-3-kv-cache.svg';
const mistral='00 Учебник/Assets/Figures/curated/long-context-2026-10-23/mistral-chunk-mask.png';
const originalSource='05 Источники/Originals/decoder-design-2026-11-10/';
const figureSpecs=[
 ['mobilellm-design-figure3.png', 'c4091f148edef89d116a20814dd2f32475aa13e2c3ff67cf60447090d4903e8c', 'mobilellm.pdf',
  '0e327790c3b5c233d0c2b4a0259409445dc074e235f2422376d7ab7f58f8c49f', 'https://proceedings.mlr.press/v235/liu24ce.html', 3, [255,320,1470,1870]],
 ['mobilellm-depth-figure4.png', '53e04edfc7faa4cf707773ca4b5dddab1de6ee5a418bc1598a9f8a1502655f81', 'mobilellm.pdf',
  '0e327790c3b5c233d0c2b4a0259409445dc074e235f2422376d7ab7f58f8c49f', 'https://proceedings.mlr.press/v235/liu24ce.html', 4, [257,319,2734,1334]],
 ['han-weight-decay-figure3.png', '474a78b38407e459c71415cd212003db761c9b1528b0a99ecccb2179b4f56b49', 'weight-decay-plasticity.pdf',
  'a4c50f6d94d5f2dd91e277924b5e0893de75c71344ca4d1f16749da75549ed52', 'https://proceedings.mlr.press/v306/han26b.html', 6, [255,321,2736,1378]],
] as const;
const images=[
 '00 Учебник/Assets/Figures/curated/llama-2026-10-22/cs336-prenorm-block.png',
 '00 Учебник/Assets/Figures/curated/gqa/mha-gqa-mqa.png',
 '00 Учебник/Assets/Figures/curated/gpt-30/gpt2-transformer-block-vectors-2.png',
 '00 Учебник/Assets/Figures/curated/edls-neural-kv-2026-10-26/hf-kv-cache.png',
 gemma,
 mistral,
 ...figureSpecs.map(([name])=>originals+name)
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
   expect(body.match(/~~~python\n/g)).toHaveLength(4);
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
 it('links the author figures and preserves the three retired composite slides in the archive',()=>{
   const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^|\]]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
   expect(embeds).toHaveLength(9);
   expect(new Set(embeds)).toEqual(new Set(images));
   expect(new Set(registry.filter((r:any)=>r.used_in?.includes(path)).map((r:any)=>r.asset)))
     .toEqual(new Set(images));
   const mistralHash='c8a46ec808ada9448b72ee0679cb464dac5bc13a7f1cafe9cc8eeede5886be69';
   expect(hash(mistral)).toBe(mistralHash);
   expect(registry.find((r:any)=>r.asset===mistral)).toMatchObject({
     sha256:mistralHash,license:'CC BY 4.0',
     used_in:['00 Учебник/08 Эффективный Attention и длинный контекст/03 Длинный контекст — расширение, разреженность и оценивание.md',path],
   });
   for(const [i,s] of slides.entries()){
     const row=registry.find((r:any)=>r.asset===s.asset);
     expect(row).toBeDefined();
     for(const [key,value] of Object.entries(s))if(key!=='name')expect(row[key]).toEqual(value);
     expect(hash(s.asset)).toBe(s.sha256);
     expect(hash(s.source_asset)).toBe(s.source_asset_sha256);
     expect(row.license).not.toBe('MIT');
     expect(row.metadata_status).toBe('needs-confirmation');
     expect(row).toMatchObject({publication_status:'retired',used_in:[],replacement:originals+figureSpecs[i][0]});
   }
   for(const [name,digest,source,sourceDigest,url,page,crop] of figureSpecs){
     const asset=originals+name;
     const row=registry.find((r:any)=>r.asset===asset);
     expect(hash(asset)).toBe(digest);
     expect(row).toMatchObject({
       sha256:digest, source_asset:originalSource+source, source_asset_sha256:sourceDigest,
       source_url:url,source_page:page,render_dpi:360,crop_box_px:crop,
       derivation:'pdf-page-render-crop',license:'CC BY 4.0',rights_status:'source-license-verified',
       metadata_status:'verified',used_in:[path]
     });
   }
 });
 it('connects the output head to the first cached decode step',()=>{
   const section=body.split('<a id="decoder-first-token"></a>')[1]?.split('### Почему кэш')[0];
   expect(section).toBeDefined();
   for(const fact of ['0.1996, 1.1409,<br>1.6305, 0.0027','1.0877, 0.8951,<br>0.2321, 1.4006',
     'start_pos=2, cache=past','past[0].shape[2] == 2','past[0].shape[2] == 3',
     'logits[:, -1]','generation.py#L160-L186']) expect(section).toContain(fact);
   const fig=images.find(asset=>asset.endsWith('/hf-kv-cache.png'))!;
   expect(hash(fig)).toBe('44931e339df5ebc418385dcb1880a75da44eeeaf5c4ae3a6aab853d8916bd680');
 });
 it('connects real Qwen tensor widths and a mixed Gemma cache to the worked block',()=>{
   for(const detail of ['id="qwen3-checkpoint-shapes"','id="gemma3-mixed-kv-cache"',
     '[2048,1024]','[1024,2048]','[1,3,16,128]','[1,3,8,128]',
     '596 049 920','310 тензоров','448 МиБ','576 МиБ','19,3%','7000 МБ','1100 МБ'])
     expect(body).toContain(detail);
   const sha256='304affd7e22f11162d405f074e64c23fa08ab580407af332cf57e8ffcd17abf5';
   expect(hash(gemma)).toBe(sha256);
   expect(registry.find((r:any)=>r.asset===gemma)).toMatchObject({
     source_url:'https://arxiv.org/html/2503.19786v1#S5.F6',
     source_image_url:'https://arxiv.org/html/2503.19786v1/mem_lc.svg',
     license:'CC BY 4.0',sha256,
     used_in:['02 Атлас моделей/Семейства/Gemma.md',path],
   });
 });
 it('compares the PMLR PDF and publication agreement with the retained archive', { tags: ['source-archive'] }, ()=>{
   for(const [name,,source,sourceDigest] of figureSpecs){
     const row=registry.find((r:any)=>r.asset===originals+name);
     expect(hash(originalSource+source)).toBe(sourceDigest);
     expect(hash(row.license_evidence)).toBe('3a3eac0ebdded57660c12c59028c5429b6d2b328be2c6fde534a104438444c31');
   }
 });
 it('distinguishes the FFN, depth and weight-decay comparisons',()=>{
   for(const fact of ['42.6% до 43.9%','376.8 до 386.7 млн',
     '43.9% до 44.8%','28.9% до 26.1%','2.6208 против 2.6088',
     'не более десяти точек','Table 10, с. 14','Table 11, с. 15'])
     expect(body).toContain(fact);
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
