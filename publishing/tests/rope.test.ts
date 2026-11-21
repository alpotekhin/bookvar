import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {createHash} from 'node:crypto';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/07 Анатомия современной LLM/04 RoPE.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const specs=JSON.parse(readFileSync(resolve(root,'publishing/tools/rope-figures.json'),'utf8'));
const rows=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
describe('RoPE explanation and examples',()=>{
  it('keeps the causal-mask qualification and independent sign/pairing examples',()=>{
    for(const text of ['Причинная маска уже задаёт порядок','при фиксированных $q,k$',
      'ответ $4-6=-2$','R_{\\rm split}=P R_{\\rm adjacent}P^T',
      'Проекции V и выхода\nостаются прежними','невращаемые координаты'])
      expect(body.toLowerCase()).toContain(text.toLowerCase());
    expect(body.match(/~~~python\n/g)).toHaveLength(6);
    expect(body).toContain('import torch');
    expect(body).toContain('positions.shape not in ((1, T), (B, T))');
  });
  it('explains cached attention through actual values and the changed-grid counterexample',()=>{
    for(const text of ['3.701805','0.117410','1.890151',
      'k_positions[None, :] <= q_positions[:, None]',
      'torch.testing.assert_close(cached, full)','stale_kv',
      '0.00232628','Кэш всё ещё экономит','Переиспользовать готовый кэш без преобразований'])
      expect(body).toContain(text);
    expect(body).toMatch(/конфигураци[яи]\s+RoPE/);
    expect(body).not.toContain('уничтожает\\nодно из преимуществ');
  });
  it('separates PI equivalence, specific scaling implementations and their limits',()=>{
    for(const text of ['2047.75','63496.04','0.00625','1.29648','1000 шагов',
      '400 шагов','200 шагов','линейная функция $r_j$',
      'не измерение attention logits','за $2^{24}$'])
      expect(body).toContain(text);
    expect(body).toContain('не нужно представлять как две разные');
    expect(fm.previous).toContain('03 Pre-norm');
    expect(fm.next).toContain('01 Dense FFN');
    expect(body).toContain('**Назад:**');
    expect(body).toContain('**Далее:**');
  });
  it('preserves the original figures and includes both Barbero positional-head figures',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^|\]]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    const newFigure='00 Учебник/Assets/Figures/curated/rope-llama31-2026-11-10/barbero-llama31-frequency-figure10.png';
    const headFigures=[
      ['barbero-previous-token-construction-figure1',2,1854,753,'688f1a391295653db04e587304c7d1b04c9b20503cc92edc1c6cdb9c3087cd98'],
      ['barbero-gemma-position-heads-figure5',7,1875,790,'05062d906976e5185d9f6ee24b51390bb5aa68e766c58f963ac9d43e46ed854d'],
    ] as const;
    const assets=headFigures.map(([name])=>newFigure.replace('barbero-llama31-frequency-figure10',name));
    expect(embeds).toHaveLength(8);
    expect(new Set(embeds)).toEqual(new Set([...specs.map((s:any)=>s.asset),newFigure,...assets]));
    for (const [i,[,page,w,h,sha256]] of headFigures.entries()){
      const row=rows.find((r:any)=>r.asset===assets[i]);
      expect(row).toMatchObject({source_page:page,dimensions_px:[w,h],sha256,
        source_image_url:'https://arxiv.org/pdf/2410.06205v3#page='+page,
        license:'CC BY 4.0',used_in:[file]});
      const png=readFileSync(resolve(root,row.asset));
      expect([png.readUInt32BE(16),png.readUInt32BE(20)]).toEqual([w,h]);
      expect(createHash('sha256').update(png).digest('hex')).toBe(sha256);
    }
    expect(new Set(rows.filter((a:any)=>a.used_in?.includes(file)).map((a:any)=>a.asset)))
      .toEqual(new Set(embeds));
    for(const spec of specs){
      expect(createHash('sha256').update(readFileSync(resolve(root,spec.asset))).digest('hex')).toBe(spec.sha256);
      expect(createHash('sha256').update(readFileSync(resolve(root,spec.source_asset))).digest('hex')).toBe(spec.source_asset_sha256);
      const row=rows.find((r:any)=>r.asset===spec.asset);
      expect(row.metadata_status).toBe('verified');
      expect(['CC BY 4.0','CC BY-NC-ND 4.0']).toContain(row.license);
    }
    const original=rows.find((r:any)=>r.asset===newFigure);
    expect(original).toMatchObject({
      source_url:'https://arxiv.org/html/2410.06205v3#A3.F10',
      source_asset:'05 Источники/Originals/rope-llama31-2026-11-10/barbero-v3.pdf', source_page:20, render_dpi:360,
      crop_box_px:[528,447,2540,2478], dimensions_px:[2012,2031],
      sha256:'4da3835ff463b369c527f40c9b9c6c2bb25a9f65ea9441f0dcdf86ca471f59ea',
      source_asset_sha256:'f16f469d75f3b6b893a299d28af88e953436d5cb5b3ba10522a3630470dd5e52',
      license:'CC BY 4.0', rights_status:'source-license-verified', used_in:[file]
    });
    for(const [path,digest] of [[newFigure,original.sha256],
        [original.source_asset,original.source_asset_sha256],
        [original.license_evidence,'6dc51010ed608883aa163e9d1b39604366e4544893dcab4e62c9193a21bf2d6f']]){
      expect(createHash('sha256').update(readFileSync(resolve(root,path))).digest('hex')).toBe(digest);
    }
  });
  it('shows how pre-rotating the query favours the preceding key',()=>{
    const weights=(scale:number)=>{
      const logits=[0,1,2,3].map(key=>scale**2*Math.cos(key-2)/Math.sqrt(2));
      const mass=logits.map(Math.exp);
      return mass.map(x=>x/mass.reduce((a,b)=>a+b));
    };
    expect(weights(1)[2]).toBeCloseTo(.356,3);
    expect(weights(3)[2]).toBeCloseTo(.903,3);
    expect(weights(3).indexOf(Math.max(...weights(3)))).toBe(2);
    for(const fact of ['11.14517','6.88781','19.66354','0.356','0.903'])
      expect(body).toContain(fact);
  });
  it('separates static Llama 3.1 scaling from context length and learned frequency usage',()=>{
    for(const fact of ['id="rope-llama31-scaling"','29 сохраняют частоту','6 попадают',
      '0.37112','0.15594 до 0.86540','800 млрд токенов',
      'у каждой\nпанели своя шкала','его значение и знак зависят от угла'])
      expect(body).toContain(fact);
  });
});
