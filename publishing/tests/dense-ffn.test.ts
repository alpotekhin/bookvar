import {readFileSync,existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import YAML from 'yaml';
import {describe,it,expect} from 'vitest';
const root=resolve(import.meta.dirname,'../..');
const file='00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md';
const body=readFileSync(resolve(root,file),'utf8');
const fm=YAML.parse(body.split('---')[1]);
const assets=YAML.parse(readFileSync(resolve(root,'05 Источники/asset-registry.yml'),'utf8')).assets;
describe('dense FFN explanation and source preservation',()=>{
  it('preserves all eight earlier figures and registers both additions',()=>{
    const embeds=[...body.matchAll(/!\[\[02 Areas\/ML & DL\/([^|\]]+)(?:\|[^\]]+)?\]\]/g)].map(m=>m[1]);
    expect(embeds).toHaveLength(10);
    expect(embeds.filter(p=>p.endsWith('.gif'))).toHaveLength(2);
    for(const path of embeds){
      expect(existsSync(resolve(root,path))).toBe(true);
      expect(assets.find((a:any)=>a.asset===path)?.used_in).toContain(file);
    }
    expect(body).toContain('Mixtral of Experts, Figure 1');
    expect(body).not.toContain('Отдельная лицензия изображения');
  });
  it('retains both complete author SVGs and links the versioned captions',()=>{
    for(const [filename,figure,sha] of [
      ['tajima-figure1-ffn.svg','https://arxiv.org/html/2506.23225v1#S1.F1',
        'e8c8f485f507726c232e1a659346b9a7c4ae242e4d6f5991e8c4f18c674f9df7'],
      ['geva-ffn-vocabulary.svg','https://arxiv.org/html/2203.14680v1#S1.F1',
        'eda7dfb34974a12ff3a837da9a57ae933c5d915cba3b8d1d0664bd6d38b5a67e'],
    ]){
      const asset=assets.find((a:any)=>a.asset.endsWith('/'+filename));
      expect(asset).toMatchObject({figure_url:figure,license:'CC BY 4.0',
        source_asset_sha256:sha,sha256:sha,derivation:'source-svg-copy'});
      expect(createHash('sha256').update(readFileSync(resolve(root,asset.asset))).digest('hex')).toBe(sha);
      expect(body).toContain(figure);
    }
  });
  it('distinguishes raising a logit from raising its softmax probability',()=>{
    const exps=[1,2,0].map(x=>Math.exp(x));
    const total=exps.reduce((a,b)=>a+b,0);
    const probabilities=exps.map(x=>x/total);
    expect(probabilities[0]).toBeCloseTo(0.245,3);
    expect(probabilities[0]).toBeLessThan(1/3);
    expect(Math.log(probabilities[0]/probabilities[2])).toBeCloseTo(1,12);
    expect(body).toContain('e/(e+e^2+1)');
    expect(body).toContain('при финальной');
  });
  it('includes full forward, sharding and backward instead of replacing manual derivations',()=>{
    expect(body.match(/~~~python\n/g)).toHaveLength(3);
    for(const s of ['def swiglu(','def swiglu_backward(','np.logaddexp','fused_projection',
      'split_y','flat_x.T @ flat_du','3,3,4','4000 префиксах','примерно ниже 68%',
      'логарифм перплексии','1.633','135 266 304','4 328 521 728','172 MiB','8320',
      '((z * silu(u)) @ wd)[0, 0]'])
      expect(body).toContain(s);
    expect(body).toContain('нет реального обмена');
  });
  it('preserves stable anchors and adjacent navigation',()=>{
    for(const id of ['gating-значение-и-пропуск-в-разных-проекциях','точный-parameter-accounting',
      'ffn-vocabulary-projection'])
      expect(body.match(new RegExp('id="'+id+'"','g'))).toHaveLength(1);
    expect(fm.previous).toContain('04 RoPE');
    expect(fm.next).toContain('05 Transformer с нуля');
    expect(body).toContain('**Назад:**');
    expect(body).toContain('**Дальше:**');
  });
  it('starts with two ReLU rows and states the SwiGLU outputs and gradient before relying on code',()=>{
    expect(body.indexOf('### Два токена через ReLU-FFN')).toBeLessThan(body.indexOf('### Размерности'));
    expect(body.indexOf('### Сложение с остаточной ветвью')).toBeLessThan(body.indexOf('## Две входные проекции SwiGLU'));
    for(const s of ['$(4,-3)$','(1.0305,-1.8961)','$D_{Z,1}=D_{H,1}$','математическое число параметров'])
      expect(body).toContain(s);
    for(const id of ['ffn-swiglu-code','ffn-shard-code','ffn-backward-code'])
      expect(body).toContain('<details id="'+id+'">');
  });
});
