import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parse} from 'yaml';
import {describe, expect, it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path),'utf8');
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const chapter='00 Учебник/01 Основы нейронных сетей/09 GAN и DCGAN — генератор и дискриминатор.md';
const edge='01 Основы нейронных сетей/09 GAN и DCGAN — генератор и дискриминатор';
const packet='05 Источники/Originals/gan-dcgan-2026-10-28/d2l/';

describe('source-first GAN/DCGAN reader',()=>{
  it('retains eight complete source bodies rather than replacing them with API fragments',()=>{
    const text=read(chapter);
    expect(text.match(/^```python$/gm)).toHaveLength(14);
    expect(text.match(/^!\[\[/gm)).toHaveLength(5);
    function blocks(name:string){
      return [...read(packet+name).matchAll(/```\{\.python \.input\}\n([\s\S]*?)\n```/g)]
        .map(m=>m[1]).filter(b=>/^#@tab [^\n]*(?:pytorch|all)/.test(b))
        .map(b=>b.replace(/^#@(?:tab|save)[^\n]*\n/gm,''));
    }
    const gan=blocks('gan.md'),dcgan=blocks('dcgan.md');
    for(const source of [gan[4],gan[5],gan[6],gan[7],dcgan[4],dcgan[7],dcgan[10],dcgan[12]])
      expect(text).toContain('\n```python\n'+source.trim()+'\n```');
    expect(text).toContain('source_language: en');
    expect(text).not.toMatch(/:numref:|:cite:|```(?:mermaid|svg)/);
  });
  it('preserves explicit source corrections and execution limits',()=>{
    const text=read(chapter).replace(/\s+/g,' ');
    for(const limit of ['generator\'s minimax gradient when D successfully rejects',
      'not guarantee convergence','does not identify A uniquely',
      'not a plot produced by the local loop','covariance still differs',
      'includes BatchNorm in its first discriminator block',
      'three BatchNorm visits','not an image-quality demonstration',
      'distinct recipes','unconditional']) expect(text).toContain(limit);
  });
  it('historical explanation preservation: inserts the chapter between VAE and text representations without rewriting either body',()=>{
    const nav=readBeforeReaderRouteNavigation('publishing/navigation.yml');
    expect(nav.match(/route: textbook\/neural-networks\/gan-dcgan$/gm)).toHaveLength(2);
    expect(nav).toContain('source: '+chapter);
    expect(readBeforeReaderRouteNavigation('00 Учебник/_index.md')).toContain(edge);
    const previous=readBeforeReaderRouteNavigation('00 Учебник/01 Основы нейронных сетей/08 Autoencoder и VAE.md');
    expect(hash(previous.replace(
      '[[00 Учебник/'+edge+'|Следующая глава]]\nразбирает обучение генератора с помощью дискриминатора.',
      '[[00 Учебник/02 Представление текста и токенизация/01 Представление текста числами|Следующая глава]]\nпереходит от представлений изображений к числовому представлению текста.')))
      .toBe('e742c3a56ecec3b2f7d768e017db822b54979f66dc273e57b0b0f968f12607f7');
    const next=readBeforeReaderRouteNavigation('00 Учебник/02 Представление текста и токенизация/01 Представление текста числами.md');
    expect(hash(next.replaceAll(edge,'01 Основы нейронных сетей/08 Autoencoder и VAE')
      .replace('|GAN и DCGAN — генератор и дискриминатор]]','|Autoencoder и VAE]]')))
      .toBe('8b2de4a9b960721f5ecf8f95aad23f2e3edbb9883b5c7d83bd25003fd387ddb0');
  });
  it('keeps all source image bytes, source snapshots and license evidence',()=>{
    const prefix='00 Учебник/Assets/Figures/curated/gan-dcgan-2026-10-28/';
    const rows=JSON.parse(read(prefix+'assets.json'));
    const registry=parse(read('05 Источники/asset-registry.yml')).assets;
    expect(rows).toHaveLength(4);
    for(const row of rows){
      expect(registry.find((r:any)=>r.id===row.id)).toEqual(row);
      expect(row.used_in).toEqual([chapter]);
      const bytes=readFileSync(resolve(root,row.asset));
      expect(bytes).toEqual(readFileSync(resolve(root,row.source_asset)));
      expect(hash(bytes)).toBe(row.sha256);
      expect(hash(readFileSync(resolve(root,row.source_page_snapshot)))).toBe(row.source_page_sha256);
      expect(hash(readFileSync(resolve(root,row.license_evidence)))).toBe(row.license_evidence_sha256);
      expect(row.license).toBe('CC BY-SA 4.0');
    }
    const gif=registry.find((r:any)=>r.id==='curated-ae-transpose-convolution-9b41abceb5b1');
    expect(gif.used_in).toContain(chapter);
    expect(gif.frames).toBe(25);
  });
});
