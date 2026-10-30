import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';
import {withoutWinogradExample} from './helpers/nn-native-baseline.js';

const root=resolve(import.meta.dirname,'../..');
const read=(path:string)=>readFileSync(resolve(root,path));
const hash=(data:Buffer|string)=>createHash('sha256').update(data).digest('hex');
const source='05 Источники/Originals/winograd-2026-10-27/';
const chapter=read('00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md').toString();

describe('source-grounded Winograd worked explanation',()=>{
  it('preserves the whole accepted CNN chapter outside the bounded addition',()=>{
    expect(hash(withoutWinogradExample(chapter)))
      .toBe('faf4455eb24ef822f831521f1549b44f9f5fc5770a74c69f9b3f61f068bd66da');
    const addition=chapter.split('<!-- BEGIN REVIEW: Winograd 2026-10-27 -->')[1]
      .split('<!-- END REVIEW: Winograd 2026-10-27 -->')[0];
    expect(addition.match(/^```python$/gm)).toHaveLength(4);
    for(const mechanism of ['Four products recover two outputs', 'A two-dimensional tile',
      'Channels become sixteen matrix multiplications', 'Transform cost, storage and rounding',
      'torch.bmm', 'ho % 2', 'not a measured speedup', 'input sizes', '0.5 instead of 1'])
      expect(addition).toContain(mechanism);
  });

  it('retains the exact licensed author code and English excerpt',()=>{
    expect(hash(read(source+'wincnn.py')))
      .toBe('144e2c2d0ab7b81241e4f267fa67ab55f141b0cf76bb7d697f4087c3e262c317');
    expect(read('site/public/examples/wincnn-c2bf6305.py')).toEqual(read(source+'wincnn.py'));
    expect(read('site/public/licenses/wincnn-c2bf6305-Apache-2.0.txt')).toEqual(read(source+'wincnn-LICENSE'));
    const quote=chapter.split('<!-- source-excerpt: wincnn-f23-verification -->\n')[1]
      .split('\n<!-- /source-excerpt -->')[0].split('\n').map(line=>line.replace(/^> ?/,'')).join('\n');
    expect(read(source+'wincnn-README.md').toString().replace(/\s+/g,' '))
      .toContain(quote.replace(/\s+/g,' '));
  });

  it('distinguishes a complete native PDF figure from a byte-copy or a redraw',()=>{
    const [figure]=JSON.parse(read('00 Учебник/Assets/Figures/curated/winograd-2026-10-27/assets.json').toString());
    expect(figure.derivation).toBe('source-pdf-render');
    expect(figure.dimensions_px).toEqual([1295,1278]);
    expect(hash(read(figure.asset))).toBe(figure.sha256);
    expect(hash(read(figure.source_asset))).toBe(figure.source_asset_sha256);
    expect(figure.license).toBe('CC BY 4.0');
    const html=read(figure.license_evidence).toString();
    expect(html).toContain('Creative Commons Attribution 4.0 International License');
    expect(chapter).toContain('Workflow for the Winograd algorithm.');
    expect(chapter).toContain('labels are\nretained');
  });
});
