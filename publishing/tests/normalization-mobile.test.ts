import { beforeReaderRouteNavigation, readBeforeReaderRouteNavigation } from '../tools/reader-route-baseline.mjs';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {describe,expect,it} from 'vitest';

const root=resolve(import.meta.dirname,'../..');
const cases=[
  {
    "file": "00 Учебник/07 Анатомия современной LLM/03 Pre-norm, RMSNorm, SwiGLU и residual.md",
    "date": "2026-10-22",
    "hash": "79047e85222742283c60fbbb415226cd71b816565aaaf075b6e446d6ad800677",
    "pairs": [
      [
        "J_y=I+J_F,\\qquad\n\\nabla_x\\mathcal L=(I+J_F)^\\top\\nabla_y\\mathcal L.",
        "\\begin{aligned}\nJ_y&=I+J_F,\\\\\n\\nabla_x\\mathcal L&=(I+J_F)^\\top\\nabla_y\\mathcal L.\n\\end{aligned}"
      ],
      [
        "\\mu=\\frac1d\\sum_i x_i,\\qquad\n\\sigma^2=\\frac1d\\sum_i(x_i-\\mu)^2,",
        "\\begin{aligned}\n\\mu&=\\frac1d\\sum_i x_i,\\\\\n\\sigma^2&=\\frac1d\\sum_i(x_i-\\mu)^2.\n\\end{aligned}"
      ],
      [
        "\\mathcal L_z=\\lambda(\\log Z)^2,\\qquad\n\\log Z=\\log\\sum_j e^{z_j}.",
        "\\begin{aligned}\n\\mathcal L_z&=\\lambda(\\log Z)^2,\\\\\n\\log Z&=\\log\\sum_j e^{z_j}.\n\\end{aligned}"
      ]
    ]
  },
  {
    "file": "00 Учебник/09 Dense FFN и Mixture of Experts/01 Dense FFN — token-wise вычисление, expansion и gating.md",
    "date": "2026-10-23",
    "hash": "cf6e628795c428a7e96332194639f3f37e6f91699829528d5a1e49d8826e5b11",
    "pairs": [
      [
        "\\operatorname{SiLU}(z)=z\\,\\sigma(z),\n\\qquad \\sigma(z)=\\frac{1}{1+e^{-z}}.",
        "\\begin{aligned}\n\\operatorname{SiLU}(z)&=z\\,\\sigma(z),\\\\\n\\sigma(z)&=\\frac{1}{1+e^{-z}}.\n\\end{aligned}"
      ]
    ]
  }
];
const hash=(text:string)=>createHash('sha256').update(text).digest('hex');
describe('narrow-screen normalization and SwiGLU formula repairs',()=>{
  for(const item of cases) it('preserves accepted explanation: '+item.file,()=>{
    const chapter=readFileSync(resolve(root,item.file),'utf8');
    let restored=beforeReaderRouteNavigation(chapter, item.file);
    for(const [before,after] of item.pairs) {
      expect(restored).toContain(after);
      restored=restored.replace(after,before);
    }
    restored=restored.replace('last_updated: 2026-10-27','last_updated: '+item.date)
      .replace('last_verified: 2026-10-27','last_verified: '+item.date);
    expect(hash(restored)).toBe(item.hash);
  });
});
