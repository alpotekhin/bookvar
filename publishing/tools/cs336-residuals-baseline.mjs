// Reverse only the bounded 29 October CS336 additions for historical audits.
// Current content and assets receive separate validation; no whitespace cleanup.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
const defaultRoot = resolve(import.meta.dirname, '../..');
const hash = text => createHash('sha256').update(text).digest('hex');

export function beforeCs336Residuals(text, source, root = defaultRoot) {
  const ledger = JSON.parse(readFileSync(resolve(root,
    'docs/audit/2026-10-29-cs336-residual-additions.json'), 'utf8'));
  const row = ledger.files.find(file => file.source === source);
  if (!row || hash(text) === row.before_sha256) return text;
  for (const { scope, sha256 } of row.blocks) {
    const pattern = new RegExp(`<!-- BEGIN REVIEW: ${scope} 2026-10-29 -->[\\s\\S]*?<!-- END REVIEW: ${scope} 2026-10-29 -->\\n\\n`, 'g');
    const matches = [...text.matchAll(pattern)];
    if (matches.length !== 1 || hash(matches[0][0]) !== sha256)
      throw new Error(`Unreviewed CS336 residual addition: ${source}: ${scope}`);
    text = text.replace(matches[0][0], '');
  }
  if (source === '02 Атлас моделей/Семейства/DeepSeek.md') {
    text = text.replace('latest_verified_release: DeepSeek-V4 (2026-04-24)\nlast_verified: 2026-10-29',
      'latest_verified_release: DeepSeek-V3.2\nlast_verified: 2026-08-06');
    text = text.replace('В поколениях V2–V3.2 модель использует причинный декодер с MLA и MoE из небольших',
      'Начиная с V2, модель использует причинный декодер с MLA и MoE из небольших');
    text = text.replace('| DeepSeek-V4 Pro / Flash (2026-04-24) | CSA/HCA вместо MLA; shared-KV MQA; mHC; MoE | 1 048 576 позиций в опубликованной конфигурации Pro | Muon и AdamW для разных параметров; SFT/RL специалистов и on-policy distillation | **A** |\n', '');
  }
  return text;
}
