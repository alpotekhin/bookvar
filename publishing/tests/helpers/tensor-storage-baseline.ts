import { beforeTopicConsolidation } from '../../tools/topic-consolidation-baseline.mjs';

export const tensorStorageStart = '<!-- BEGIN REVIEW: tensor storage 2026-10-27 -->';
export const tensorStorageEnd = '<!-- END REVIEW: tensor storage 2026-10-27 -->';

export function withoutTensorStorageAddition(text: string): string {
  text = beforeTopicConsolidation(text,
    '00 Учебник/00 Математические и ML-основания/01 Векторы, матрицы и тензоры.md',
    undefined, '2026-10-30-clean-checkout-additions.json');
  if (!text.includes(tensorStorageStart)) return text;
  const addition = text.split(tensorStorageStart)[1].split(tensorStorageEnd)[0];
  return text.replace(tensorStorageStart + addition + tensorStorageEnd + '\n\n', '');
}
