export const tensorStorageStart = '<!-- BEGIN REVIEW: tensor storage 2026-10-27 -->';
export const tensorStorageEnd = '<!-- END REVIEW: tensor storage 2026-10-27 -->';

export function withoutTensorStorageAddition(text: string): string {
  if (!text.includes(tensorStorageStart)) return text;
  const addition = text.split(tensorStorageStart)[1].split(tensorStorageEnd)[0];
  return text.replace(tensorStorageStart + addition + tensorStorageEnd + '\n\n', '');
}
