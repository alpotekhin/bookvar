import { beforeReaderRouteNavigation } from '../../tools/reader-route-baseline.mjs';

export const nativeStart = '<!-- BEGIN REVIEW: Harvard im2row 2026-10-27 -->';
export const nativeEnd = '<!-- END REVIEW: Harvard im2row 2026-10-27 -->';

export function withoutWinogradExample(text: string): string {
  text = beforeReaderRouteNavigation(text, '00 Учебник/01 Основы нейронных сетей/07 CNN — от свёртки до ResNet.md');
  return text.replace(/<!-- BEGIN REVIEW: Winograd 2026-10-27 -->\n[\s\S]*?<!-- END REVIEW: Winograd 2026-10-27 -->\n\n/, '');
}

export function withoutNativeWindowExample(text: string): string {
  text = withoutWinogradExample(text);
  const addition = text.split(nativeStart)[1].split(nativeEnd)[0];
  return text.replace(nativeStart + addition + nativeEnd + '\n\n', '')
    .replace('last_updated: 2026-10-27', 'last_updated: 2026-10-26');
}
