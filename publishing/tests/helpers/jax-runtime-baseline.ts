export const jaxRuntimeStart = '<!-- BEGIN REVIEW: JAX runtime 2026-10-27 -->';
export const jaxRuntimeEnd = '<!-- END REVIEW: JAX runtime 2026-10-27 -->';

export function withoutJaxRuntimeAddition(text: string): string {
  text = beforeReaderRouteNavigation(text,
    '00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md');
  if (!text.includes(jaxRuntimeStart)) return text;
  const addition = text.split(jaxRuntimeStart)[1].split(jaxRuntimeEnd)[0];
  return text.replace(jaxRuntimeStart + addition + jaxRuntimeEnd + '\n\n', '');
}
import { beforeReaderRouteNavigation } from '../../tools/reader-route-baseline.mjs';

