export const tracingStart='<!-- BEGIN REVIEW: tracing control flow 2026-10-27 -->';
export const tracingEnd='<!-- END REVIEW: tracing control flow 2026-10-27 -->';

export function withoutTracingControlAddition(text:string):string {
  text=withoutJaxRuntimeAddition(text);
  if(!text.includes(tracingStart)) return text;
  const addition=text.split(tracingStart)[1].split(tracingEnd)[0];
  return text.replace(tracingStart+addition+tracingEnd+'\n\n','');
}
import { withoutJaxRuntimeAddition } from './jax-runtime-baseline.js';

