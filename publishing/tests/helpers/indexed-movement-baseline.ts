import { withoutTracingControlAddition } from './tracing-control-baseline.js';

export const indexedStart = '<!-- BEGIN REVIEW: indexed movement 2026-10-27 -->';
export const indexedEnd = '<!-- END REVIEW: indexed movement 2026-10-27 -->';
export const indexedSources = [
  'https://github.com/rusty1s/pytorch_scatter/blob/f514c10f920b5aeed2eb162092f0ad20d3edee52/torch_scatter/scatter.py',
  'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_torch_docs.py',
  'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/_tensor_docs.py',
  'https://github.com/onnx/onnx/blob/3646b038c23824207d064d8625a908b1bf8710f4/onnx/defs/doc_strings.cc'
];

export function withoutIndexedMovementAddition(text: string): string {
  text = withoutTracingControlAddition(text);
  if (!text.includes(indexedStart)) return text;
  const addition = text.split(indexedStart)[1].split(indexedEnd)[0];
  let baseline = text.replace(indexedStart + addition + indexedEnd + '\n\n', '');
  for (const url of indexedSources) baseline = baseline.replace(`  - ${url}\n`, '');
  return baseline;
}
