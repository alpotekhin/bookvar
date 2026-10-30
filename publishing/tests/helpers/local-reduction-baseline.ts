import { withoutIndexedMovementAddition } from './indexed-movement-baseline.js';

export const reductionStart = '<!-- BEGIN REVIEW: local reduction tree 2026-10-27 -->';
export const reductionEnd = '<!-- END REVIEW: local reduction tree 2026-10-27 -->';
export const reductionSources = [
  'https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf',
  'https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/cpp/2_Concepts_and_Techniques/reduction/reduction_kernel.cu'
];

export function withoutLocalReductionAddition(text: string): string {
  text = withoutIndexedMovementAddition(text);
  const addition = text.split(reductionStart)[1].split(reductionEnd)[0];
  let baseline = text.replace(reductionStart + addition + reductionEnd + '\n\n', '');
  baseline = baseline.replace(
    String.raw`$$
\begin{aligned}
(&0{,}012755,\;0{,}034671,\\
 &0{,}256187,\;0{,}696387).
\end{aligned}
$$`,
    String.raw`$(0{,}012755,0{,}034671,0{,}256187,0{,}696387)$.`);
  for (const url of reductionSources) baseline = baseline.replace(`  - ${url}\n`, '');
  return baseline.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-26')
    .replace('last_verified: 2026-10-27', 'last_verified: 2026-10-26');
}
