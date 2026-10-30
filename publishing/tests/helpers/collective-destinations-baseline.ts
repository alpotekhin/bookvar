export const collectiveStart = '<!-- BEGIN REVIEW: collective destinations 2026-10-27 -->';
export const collectiveEnd = '<!-- END REVIEW: collective destinations 2026-10-27 -->';
export const collectiveSources = [
  'https://github.com/pytorch/tutorials/blob/c8ccbbf7fcb8cb71adda1b957bd6bab50e29c2e6/intermediate_source/dist_tuto.rst',
  'https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/distributed/distributed_c10d.py'
];

export function withoutCollectiveDestinations(text: string): string {
  const addition = text.split(collectiveStart)[1].split(collectiveEnd)[0];
  let baseline = text.replace(collectiveStart + addition + collectiveEnd + '\n\n', '');
  for (const url of collectiveSources) baseline = baseline.replace(`  - ${url}\n`, '');
  return baseline.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-26')
    .replace('last_verified: 2026-10-27', 'last_verified: 2026-10-26');
}
