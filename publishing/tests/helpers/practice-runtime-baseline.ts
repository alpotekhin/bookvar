/** Remove only the bounded 29 October addition for historical preservation tests. */
export function beforeTextCheckpointRuntime(text: string): string {
  return text
    .replace(/<!-- BEGIN REVIEW: text checkpoint runtime 2026-10-29 -->[\s\S]*?<!-- END REVIEW: text checkpoint runtime 2026-10-29 -->\n\n/, '')
    .replace('last_updated: 2026-10-29', 'last_updated: 2026-10-26')
    .replace('last_verified: 2026-10-29', 'last_verified: 2026-10-26');
}
