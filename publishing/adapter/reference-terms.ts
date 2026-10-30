export interface ReferenceTerm {
  label: string;
  label_en: string;
  anchor: string;
}

export function referenceTerms(source: string, input: unknown,
  headings: Array<{ explicitAnchor?: string }>): ReferenceTerm[] {
  if (input === undefined) return [];
  if (!Array.isArray(input)) throw new Error('reference_terms must be a list: ' + source);
  const labels = new Set<string>();
  return input.map(candidate => {
    const term = candidate as Partial<ReferenceTerm> | null;
    if (!term || typeof term.label !== 'string' || !term.label.trim()
      || typeof term.label_en !== 'string' || !term.label_en.trim()
      || /[\r\n|\[\]]/.test(term.label + term.label_en)
      || typeof term.anchor !== 'string' || !/^[a-z0-9][a-z0-9-]*$/.test(term.anchor)
      || headings.filter(heading => heading.explicitAnchor === term.anchor).length !== 1
      || labels.has(term.label.trim()))
      throw new Error('Invalid or missing reference term anchor: ' + source);
    labels.add(term.label.trim());
    return { label: term.label.trim(), label_en: term.label_en.trim(), anchor: term.anchor };
  });
}
