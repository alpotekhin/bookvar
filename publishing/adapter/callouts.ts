const CALLOUT_TYPES: Readonly<Record<string, string>> = {
  note: 'note',
  info: 'note',
  abstract: 'note',
  tip: 'tip',
  warning: 'caution',
  caution: 'caution',
  danger: 'danger',
  failure: 'danger'
};

export function convertCallouts(markdown: string): string {
  const lines = markdown.split('\n');
  const converted: string[] = [];

  for (let index = 0; index < lines.length; index += 1) {
    const header = lines[index].match(/^\s*>\s*\[!([^\]]+)\](?:\s+(.*))?\s*$/i);
    const directiveType = header && CALLOUT_TYPES[header[1].toLowerCase()];

    if (!header || !directiveType) {
      converted.push(lines[index]);
      continue;
    }

    const title = header[2]?.trim();
    converted.push(`:::${directiveType}${title ? `[${title}]` : ''}`);

    while (index + 1 < lines.length) {
      const content = lines[index + 1].match(/^\s*> ?(.*)$/);
      if (!content) break;
      converted.push(content[1]);
      index += 1;
    }

    converted.push(':::');
  }

  return converted.join('\n');
}
