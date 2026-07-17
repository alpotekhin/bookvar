import matter from 'gray-matter';

export interface ParsedPage {
  title: string;
  type: string;
  status: string;
  lastUpdated?: string;
  body: string;
}

const FAST_CHANGING_TYPES = new Set([
  'model-family',
  'model-release',
  'research-line'
]);

function dateString(value: unknown): string | undefined {
  if (typeof value === 'string' && value.trim()) return value;
  if (value instanceof Date && !Number.isNaN(value.valueOf())) {
    return value.toISOString().slice(0, 10);
  }
  return undefined;
}

export function readPage(sourcePath: string, raw: string): ParsedPage {
  const parsed = matter(raw);
  const metadata: Readonly<Record<string, unknown>> = parsed.data;
  const errors: string[] = [];

  for (const field of ['title', 'type', 'status'] as const) {
    if (typeof metadata[field] !== 'string' || !metadata[field].trim()) {
      errors.push(`missing required field "${field}"`);
    }
  }

  const type = typeof metadata.type === 'string' ? metadata.type : '';
  const lastVerified = dateString(metadata.last_verified);
  const lastUpdated = dateString(metadata.last_updated);

  if (FAST_CHANGING_TYPES.has(type) && !lastVerified) {
    errors.push(`missing required field "last_verified" for type "${type}"`);
  }

  if (errors.length > 0) {
    throw new Error([
      'Publication metadata errors:',
      ...errors.map((error) => `- ${sourcePath}: ${error}`)
    ].join('\n'));
  }

  return {
    title: metadata.title as string,
    type,
    status: metadata.status as string,
    lastUpdated: lastVerified ?? lastUpdated,
    body: parsed.content
  };
}
