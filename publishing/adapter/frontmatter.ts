import matter from 'gray-matter';

export interface ParsedPage {
  title: string;
  type: string;
  status: string;
  lastUpdated?: string;
  body: string;
  sourceFragments: SourceFragment[];
}

export interface SourceFragment {
  author: string;
  course: string;
  sourceUrl: string;
  license: string;
  language: 'en' | 'ru';
  transformation: 'original excerpt' | 'adaptation' | 'editorial bridge';
}

const FAST_CHANGING_TYPES = new Set([
  'model-family',
  'model-release',
  'research-line'
]);

export const PUBLICATION_PAGE_TYPES = new Set([
  'textbook-chapter',
  'concept',
  'model-family',
  'model-release',
  'research-line',
  'source-note',
  'external-resource',
  'question-index',
  'practice',
  'visual'
]);

function dateString(value: unknown): string | undefined {
  if (typeof value === 'string' && value.trim()) return value;
  if (value instanceof Date && !Number.isNaN(value.valueOf())) {
    return value.toISOString().slice(0, 10);
  }
  return undefined;
}

function sourceFragments(value: unknown, errors: string[]): SourceFragment[] {
  if (value === undefined) return [];
  const fragments = Array.isArray(value) ? value : [value];
  return fragments.flatMap((candidate, index) => {
    const label = fragments.length === 1 ? 'source_fragment' : `source_fragment[${index}]`;
    if (!candidate || typeof candidate !== 'object' || Array.isArray(candidate)) {
      errors.push(`${label} must be an object`);
      return [];
    }
    const record = candidate as Record<string, unknown>;
    const required = ['author', 'course', 'source_url', 'license', 'language', 'transformation'] as const;
    for (const field of required) {
      if (typeof record[field] !== 'string' || !record[field].trim()) {
        errors.push(`${label}.${field} must be a non-empty string`);
      }
    }
    if (typeof record.source_url === 'string'
      && !/\/blob\/[0-9a-f]{40}\//.test(record.source_url)) {
      errors.push(`${label}.source_url must pin a 40-character commit`);
    }
    if (typeof record.language === 'string' && !/^(en|ru)$/.test(record.language)) {
      errors.push(`${label}.language must be "en" or "ru"`);
    }
    if (typeof record.transformation === 'string'
      && !/^(original excerpt|adaptation|editorial bridge)$/.test(record.transformation)) {
      errors.push(`${label}.transformation must identify the permitted transformation`);
    }
    if (required.some((field) => typeof record[field] !== 'string' || !record[field].trim())) return [];
    if (!/\/blob\/[0-9a-f]{40}\//.test(record.source_url as string)
      || !/^(en|ru)$/.test(record.language as string)
      || !/^(original excerpt|adaptation|editorial bridge)$/.test(record.transformation as string)) return [];
    return [{
      author: (record.author as string).trim(),
      course: (record.course as string).trim(),
      sourceUrl: (record.source_url as string).trim(),
      license: (record.license as string).trim(),
      language: record.language as SourceFragment['language'],
      transformation: record.transformation as SourceFragment['transformation']
    }];
  });
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
  const fragments = sourceFragments(metadata.source_fragment, errors);

  if (type && !PUBLICATION_PAGE_TYPES.has(type)) {
    errors.push(`unsupported type "${type}"`);
  }

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
    body: parsed.content,
    sourceFragments: fragments
  };
}
