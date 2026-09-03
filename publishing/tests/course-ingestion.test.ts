import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { parse, stringify } from 'yaml';
import { afterEach, describe, expect, it } from 'vitest';
import { validateCourseLedger } from '../tools/course-ledger.ts';

const fixtureDirectory = resolve(import.meta.dirname, 'fixtures');
const temporaryDirectories: string[] = [];

type LedgerRow = Record<string, unknown>;

type Fixture = Record<string, unknown> & {
  source_manifest: Record<string, unknown>;
  source_units: Record<string, unknown>;
  coverage: Record<string, unknown>;
  visuals: Record<string, unknown>;
  asset_registry: Record<string, unknown>;
  files?: string[];
  pages?: Record<string, string>;
};

function fixture(name: string): Fixture {
  return parse(readFileSync(resolve(fixtureDirectory, name), 'utf8')) as Fixture;
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function rows(document: Record<string, unknown>, key: string): LedgerRow[] {
  return document[key] as LedgerRow[];
}

function rowById(document: Record<string, unknown>, key: string, id: string): LedgerRow {
  const result = rows(document, key).find((entry) => entry.id === id);
  if (!result) throw new Error('Fixture row not found: ' + id);
  return result;
}

function object(bundle: Fixture, id: string): LedgerRow {
  return rowById(bundle.source_manifest, 'objects', id);
}

function unit(bundle: Fixture, id: string): LedgerRow {
  return rowById(bundle.source_units, 'units', id);
}

function coverageRow(bundle: Fixture, id: string): LedgerRow {
  return rowById(bundle.coverage, 'rows', id);
}

function visual(bundle: Fixture, id: string): LedgerRow {
  return rowById(bundle.visuals, 'rows', id);
}

function materialize(bundle: Fixture): { courseRoot: string; repositoryRoot: string } {
  const repositoryRoot = mkdtempSync(resolve(tmpdir(), 'course-ledger-'));
  temporaryDirectories.push(repositoryRoot);
  const courseRoot = resolve(repositoryRoot, 'courses/fixture');
  mkdirSync(courseRoot, { recursive: true });
  const documents: Array<[string, unknown]> = [
    ['source-manifest.yml', bundle.source_manifest],
    ['source-units.yml', bundle.source_units],
    ['coverage.yml', bundle.coverage],
    ['visuals.yml', bundle.visuals]
  ];
  for (const [name, document] of documents) writeFileSync(resolve(courseRoot, name), stringify(document));
  writeFileSync(resolve(repositoryRoot, 'asset-registry.yml'), stringify(bundle.asset_registry));
  for (const relativePath of bundle.files ?? []) {
    const target = resolve(repositoryRoot, relativePath);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, bundle.pages?.[relativePath] ?? 'fixture source');
  }
  return { courseRoot, repositoryRoot };
}

function validate(bundle: Fixture): void {
  const { courseRoot, repositoryRoot } = materialize(bundle);
  validateCourseLedger(courseRoot, { repositoryRoot, assetRegistryPath: 'asset-registry.yml' });
}

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

describe('course ingestion ledger', () => {
  it('accepts explicit lecture, PDF, assignment, and video unit schemas with every coverage disposition', () => {
    validate(fixture('course-ledger.valid.yml'));
  });

  it('accepts a raw GitHub content URL pinned to the recorded commit', () => {
    const bundle = fixture('course-ledger.valid.yml');
    object(bundle, 'lecture-01').canonical_url =
      'https://raw.githubusercontent.com/example/course/0123456789abcdef0123456789abcdef01234567/lecture.md';
    validate(bundle);
  });

  it('rejects the required malformed fixture', () => {
    expect(() => validate(fixture('course-ledger.invalid.yml'))).toThrow(/source_location/i);
  });

  const invalidMutations: Array<[string, (bundle: Fixture) => void, RegExp]> = [
    [
      'duplicate IDs',
      (bundle) => { rows(bundle.source_units, 'units').push(clone(rows(bundle.source_units, 'units')[0])); },
      /duplicate.*id/i
    ],
    [
      'unknown source objects',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').source_object = 'missing-object'; },
      /unknown.*source object/i
    ],
    [
      'a known source object that does not own the referenced unit',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').source_object = 'assignment-01'; },
      /source_object.*does not match/i
    ],
    [
      'unknown source units',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').source_unit = 'missing-unit'; },
      /unknown.*source unit/i
    ],
    [
      'coverage source locations that disagree with extraction metadata',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').source_location = 'Lectures/other.md'; },
      /source_location.*does not match/i
    ],
    [
      'coverage kinds that disagree with extraction metadata',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').kind = 'rendered-text'; },
      /kind.*does not match/i
    ],
    [
      'coverage titles that disagree with extraction metadata',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').title = 'Other title'; },
      /title.*does not match/i
    ],
    [
      'unpinned GitHub URLs',
      (bundle) => { object(bundle, 'lecture-01').canonical_url = 'https://github.com/example/course/blob/main/lecture.md'; },
      /pinned GitHub SHA/i
    ],
    [
      'unpinned raw GitHub content URLs',
      (bundle) => { object(bundle, 'lecture-01').canonical_url = 'https://raw.githubusercontent.com/example/course/main/lecture.md'; },
      /pinned GitHub SHA/i
    ],
    [
      'extracted units absent from coverage',
      (bundle) => { rows(bundle.coverage, 'rows').pop(); },
      /missing.*coverage/i
    ],
    [
      'unsupported source-unit kinds',
      (bundle) => { unit(bundle, 'lecture-01-section').kind = 'anything'; },
      /kind.*allowed value/i
    ],
    [
      'source units missing their kind-specific schema fields',
      (bundle) => { delete unit(bundle, 'lecture-01-section').heading_level; },
      /heading_level/i
    ],
    [
      'source units with non-contiguous event order',
      (bundle) => { unit(bundle, 'pdf-01-page').order = 2; },
      /order must be 1/i
    ],
    [
      'videos with empty metadata',
      (bundle) => { object(bundle, 'video-01').video_metadata = {}; },
      /video_metadata\.duration_seconds/i
    ],
    [
      'integrated units without a destination anchor',
      (bundle) => { delete coverageRow(bundle, 'coverage-integrated').destination_anchor; },
      /destination_anchor/i
    ],
    [
      'integrated units whose destination anchor does not resolve',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').destination_anchor = 'missing-anchor'; },
      /destination_anchor.*does not exist/i
    ],
    [
      'missing reciprocal source unit IDs',
      (bundle) => { bundle.pages!['textbook/foundations.md'] = '# Foundations\n\n## Foundations figure'; },
      /reciprocal.*source_unit_id/i
    ],
    [
      'source unit IDs mentioned only in the Markdown body',
      (bundle) => {
        bundle.pages!['textbook/foundations.md'] =
          '# Foundations\n\nsource_unit_id: lecture-01-section\n\n## Foundations figure';
      },
      /reciprocal.*source_unit_id/i
    ],
    [
      'source-only rows outside the source hub',
      (bundle) => {
        const row = coverageRow(bundle, 'coverage-source-only');
        row.destination = 'textbook/foundations.md';
        row.destination_anchor = 'foundations';
      },
      /source hub/i
    ],
    [
      'source-only rows whose source-hub anchor does not resolve',
      (bundle) => { coverageRow(bundle, 'coverage-source-only').destination_anchor = 'missing-anchor'; },
      /destination_anchor.*does not exist/i
    ],
    [
      'exclusions without reasons and evidence',
      (bundle) => {
        const row = coverageRow(bundle, 'coverage-excluded');
        delete row.reason;
        delete row.evidence;
      },
      /reason.*evidence/i
    ],
    [
      'missing mirrored source files',
      (bundle) => { bundle.files = bundle.files?.filter((path) => path !== 'courses/fixture/Lectures/lecture-01.md'); },
      /missing.*source file/i
    ],
    [
      'unverified official artifacts',
      (bundle) => { object(bundle, 'lecture-01').verification_status = 'unverified'; },
      /official.*verified/i
    ],
    [
      'reused visuals without attribution and rights evidence',
      (bundle) => {
        const row = visual(bundle, 'visual-01');
        delete row.attribution;
        delete row.rights_evidence;
      },
      /attribution|rights_evidence/i
    ],
    [
      'non-reused visuals without attribution',
      (bundle) => { delete visual(bundle, 'visual-source-only').attribution; },
      /attribution/i
    ],
    [
      'non-reused visuals without rights evidence',
      (bundle) => { delete visual(bundle, 'visual-source-only').rights_evidence; },
      /rights_evidence/i
    ],
    [
      'unordered visual sequences',
      (bundle) => {
        visual(bundle, 'visual-01').sequence_members = [
          { order: 2, source_location: 'two' },
          { order: 1, source_location: 'one' }
        ];
      },
      /sequence_members.*ordered/i
    ],
    [
      'empty visual sequences',
      (bundle) => { visual(bundle, 'visual-01').sequence_members = []; },
      /sequence_members.*must not be empty/i
    ],
    [
      'visuals without exact pages or frames',
      (bundle) => { delete visual(bundle, 'visual-01').source_pages; },
      /source_pages.*list/i
    ],
    [
      'integrated visuals whose destination anchor does not resolve',
      (bundle) => { visual(bundle, 'visual-01').destination_anchor = 'missing-anchor'; },
      /destination_anchor.*does not exist/i
    ],
    [
      'unregistered local images',
      (bundle) => { rows(bundle.asset_registry, 'assets').pop(); },
      /asset registry/i
    ],
    [
      'a missing source location in isolation',
      (bundle) => { delete unit(bundle, 'lecture-01-section').source_location; },
      /source_location/i
    ],
    [
      'multiple dispositions in isolation',
      (bundle) => { coverageRow(bundle, 'coverage-integrated').disposition = ['integrated', 'source-only']; },
      /disposition/i
    ],
    [
      'an integrated visual without a local file in isolation',
      (bundle) => { delete visual(bundle, 'visual-01').local_file; },
      /local_file/i
    ]
  ];

  it.each(invalidMutations)('rejects %s', (_label, mutate, error) => {
    const bundle = clone(fixture('course-ledger.valid.yml'));
    mutate(bundle);
    expect(() => validate(bundle)).toThrow(error);
  });
});
