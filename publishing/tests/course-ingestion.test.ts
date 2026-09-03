import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { parse, stringify } from 'yaml';
import { afterEach, describe, expect, it } from 'vitest';
import { validateCourseLedger } from '../tools/course-ledger.ts';

const fixtureDirectory = resolve(import.meta.dirname, 'fixtures');
const temporaryDirectories: string[] = [];

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
  it('accepts a complete ledger with every coverage disposition', () => {
    validate(fixture('course-ledger.valid.yml'));
  });

  it('rejects the required malformed fixture', () => {
    expect(() => validate(fixture('course-ledger.invalid.yml'))).toThrow(/source_location|disposition|local_file/i);
  });

  const invalidMutations: Array<[string, (bundle: Fixture) => void, RegExp]> = [
    ['duplicate IDs', (bundle) => { (bundle.source_units.units as unknown[]).push(clone((bundle.source_units.units as unknown[])[0])); }, /duplicate.*id/i],
    ['unknown source objects', (bundle) => { ((bundle.coverage.rows as Array<Record<string, unknown>>)[0]).source_object = 'missing-object'; }, /unknown.*source object/i],
    ['unknown source units', (bundle) => { ((bundle.coverage.rows as Array<Record<string, unknown>>)[0]).source_unit = 'missing-unit'; }, /unknown.*source unit/i],
    ['unpinned GitHub URLs', (bundle) => { ((bundle.source_manifest.objects as Array<Record<string, unknown>>)[0]).canonical_url = 'https://github.com/example/course/blob/main/lecture.md'; }, /pinned.*github/i],
    ['extracted units absent from coverage', (bundle) => { (bundle.coverage.rows as unknown[]).pop(); }, /missing.*coverage/i],
    ['integrated units without a destination anchor', (bundle) => { delete ((bundle.coverage.rows as Array<Record<string, unknown>>)[0]).destination_anchor; }, /destination_anchor/i],
    ['missing reciprocal source unit IDs', (bundle) => { bundle.pages!['textbook/foundations.md'] = '# Foundations'; }, /reciprocal.*source_unit_id/i],
    ['source-only rows outside the source hub', (bundle) => { ((bundle.coverage.rows as Array<Record<string, unknown>>)[2]).destination = 'textbook/foundations.md'; }, /source hub/i],
    ['exclusions without reasons and evidence', (bundle) => { const row = (bundle.coverage.rows as Array<Record<string, unknown>>)[3]; delete row.reason; delete row.evidence; }, /reason.*evidence/i],
    ['missing mirrored source files', (bundle) => { bundle.files = bundle.files?.filter((path) => path !== 'courses/fixture/Lectures/lecture-01.md'); }, /missing.*source file/i],
    ['unverified official artifacts', (bundle) => { ((bundle.source_manifest.objects as Array<Record<string, unknown>>)[0]).verification_status = 'unverified'; }, /official.*verified/i],
    ['reused visuals without attribution and rights evidence', (bundle) => { const visual = (bundle.visuals.rows as Array<Record<string, unknown>>)[0]; delete visual.attribution; delete visual.rights_evidence; }, /attribution.*rights_evidence|rights_evidence.*attribution/i],
    ['unordered visual sequences', (bundle) => { const visual = (bundle.visuals.rows as Array<Record<string, unknown>>)[0]; visual.sequence_members = [{ order: 2, source_location: 'two' }, { order: 1, source_location: 'one' }]; }, /sequence_members.*ordered/i],
    ['unregistered local images', (bundle) => { (bundle.asset_registry.assets as unknown[]).pop(); }, /asset registry/i]
  ];

  it.each(invalidMutations)('rejects %s', (_label, mutate, error) => {
    const bundle = clone(fixture('course-ledger.valid.yml'));
    mutate(bundle);
    expect(() => validate(bundle)).toThrow(error);
  });
});
