import { existsSync, readFileSync } from 'node:fs';
import { relative, resolve, sep } from 'node:path';
import { parse } from 'yaml';

type Row = Record<string, unknown>;

export type CourseLedgerOptions = {
  repositoryRoot?: string;
  assetRegistryPath?: string;
};

const dispositions = new Set(['integrated', 'covered-existing', 'source-only', 'excluded']);
const authorities = new Set(['official-course', 'official-author', 'primary-paper', 'third-party-mirror', 'bookvar-original']);
const rights = new Set(['licensed', 'permission-recorded', 'link-only', 'unknown']);

function fail(message: string): never {
  throw new Error(`Course ledger: ${message}`);
}

function isRow(value: unknown): value is Row {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function row(value: unknown, label: string): Row {
  if (!isRow(value)) fail(`${label} must be a mapping`);
  return value;
}

function list(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) fail(`${label} must be a list`);
  return value;
}

function text(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') fail(`${label} must be a non-empty string`);
  return value.trim();
}

function field(value: Row, name: string, label: string): string {
  return text(value[name], `${label}.${name}`);
}

function enumField(value: Row, name: string, values: Set<string>, label: string): string {
  const result = field(value, name, label);
  if (!values.has(result)) fail(`${label}.${name} is not an allowed value`);
  return result;
}

function date(value: Row, name: string, label: string): string {
  const result = field(value, name, label);
  if (!/^\d{4}-\d{2}-\d{2}(?:T[^\s]+)?$/.test(result)) fail(`${label}.${name} must be an ISO date or timestamp`);
  return result;
}

function under(root: string, relativePath: string, label: string): string {
  const target = resolve(root, relativePath);
  const pathFromRoot = relative(root, target);
  if (pathFromRoot === '..' || pathFromRoot.startsWith(`..${sep}`)) fail(`${label} must stay below its root`);
  return target;
}

function yaml(path: string): Row {
  if (!existsSync(path)) fail(`missing required ledger file: ${path}`);
  return row(parse(readFileSync(path, 'utf8')), path);
}

function schema(document: Row, label: string): void {
  if (document.schema_version !== 1) fail(`${label}.schema_version must equal 1`);
}

function records(document: Row, key: string, label: string): Row[] {
  return list(document[key], `${label}.${key}`).map((value, index) => row(value, `${label}.${key}[${index}]`));
}

function ids(rows: Row[], label: string): Set<string> {
  const result = new Set<string>();
  for (const entry of rows) {
    const id = field(entry, 'id', label);
    if (result.has(id)) fail(`duplicate ${label} id: ${id}`);
    result.add(id);
  }
  return result;
}

function existingPage(repositoryRoot: string, destination: string, label: string): string {
  const page = under(repositoryRoot, destination, `${label}.destination`);
  if (!existsSync(page)) fail(`${label}.destination does not exist: ${destination}`);
  return page;
}

function hasSourceUnit(page: string, sourceUnit: string): boolean {
  const escaped = sourceUnit.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?:^|\\n)source_unit_id:\\s*["']?${escaped}["']?(?:\\s|$)`, 'm').test(readFileSync(page, 'utf8'));
}

function integratedDestination(entry: Row, repositoryRoot: string, sourceUnit: string, label: string): void {
  const page = existingPage(repositoryRoot, field(entry, 'destination', label), label);
  field(entry, 'destination_anchor', label);
  if (!hasSourceUnit(page, sourceUnit)) fail(`${label}.destination must carry reciprocal source_unit_id: ${sourceUnit}`);
}

function sourceHubDestination(entry: Row, repositoryRoot: string, courseRoot: string, label: string): void {
  const page = existingPage(repositoryRoot, field(entry, 'destination', label), label);
  field(entry, 'destination_anchor', label);
  if (page !== resolve(courseRoot, '_index.md')) fail(`${label}.destination must point to the course source hub`);
  field(entry, 'reason', label);
}

function exclusion(entry: Row, label: string): void {
  const reason = entry.reason;
  const evidence = entry.evidence;
  if (typeof reason !== 'string' || reason.trim() === '' || typeof evidence !== 'string' || evidence.trim() === '') {
    fail(`${label} requires reason and evidence`);
  }
}

function manifest(document: Row, courseRoot: string): Set<string> {
  schema(document, 'source-manifest');
  field(document, 'course', 'source-manifest');
  field(document, 'offering', 'source-manifest');
  if (!/^[a-f0-9]{40}$/.test(field(document, 'integration_base_commit', 'source-manifest'))) {
    fail('source-manifest.integration_base_commit must be a 40-character Git SHA');
  }
  date(document, 'retrieved_at', 'source-manifest');
  const objects = records(document, 'objects', 'source-manifest');
  const result = ids(objects, 'source-manifest.objects');
  for (const object of objects) {
    const label = `source-manifest.objects[${field(object, 'id', 'source-manifest.objects')}]`;
    field(object, 'kind', label);
    const authority = enumField(object, 'source_authority', authorities, label);
    const verified = enumField(object, 'verification_status', new Set(['verified', 'unverified']), label);
    if ((authority === 'official-course' || authority === 'official-author') && verified !== 'verified') {
      fail(`${label}: official artifacts must be verified`);
    }
    text(object.lecturer_or_author ?? object.lecturer ?? object.author, `${label}.lecturer_or_author`);
    date(object, 'meeting_date', label);
    field(object, 'title', label);
    const url = field(object, 'canonical_url', label);
    const revision = field(object, 'revision_or_checksum', label);
    if (!/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(revision)) fail(`${label}.revision_or_checksum must be a Git SHA or SHA-256`);
    if (url.includes('github.com')) {
      const pin = url.match(/github\.com\/[^/]+\/[^/]+\/(?:blob|tree|raw)\/([a-f0-9]{40})(?:\/|$)|github\.com\/[^/]+\/[^/]+\/commit\/([a-f0-9]{40})(?:$|[?#])/);
      const sha = pin?.[1] ?? pin?.[2];
      if (!sha) fail(`${label}.canonical_url must use a pinned GitHub SHA`);
      if (revision.length === 40 && revision !== sha) fail(`${label}.revision_or_checksum must match the pinned GitHub SHA`);
    }
    if (object.local_path !== undefined && object.local_path !== null) {
      const localPath = text(object.local_path, `${label}.local_path`);
      if (!existsSync(under(courseRoot, localPath, `${label}.local_path`))) fail(`missing source file: ${localPath}`);
    }
    field(object, 'mime', label);
    if (typeof object.bytes !== 'number' || !Number.isInteger(object.bytes) || object.bytes < 0) fail(`${label}.bytes must be a non-negative integer`);
    if (!(typeof object.page_count === 'number' && Number.isInteger(object.page_count) && object.page_count >= 0) && !isRow(object.video_metadata)) {
      fail(`${label} must record page_count or video_metadata`);
    }
    field(object, 'language', label);
    enumField(object, 'rights_status', rights, label);
    field(object, 'rights_evidence', label);
    date(object, 'retrieved_at', label);
  }
  return result;
}

function sourceUnits(document: Row, objectIds: Set<string>): Set<string> {
  schema(document, 'source-units');
  const units = records(document, 'units', 'source-units');
  const result = ids(units, 'source-units.units');
  for (const unit of units) {
    const label = `source-units.units[${field(unit, 'id', 'source-units.units')}]`;
    const object = field(unit, 'source_object', label);
    if (!objectIds.has(object)) fail(`${label} refers to unknown source object: ${object}`);
    field(unit, 'source_location', label);
    field(unit, 'kind', label);
    field(unit, 'title', label);
  }
  return result;
}

function coverage(document: Row, objectIds: Set<string>, unitIds: Set<string>, repositoryRoot: string, courseRoot: string): void {
  schema(document, 'coverage');
  const rows = records(document, 'rows', 'coverage');
  ids(rows, 'coverage.rows');
  const covered = new Set<string>();
  for (const entry of rows) {
    const label = `coverage.rows[${field(entry, 'id', 'coverage.rows')}]`;
    const object = field(entry, 'source_object', label);
    if (!objectIds.has(object)) fail(`${label} refers to unknown source object: ${object}`);
    const unit = field(entry, 'source_unit', label);
    if (!unitIds.has(unit)) fail(`${label} refers to unknown source unit: ${unit}`);
    if (covered.has(unit)) fail(`duplicate coverage for source unit: ${unit}`);
    covered.add(unit);
    field(entry, 'source_location', label);
    field(entry, 'kind', label);
    field(entry, 'title', label);
    const disposition = enumField(entry, 'disposition', dispositions, label);
    const sources = list(entry.primary_sources, `${label}.primary_sources`);
    if (sources.length === 0) fail(`${label}.primary_sources must not be empty`);
    for (const source of sources) {
      const id = text(source, `${label}.primary_sources entry`);
      if (!objectIds.has(id)) fail(`${label} refers to unknown primary source object: ${id}`);
    }
    if (disposition === 'integrated' || disposition === 'covered-existing') integratedDestination(entry, repositoryRoot, unit, label);
    else if (disposition === 'source-only') sourceHubDestination(entry, repositoryRoot, courseRoot, label);
    else exclusion(entry, label);
  }
  for (const unit of unitIds) if (!covered.has(unit)) fail(`missing coverage row for extracted source unit: ${unit}`);
}

function assetPaths(path: string): Set<string> {
  const registry = yaml(path);
  return new Set(records(registry, 'assets', 'asset-registry').map((asset) => field(asset, 'asset', 'asset-registry.assets')));
}

function visuals(document: Row, objectIds: Set<string>, repositoryRoot: string, courseRoot: string, registered: Set<string>): void {
  schema(document, 'visuals');
  const rows = records(document, 'rows', 'visuals');
  ids(rows, 'visuals.rows');
  for (const entry of rows) {
    const label = `visuals.rows[${field(entry, 'id', 'visuals.rows')}]`;
    const object = field(entry, 'source_object', label);
    if (!objectIds.has(object)) fail(`${label} refers to unknown source object: ${object}`);
    field(entry, 'source_location', label);
    field(entry, 'question', label);
    const members = list(entry.sequence_members, `${label}.sequence_members`).map((member, index) => row(member, `${label}.sequence_members[${index}]`));
    for (let index = 0; index < members.length; index += 1) {
      if (members[index].order !== index + 1) fail(`${label}.sequence_members must be ordered from 1 without gaps`);
      field(members[index], 'source_location', `${label}.sequence_members[${index}]`);
    }
    const disposition = enumField(entry, 'disposition', dispositions, label);
    if (disposition === 'integrated' || disposition === 'covered-existing') {
      const localFile = field(entry, 'local_file', label);
      field(entry, 'destination', label);
      field(entry, 'destination_anchor', label);
      existingPage(repositoryRoot, field(entry, 'destination', label), label);
      if (!existsSync(under(repositoryRoot, localFile, `${label}.local_file`))) fail(`${label}.local_file does not exist: ${localFile}`);
      if (!registered.has(localFile)) fail(`${label}.local_file is absent from the asset registry: ${localFile}`);
      if (!/^[a-f0-9]{64}$/.test(field(entry, 'parent_sha256', label))) fail(`${label}.parent_sha256 must be a SHA-256`);
      field(entry, 'transformation', label);
      field(entry, 'caption', label);
      if (typeof entry.attribution !== 'string' || entry.attribution.trim() === '' || typeof entry.rights_evidence !== 'string' || entry.rights_evidence.trim() === '') {
        fail(`${label} requires attribution and rights_evidence`);
      }
      field(entry, 'attribution', label);
      const rightsStatus = enumField(entry, 'rights_status', rights, label);
      if (rightsStatus === 'link-only' || rightsStatus === 'unknown') fail(`${label} cannot reuse a ${rightsStatus} visual`);
      field(entry, 'rights_holder', label);
      field(entry, 'rights_scope', label);
      field(entry, 'license_identifier', label);
      field(entry, 'license_url', label);
      field(entry, 'rights_evidence', label);
    } else if (disposition === 'source-only') sourceHubDestination(entry, repositoryRoot, courseRoot, label);
    else exclusion(entry, label);
    field(entry, 'rendered_route', label);
    field(entry, 'desktop_evidence', label);
    field(entry, 'narrow_evidence', label);
    field(entry, 'reviewer', label);
    date(entry, 'checked_at', label);
  }
}

/** Validate all four ledger documents for one pinned course offering. */
export function validateCourseLedger(courseRoot: string, options: CourseLedgerOptions = {}): void {
  const absoluteCourseRoot = resolve(courseRoot);
  const repositoryRoot = resolve(options.repositoryRoot ?? resolve(absoluteCourseRoot, '../../..'));
  const assetRegistryPath = resolve(repositoryRoot, options.assetRegistryPath ?? '05 Источники/asset-registry.yml');
  const objectIds = manifest(yaml(resolve(absoluteCourseRoot, 'source-manifest.yml')), absoluteCourseRoot);
  const unitIds = sourceUnits(yaml(resolve(absoluteCourseRoot, 'source-units.yml')), objectIds);
  coverage(yaml(resolve(absoluteCourseRoot, 'coverage.yml')), objectIds, unitIds, repositoryRoot, absoluteCourseRoot);
  visuals(yaml(resolve(absoluteCourseRoot, 'visuals.yml')), objectIds, repositoryRoot, absoluteCourseRoot, assetPaths(assetRegistryPath));
}

/** Validate real ledgers only after their course directories have been created. */
export function validateKnownCourseLedgers(repositoryRoot: string): string[] {
  const coursesRoot = resolve(repositoryRoot, '05 Источники/Courses');
  const offerings = ['Stanford CS336 Spring 2026', 'Berkeley Advanced LLM Agents Spring 2025'];
  const validated: string[] = [];
  for (const offering of offerings) {
    const courseRoot = resolve(coursesRoot, offering);
    if (!existsSync(courseRoot)) continue;
    validateCourseLedger(courseRoot, { repositoryRoot });
    validated.push(offering);
  }
  return validated;
}
