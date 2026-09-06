import { existsSync, readFileSync } from 'node:fs';
import { relative, resolve, sep } from 'node:path';
import matter from 'gray-matter';
import { parse } from 'yaml';
import { parseMarkdownHeadings } from '../adapter/build.ts';
import { wikiHeadingSlug } from '../adapter/links.ts';

type Row = Record<string, unknown>;

type SourceObject = {
  id: string;
  kind: string;
  pageCount?: number;
  videoDuration?: number;
  videoFrameCount?: number;
};

type SourceUnit = {
  id: string;
  sourceObject: string;
  sourceLocation: string;
  kind: string;
  title: string;
  testFile?: string;
  interfaceName?: string;
};

export type CourseLedgerOptions = {
  repositoryRoot?: string;
  assetRegistryPath?: string;
};

const coverageDispositions = new Set(['integrated', 'covered-existing', 'contract-only', 'source-only', 'excluded']);
const visualDispositions = new Set(['integrated', 'covered-existing', 'source-only', 'excluded']);
const authorities = new Set(['official-course', 'official-author', 'primary-paper', 'third-party-mirror', 'bookvar-original']);
const rights = new Set(['licensed', 'permission-recorded', 'link-only', 'unknown']);
const sourceObjectKinds = new Set(['executable-lecture', 'pdf', 'assignment', 'video']);
const semanticUnitKinds = new Set([
  'section',
  'mechanism',
  'derivation',
  'experiment',
  'worked-example',
  'failure-mode',
  'figure',
  'table',
  'code-trace',
  'visual-sequence',
  'administrative'
]);
const unitKindsByObject = new Map<string, ReadonlySet<string>>([
  ['executable-lecture', new Set(['section-boundary', 'rendered-text', 'rendered-image', 'rendered-link', ...semanticUnitKinds])],
  ['pdf', new Set(['page', 'heading', 'figure', 'table', 'multi-page-build', ...semanticUnitKinds])],
  ['assignment', new Set(['task', 'deliverable', 'test-interface', 'evaluation-requirement', ...semanticUnitKinds])],
  ['video', new Set(['segment', 'frame', 'transcript-event'])]
]);

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

function enumField(value: Row, name: string, values: ReadonlySet<string>, label: string): string {
  const result = field(value, name, label);
  if (!values.has(result)) fail(`${label}.${name} is not an allowed value`);
  return result;
}

function date(value: Row, name: string, label: string): string {
  const result = field(value, name, label);
  if (!/^\d{4}-\d{2}-\d{2}(?:T[^\s]+)?$/.test(result)) fail(`${label}.${name} must be an ISO date or timestamp`);
  return result;
}

function integer(value: unknown, label: string, minimum: number): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < minimum) {
    fail(`${label} must be an integer greater than or equal to ${minimum}`);
  }
  return value;
}

function positiveNumber(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) fail(`${label} must be a positive number`);
  return value;
}

function url(value: unknown, label: string): URL {
  const raw = text(value, label);
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    fail(`${label} must be an absolute URL`);
  }
  if (parsed.protocol !== 'https:' && parsed.protocol !== 'http:') fail(`${label} must use HTTP or HTTPS`);
  return parsed;
}

function sha256(value: unknown, label: string): string {
  const result = text(value, label);
  if (!/^[a-f0-9]{64}$/.test(result)) fail(`${label} must be a SHA-256`);
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

function markdownAnchors(page: string): Set<string> {
  const content = matter(readFileSync(page, 'utf8')).content;
  const anchors = new Set<string>();
  const occurrences = new Map<string, number>();
  for (const heading of parseMarkdownHeadings(content)) {
    const base = wikiHeadingSlug(heading.text);
    const occurrence = occurrences.get(base) ?? 0;
    anchors.add(occurrence === 0 ? base : `${base}-${occurrence}`);
    occurrences.set(base, occurrence + 1);
  }
  for (const match of content.matchAll(/<(?:a|span|div|section|h[1-6])\b[^>]*\b(?:id|name)=["']([^"']+)["'][^>]*>/gi)) {
    anchors.add(match[1]);
  }
  return anchors;
}

function destinationPage(entry: Row, repositoryRoot: string, label: string): string {
  const page = existingPage(repositoryRoot, field(entry, 'destination', label), label);
  const anchor = field(entry, 'destination_anchor', label);
  if (!markdownAnchors(page).has(anchor)) fail(`${label}.destination_anchor does not exist in destination: ${anchor}`);
  return page;
}

function hasSourceUnit(page: string, sourceUnit: string): boolean {
  const value = matter(readFileSync(page, 'utf8')).data.source_unit_id;
  return value === sourceUnit || (Array.isArray(value) && value.includes(sourceUnit));
}

function integratedDestination(entry: Row, repositoryRoot: string, sourceUnit: string, label: string): void {
  const page = destinationPage(entry, repositoryRoot, label);
  if (!hasSourceUnit(page, sourceUnit)) fail(`${label}.destination must carry reciprocal source_unit_id: ${sourceUnit}`);
}

function contractOnlyDestination(entry: Row, repositoryRoot: string, unit: SourceUnit, label: string): void {
  integratedDestination(entry, repositoryRoot, unit.id, label);
  if (unit.kind !== 'test-interface' || unit.testFile === undefined || unit.interfaceName === undefined) {
    fail(`${label}.contract-only is restricted to test-interface units`);
  }
  const evidence = field(entry, 'evidence', label);
  const marker = '#expected_tests';
  if (!evidence.endsWith(marker)) fail(`${label}.evidence must point to #expected_tests`);
  const contractPath = under(repositoryRoot, evidence.slice(0, -marker.length), `${label}.evidence`);
  const contract = yaml(contractPath);
  const expectedTests = list(contract.expected_tests, `${label}.evidence.expected_tests`)
    .map((value, index) => text(value, `${label}.evidence.expected_tests[${index}]`));
  const expectedTest = `${unit.testFile}::${unit.interfaceName}`;
  if (!expectedTests.includes(expectedTest)) fail(`${label}.evidence does not list ${expectedTest}`);
}

function sourceHubDestination(entry: Row, repositoryRoot: string, courseRoot: string, label: string): void {
  const page = destinationPage(entry, repositoryRoot, label);
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

function githubPin(canonicalUrl: URL, label: string): string | undefined {
  const hostname = canonicalUrl.hostname.toLowerCase();
  const path = canonicalUrl.pathname;
  let match: RegExpMatchArray | null = null;
  if (hostname === 'github.com' || hostname === 'www.github.com') {
    match = path.match(/^\/[^/]+\/[^/]+\/(?:blob|tree|raw|commit)\/([a-f0-9]{40})(?:\/|$)/);
  } else if (hostname === 'raw.githubusercontent.com' || hostname === 'raw.github.com') {
    match = path.match(/^\/[^/]+\/[^/]+\/([a-f0-9]{40})(?:\/|$)/);
  } else if (hostname === 'media.githubusercontent.com') {
    match = path.match(/^\/media\/[^/]+\/[^/]+\/([a-f0-9]{40})(?:\/|$)/);
  } else if (hostname === 'gist.githubusercontent.com') {
    match = path.match(/^\/[^/]+\/[^/]+\/raw\/([a-f0-9]{40})(?:\/|$)/);
  } else if (hostname.endsWith('.githubusercontent.com')) {
    fail(`${label}.canonical_url must use a pinned GitHub SHA`);
  }
  return match?.[1];
}

function videoMetadata(value: unknown, label: string): { duration: number; frameCount: number } {
  const metadata = row(value, `${label}.video_metadata`);
  const duration = positiveNumber(metadata.duration_seconds, `${label}.video_metadata.duration_seconds`);
  positiveNumber(metadata.frame_rate, `${label}.video_metadata.frame_rate`);
  const frameCount = integer(metadata.frame_count, `${label}.video_metadata.frame_count`, 1);
  return { duration, frameCount };
}

function manifest(document: Row, courseRoot: string): Map<string, SourceObject> {
  schema(document, 'source-manifest');
  field(document, 'course', 'source-manifest');
  field(document, 'offering', 'source-manifest');
  if (!/^[a-f0-9]{40}$/.test(field(document, 'integration_base_commit', 'source-manifest'))) {
    fail('source-manifest.integration_base_commit must be a 40-character Git SHA');
  }
  date(document, 'retrieved_at', 'source-manifest');
  const objects = records(document, 'objects', 'source-manifest');
  ids(objects, 'source-manifest.objects');
  const result = new Map<string, SourceObject>();
  for (const object of objects) {
    const id = field(object, 'id', 'source-manifest.objects');
    const label = `source-manifest.objects[${id}]`;
    const kind = enumField(object, 'kind', sourceObjectKinds, label);
    const authority = enumField(object, 'source_authority', authorities, label);
    const verified = enumField(object, 'verification_status', new Set(['verified', 'unverified']), label);
    if ((authority === 'official-course' || authority === 'official-author') && verified !== 'verified') {
      fail(`${label}: official artifacts must be verified`);
    }
    text(object.lecturer_or_author ?? object.lecturer ?? object.author, `${label}.lecturer_or_author`);
    date(object, 'meeting_date', label);
    field(object, 'title', label);
    const canonicalUrl = url(object.canonical_url, `${label}.canonical_url`);
    const revision = field(object, 'revision_or_checksum', label);
    if (!/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(revision)) fail(`${label}.revision_or_checksum must be a Git SHA or SHA-256`);
    const githubSource = canonicalUrl.hostname === 'github.com'
      || canonicalUrl.hostname === 'www.github.com'
      || canonicalUrl.hostname === 'raw.github.com'
      || canonicalUrl.hostname.endsWith('.githubusercontent.com');
    if (githubSource) {
      const pin = githubPin(canonicalUrl, label);
      if (!pin) fail(`${label}.canonical_url must use a pinned GitHub SHA`);
      if (revision.length === 40 && revision !== pin) fail(`${label}.revision_or_checksum must match the pinned GitHub SHA`);
    }
    if (object.local_path !== undefined && object.local_path !== null) {
      const localPath = text(object.local_path, `${label}.local_path`);
      if (!existsSync(under(courseRoot, localPath, `${label}.local_path`))) fail(`missing source file: ${localPath}`);
    }
    field(object, 'mime', label);
    integer(object.bytes, `${label}.bytes`, 0);
    let pageCount: number | undefined;
    let videoDuration: number | undefined;
    let videoFrameCount: number | undefined;
    if (kind === 'video') {
      const metadata = videoMetadata(object.video_metadata, label);
      videoDuration = metadata.duration;
      videoFrameCount = metadata.frameCount;
    } else {
      pageCount = integer(object.page_count, `${label}.page_count`, 1);
      if (object.video_metadata !== undefined) videoMetadata(object.video_metadata, label);
    }
    field(object, 'language', label);
    enumField(object, 'rights_status', rights, label);
    field(object, 'rights_evidence', label);
    date(object, 'retrieved_at', label);
    result.set(id, { id, kind, pageCount, videoDuration, videoFrameCount });
  }
  return result;
}

function checkedPage(unit: Row, name: string, sourceObject: SourceObject, label: string): number {
  const page = integer(unit[name], `${label}.${name}`, 1);
  if (sourceObject.pageCount !== undefined && page > sourceObject.pageCount) {
    fail(`${label}.${name} exceeds source object page_count`);
  }
  return page;
}

function checkedSeconds(unit: Row, name: string, sourceObject: SourceObject, label: string): number {
  const seconds = typeof unit[name] === 'number' && Number.isFinite(unit[name]) && unit[name] >= 0
    ? unit[name] as number
    : fail(`${label}.${name} must be a non-negative number`);
  if (sourceObject.videoDuration !== undefined && seconds > sourceObject.videoDuration) {
    fail(`${label}.${name} exceeds source object duration`);
  }
  return seconds;
}

function validateUnitShape(unit: Row, sourceObject: SourceObject, kind: string, label: string): void {
  integer(unit.order, `${label}.order`, 1);
  // `figure` and `table` predate the semantic ledger extension for PDF rows.
  // Treat them as semantic only when the extractor supplies an explicit stable ID.
  const usesSemanticShape = semanticUnitKinds.has(kind)
    && (!(kind === 'figure' || kind === 'table') || unit.semantic_id !== undefined);
  if (usesSemanticShape) {
    field(unit, 'semantic_id', label);
    if (sourceObject.kind === 'executable-lecture') {
      const start = integer(unit.event_start, `${label}.event_start`, 1);
      const end = integer(unit.event_end, `${label}.event_end`, 1);
      if (end < start) fail(`${label}.event_end must be greater than or equal to event_start`);
    } else {
      if (unit.page !== undefined) {
        checkedPage(unit, 'page', sourceObject, label);
      } else {
        const start = checkedPage(unit, 'page_start', sourceObject, label);
        const end = checkedPage(unit, 'page_end', sourceObject, label);
        if (end < start) fail(`${label}.page_end must be greater than or equal to page_start`);
      }
    }
    return;
  }
  if (sourceObject.kind === 'executable-lecture') {
    if (kind === 'section-boundary') {
      const level = integer(unit.heading_level, `${label}.heading_level`, 1);
      if (level > 6) fail(`${label}.heading_level must not exceed 6`);
    } else if (kind === 'rendered-text') {
      sha256(unit.content_sha256, `${label}.content_sha256`);
    } else if (kind === 'rendered-image') {
      url(unit.asset_url, `${label}.asset_url`);
    } else {
      url(unit.target_url, `${label}.target_url`);
    }
  } else if (sourceObject.kind === 'pdf') {
    if (kind === 'multi-page-build') {
      const start = checkedPage(unit, 'page_start', sourceObject, label);
      const end = checkedPage(unit, 'page_end', sourceObject, label);
      if (end <= start) fail(`${label}.page_end must be greater than page_start`);
    } else {
      checkedPage(unit, 'page', sourceObject, label);
      if (kind === 'heading') {
        const level = integer(unit.heading_level, `${label}.heading_level`, 1);
        if (level > 6) fail(`${label}.heading_level must not exceed 6`);
      } else if (kind === 'figure') field(unit, 'figure_label', label);
      else if (kind === 'table') field(unit, 'table_label', label);
    }
  } else if (sourceObject.kind === 'assignment') {
    const identifiers: Record<string, string> = {
      task: 'task_id',
      deliverable: 'deliverable_id',
      'test-interface': 'interface_name',
      'evaluation-requirement': 'criterion_id'
    };
    field(unit, identifiers[kind], label);
  } else if (kind === 'frame') {
    const frame = integer(unit.frame, `${label}.frame`, 0);
    if (sourceObject.videoFrameCount !== undefined && frame >= sourceObject.videoFrameCount) {
      fail(`${label}.frame exceeds source object frame_count`);
    }
  } else {
    const start = checkedSeconds(unit, 'start_seconds', sourceObject, label);
    const end = checkedSeconds(unit, 'end_seconds', sourceObject, label);
    if (end <= start) fail(`${label}.end_seconds must be greater than start_seconds`);
  }
}

function sourceUnits(document: Row, objects: ReadonlyMap<string, SourceObject>): Map<string, SourceUnit> {
  schema(document, 'source-units');
  const units = records(document, 'units', 'source-units');
  ids(units, 'source-units.units');
  const result = new Map<string, SourceUnit>();
  const expectedOrder = new Map<string, number>();
  for (const unit of units) {
    const id = field(unit, 'id', 'source-units.units');
    const label = `source-units.units[${id}]`;
    const objectId = field(unit, 'source_object', label);
    const sourceObject = objects.get(objectId);
    if (!sourceObject) fail(`${label} refers to unknown source object: ${objectId}`);
    const kind = enumField(unit, 'kind', unitKindsByObject.get(sourceObject.kind)!, label);
    const order = integer(unit.order, `${label}.order`, 1);
    const expected = expectedOrder.get(objectId) ?? 1;
    if (order !== expected) fail(`${label}.order must be ${expected} for source object ${objectId}`);
    expectedOrder.set(objectId, expected + 1);
    validateUnitShape(unit, sourceObject, kind, label);
    result.set(id, {
      id,
      sourceObject: objectId,
      sourceLocation: field(unit, 'source_location', label),
      kind,
      title: field(unit, 'title', label),
      testFile: kind === 'test-interface' && typeof unit.test_file === 'string'
        ? text(unit.test_file, `${label}.test_file`)
        : undefined,
      interfaceName: kind === 'test-interface' ? field(unit, 'interface_name', label) : undefined
    });
  }
  return result;
}

function matchingCoverageField(entry: Row, name: string, expected: string, label: string): void {
  const actual = field(entry, name, label);
  if (actual !== expected) fail(`${label}.${name} does not match extracted source unit`);
}

function coverage(
  document: Row,
  objects: ReadonlyMap<string, SourceObject>,
  units: ReadonlyMap<string, SourceUnit>,
  repositoryRoot: string,
  courseRoot: string
): void {
  schema(document, 'coverage');
  const rows = records(document, 'rows', 'coverage');
  ids(rows, 'coverage.rows');
  const covered = new Set<string>();
  for (const entry of rows) {
    const label = `coverage.rows[${field(entry, 'id', 'coverage.rows')}]`;
    const object = field(entry, 'source_object', label);
    if (!objects.has(object)) fail(`${label} refers to unknown source object: ${object}`);
    const unitId = field(entry, 'source_unit', label);
    const unit = units.get(unitId);
    if (!unit) fail(`${label} refers to unknown source unit: ${unitId}`);
    if (covered.has(unitId)) fail(`duplicate coverage for source unit: ${unitId}`);
    covered.add(unitId);
    matchingCoverageField(entry, 'source_object', unit.sourceObject, label);
    matchingCoverageField(entry, 'source_location', unit.sourceLocation, label);
    matchingCoverageField(entry, 'kind', unit.kind, label);
    matchingCoverageField(entry, 'title', unit.title, label);
    const disposition = enumField(entry, 'disposition', coverageDispositions, label);
    const sources = list(entry.primary_sources, `${label}.primary_sources`);
    if (sources.length === 0) fail(`${label}.primary_sources must not be empty`);
    for (const source of sources) {
      const id = text(source, `${label}.primary_sources entry`);
      if (!objects.has(id)) fail(`${label} refers to unknown primary source object: ${id}`);
    }
    if (disposition === 'integrated' || disposition === 'covered-existing') {
      integratedDestination(entry, repositoryRoot, unitId, label);
    } else if (disposition === 'contract-only') {
      contractOnlyDestination(entry, repositoryRoot, unit, label);
    } else if (disposition === 'source-only') {
      sourceHubDestination(entry, repositoryRoot, courseRoot, label);
    } else {
      exclusion(entry, label);
    }
  }
  for (const unit of units.keys()) if (!covered.has(unit)) fail(`missing coverage row for extracted source unit: ${unit}`);
}

function assetPaths(path: string): Set<string> {
  const registry = yaml(path);
  return new Set(records(registry, 'assets', 'asset-registry').map((asset) => field(asset, 'asset', 'asset-registry.assets')));
}

function orderedLocations(value: unknown, label: string, minimum: number, maximum?: number): void {
  const locations = list(value, label);
  if (locations.length === 0) fail(`${label} must not be empty`);
  let previous = minimum - 1;
  for (const [index, location] of locations.entries()) {
    const current = integer(location, `${label}[${index}]`, minimum);
    if (current <= previous) fail(`${label} must be strictly ordered without duplicates`);
    if (maximum !== undefined && current > maximum) fail(`${label}[${index}] exceeds its source object`);
    previous = current;
  }
}

function exactVisualLocation(entry: Row, sourceObject: SourceObject, label: string): void {
  if (sourceObject.kind === 'video') {
    if (entry.source_pages !== undefined) fail(`${label} must use source_frames for a video object`);
    orderedLocations(entry.source_frames, `${label}.source_frames`, 0, sourceObject.videoFrameCount === undefined ? undefined : sourceObject.videoFrameCount - 1);
  } else {
    if (entry.source_frames !== undefined) fail(`${label} must use source_pages for a non-video object`);
    orderedLocations(entry.source_pages, `${label}.source_pages`, 1, sourceObject.pageCount);
  }
}

function visualRights(entry: Row, label: string): string {
  field(entry, 'attribution', label);
  const status = enumField(entry, 'rights_status', rights, label);
  field(entry, 'rights_holder', label);
  field(entry, 'rights_scope', label);
  field(entry, 'rights_evidence', label);
  if (status === 'licensed' || status === 'permission-recorded') {
    field(entry, 'license_identifier', label);
    url(entry.license_url, `${label}.license_url`);
  }
  return status;
}

function visuals(
  document: Row,
  objects: ReadonlyMap<string, SourceObject>,
  units: ReadonlyMap<string, SourceUnit>,
  repositoryRoot: string,
  courseRoot: string,
  registered: ReadonlySet<string>
): void {
  schema(document, 'visuals');
  const rows = records(document, 'rows', 'visuals');
  ids(rows, 'visuals.rows');
  for (const entry of rows) {
    const label = `visuals.rows[${field(entry, 'id', 'visuals.rows')}]`;
    const objectId = field(entry, 'source_object', label);
    const sourceObject = objects.get(objectId);
    if (!sourceObject) fail(`${label} refers to unknown source object: ${objectId}`);
    if (entry.source_units !== undefined) {
      const linkedUnits = list(entry.source_units, `${label}.source_units`);
      if (linkedUnits.length === 0) fail(`${label}.source_units must not be empty`);
      for (const [index, value] of linkedUnits.entries()) {
        const unitId = text(value, `${label}.source_units[${index}]`);
        const unit = units.get(unitId);
        if (!unit) fail(`${label}.source_units refers to unknown source unit: ${unitId}`);
        if (unit.sourceObject !== objectId) fail(`${label}.source_units refers to a unit owned by another source object`);
      }
    }
    field(entry, 'source_location', label);
    exactVisualLocation(entry, sourceObject, label);
    field(entry, 'question', label);
    const members = list(entry.sequence_members, `${label}.sequence_members`)
      .map((member, index) => row(member, `${label}.sequence_members[${index}]`));
    if (members.length === 0) fail(`${label}.sequence_members must not be empty`);
    for (let index = 0; index < members.length; index += 1) {
      if (members[index].order !== index + 1) fail(`${label}.sequence_members must be ordered from 1 without gaps`);
      field(members[index], 'source_location', `${label}.sequence_members[${index}]`);
    }
    const rightsStatus = visualRights(entry, label);
    const disposition = enumField(entry, 'disposition', visualDispositions, label);
    if (disposition === 'integrated' || disposition === 'covered-existing') {
      if (rightsStatus === 'link-only' || rightsStatus === 'unknown') fail(`${label} cannot reuse a ${rightsStatus} visual`);
      const localFile = field(entry, 'local_file', label);
      destinationPage(entry, repositoryRoot, label);
      if (!existsSync(under(repositoryRoot, localFile, `${label}.local_file`))) fail(`${label}.local_file does not exist: ${localFile}`);
      if (!registered.has(localFile)) fail(`${label}.local_file is absent from the asset registry: ${localFile}`);
      sha256(entry.parent_sha256, `${label}.parent_sha256`);
      field(entry, 'transformation', label);
      field(entry, 'caption', label);
    } else if (disposition === 'source-only') {
      sourceHubDestination(entry, repositoryRoot, courseRoot, label);
    } else {
      exclusion(entry, label);
    }
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
  const objects = manifest(yaml(resolve(absoluteCourseRoot, 'source-manifest.yml')), absoluteCourseRoot);
  const units = sourceUnits(yaml(resolve(absoluteCourseRoot, 'source-units.yml')), objects);
  coverage(yaml(resolve(absoluteCourseRoot, 'coverage.yml')), objects, units, repositoryRoot, absoluteCourseRoot);
  visuals(yaml(resolve(absoluteCourseRoot, 'visuals.yml')), objects, units, repositoryRoot, absoluteCourseRoot, assetPaths(assetRegistryPath));
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
