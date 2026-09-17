#!/usr/bin/env node
'use strict';

// Local reconciliation only: no page edits, source rewrites, network or git.
// --emit prints deterministic JSON; --check compares it with finding-ledger.json.
// --self-test exercises loss, duplication, status and original-field safeguards.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const ROOT = path.resolve(__dirname, '../../..');
const ORIGINAL_DIR = 'docs/audit/2026-09-15';
const COMPLETION_DIR = 'docs/audit/2026-09-15-completion';
const ORIGINALS = ['foundations', 'systems', 'advanced', 'reference-practice', 'questions'];
const COMPLETIONS = ['foundations', 'systems', 'advanced', 'reference-concepts', 'atlas', 'practice', 'navigation', 'questions'];
const FIELDS = ['severity', 'line', 'excerpt', 'issue', 'action'];

function reconcile(originals, completions, inventory, supplementalReviews = []) {
  const signature = (page, finding) => JSON.stringify([page, ...FIELDS.map(k => finding[k])]);
  const pointer = (report, json_pointer) => ({report: report.path, json_pointer});
  const originalFindings = new Map(), completionFindings = new Map(), pageMap = new Map();
  const inventoryPages = new Map(inventory.document.pages.map((p, i) => [p.path, i]));
  assert.equal(inventoryPages.size, inventory.document.pages.length, 'Duplicate inventory page');
  const pages = [], findings = [], by_original_report = {}, by_completion_report = {};
  for (const report of originals) {
    by_original_report[report.path] = {pages: 0, findings: 0};
    for (const [i, page] of report.document.pages.entries()) {
      assert(!pageMap.has(page.path), 'Duplicate original page: ' + page.path);
      assert(inventoryPages.has(page.path), 'Original page absent from inventory: ' + page.path);
      const row = {path: page.path, original_finding_count: page.findings.length,
        original_audit: {...pointer(report, '/pages/' + i), read_sha256: page.read_sha256 ?? null,
          reported_reading: page.reading ?? null},
        inventory: {...pointer(inventory, '/pages/' + inventoryPages.get(page.path)),
          recorded_sha256: inventory.document.pages[inventoryPages.get(page.path)].sha256 ?? null},
        completion_full_read: 'not-reported-in-completion', completion: [], finding_ids: []};
      pageMap.set(page.path, row); pages.push(row);
      by_original_report[report.path].pages++;
      for (const [j, finding] of page.findings.entries()) {
        for (const field of FIELDS) assert(Object.hasOwn(finding, field), 'Missing original field: ' + field);
        const key = signature(page.path, finding);
        assert(!originalFindings.has(key), 'Duplicate original finding: ' + key);
        const id = path.basename(report.path, '.json') + ':' + String(i + 1).padStart(3, '0') + ':' + (j + 1);
        const record = {id, path: page.path, ...Object.fromEntries(FIELDS.map(k => [k, finding[k]])),
          original_extra: Object.fromEntries(Object.entries(finding).filter(([k]) => !FIELDS.includes(k))),
          origin: pointer(report, '/pages/' + i + '/findings/' + j)};
        originalFindings.set(key, record); row.finding_ids.push(id);
        by_original_report[report.path].findings++;
      }
    }
  }
  assert.equal(pageMap.size, inventoryPages.size, 'Inventory/original page coverage mismatch');
  for (const report of completions) {
    by_completion_report[report.path] = {pages: 0, findings: 0, dispositions: {}};
    for (const [i, page] of report.document.pages.entries()) {
      assert(pageMap.has(page.path), 'Unexpected completion page: ' + page.path);
      const target = pageMap.get(page.path);
      assert(!target.completion.some(p => p.report === report.path), 'Duplicate completion page: ' + page.path);
      const reading = page.full_read_before ?? page.full_read ?? page.reading ?? page.read_before;
      const full = reading === true || ['full', 'full-page', 'complete-full-page'].includes(reading);
      if (full) target.completion_full_read = 'reported-full';
      else if (target.completion_full_read !== 'reported-full') target.completion_full_read = 'not-established-by-report';
      target.completion.push({...pointer(report, '/pages/' + i),
        recorded_hashes: Object.fromEntries(Object.entries(page).filter(([k]) => /sha256|^(before|after)$/.test(k))),
        recorded_read_evidence: Object.fromEntries(Object.entries(page).filter(([k]) => /read/.test(k) && !/sha256/.test(k))),
        recorded_snapshot_evidence: Object.fromEntries(Object.entries(page).filter(([k]) => /snapshot/.test(k))),
        source_verification_pointer: report.path + '#/pages/' + i,
        hash_scope: 'Values recorded by this completion source; not final live-page hashes.'});
      by_completion_report[report.path].pages++;
      for (const [j, finding] of page.findings.entries()) {
        const key = signature(page.path, finding);
        assert(originalFindings.has(key), 'Unexpected completion finding (exact identity mismatch): ' + key);
        assert(!completionFindings.has(key), 'Duplicate completion finding: ' + key);
        const reported = finding.status ?? finding.outcome;
        if (finding.status && finding.outcome) assert.equal(finding.status, finding.outcome, 'Conflicting disposition');
        const aliases = {fixed: 'fixed', 'already-fixed': 'already-fixed', already_fixed: 'already-fixed',
          'rejected-with-evidence': 'rejected-with-evidence', rejected_with_evidence: 'rejected-with-evidence',
          open: 'pending', pending: 'pending'};
        assert(Object.hasOwn(aliases, reported), 'Unknown disposition: ' + reported);
        const evidence = finding.resolution ?? finding.evidence;
        assert(evidence, 'Missing completion evidence: ' + key);
        const disposition = aliases[reported];
        completionFindings.set(key, {disposition, reported_disposition: reported,
          evidence: {text: evidence, ...pointer(report, '/pages/' + i + '/findings/' + j)},
          completion_page: pointer(report, '/pages/' + i),
          acceptance: finding.acceptance ?? 'Not independently accepted by this reconciliation; see source report.',
          completion_report_status: report.document.status ?? null});
        by_completion_report[report.path].findings++;
        const counts = by_completion_report[report.path].dispositions;
        counts[disposition] = (counts[disposition] ?? 0) + 1;
      }
    }
  }
  for (const report of supplementalReviews) {
    let matched = 0;
    const seen = new Set();
    for (const [i, line] of report.text.split('\n').entries()) {
      const match = line.match(/^\| \x60([^\x60]+\.md)\x60 \| 1–EOF \| (\d+) \| (\d+) \| \x60([a-f0-9]{64})\x60 \|$/);
      if (!match) continue;
      const [, sourcePath, lines, bytes, hash] = match;
      assert(pageMap.has(sourcePath), 'Unexpected supplemental page: ' + sourcePath);
      assert(!seen.has(sourcePath), 'Duplicate supplemental full-read evidence: ' + sourcePath);
      seen.add(sourcePath);
      const page = pageMap.get(sourcePath);
      assert.equal(page.original_finding_count, 0, 'Supplemental review must cover original zero-finding pages');
      assert.equal(page.original_audit.read_sha256, hash, 'Supplemental audit SHA mismatch: ' + sourcePath);
      page.supplemental_read_reviews ??= [];
      page.supplemental_read_reviews.push({report: report.path, report_sha256: report.sha256,
        line: i + 1, reading: 'full', read_span: '1–EOF',
        lines_excluding_trailing_empty: Number(lines), bytes: Number(bytes),
        recorded_sha256: hash, matches_original_audit_sha256: true,
        acceptance: 'Full source reading and source-level links; final rendered-link acceptance remains separate.'});
      matched++;
    }
    assert(matched > 0, 'No full-read evidence rows in supplemental review: ' + report.path);
  }
  for (const page of pages) {
    page.renewed_full_read = page.completion_full_read === 'reported-full'
      || (page.supplemental_read_reviews?.length ?? 0) > 0;
  }
  const dispositions = {}, severity = {};
  for (const [key, original] of originalFindings) {
    assert(completionFindings.has(key), 'Missing completion finding: ' + key);
    const finding = {...original, ...completionFindings.get(key)};
    findings.push(finding);
    dispositions[finding.disposition] = (dispositions[finding.disposition] ?? 0) + 1;
    severity[finding.severity] = (severity[finding.severity] ?? 0) + 1;
  }
  return {
    schema: 'bookvar-original-finding-ledger-v1',
    audit_date: '2026-09-15',
    status: 'reconciled-source-reports; final-acceptance-and-superseding-hashes-pending',
    policy: {
      identity: 'Exact path plus severity, line, excerpt, issue and action; no fuzzy matches.',
      scope: 'Only findings in pages[].findings of five original audits count.',
      reported_status: 'outcome/status normalized; open is pending. Fixed is a source-report disposition, not independent final acceptance.',
      reading: 'Completion reading is reported per source, not performed by this reconciliation. A zero-finding page alone proves no new full reading.',
      renewed_reading: 'Union of explicit completion JSON full-read evidence and supplemental Markdown full-read rows. Historical 274 completion rows remain unchanged; six later reads are additive, with report SHA and line pointers.',
      hashes: 'Per-source recorded audit/read/snapshot hashes only. No assertion that old after hashes equal the live page after later deltas.',
      later_findings: 'New render/P1/review/course findings are excluded from original counts and require separate tracking.',
      questions: 'archived_answers[].findings mirrors 45 page findings and is not counted again; extra_archive_items are not original finding records.'
    },
    sources: [
      {path: inventory.path, sha256: inventory.sha256, role: 'original-inventory'},
      ...originals.map(r => ({path: r.path, sha256: r.sha256, role: 'original-audit', status: r.document.status ?? null})),
      ...completions.map(r => ({path: r.path, sha256: r.sha256, role: 'completion-report',
        status: r.document.status ?? null, report_level_evidence: r.path + '#/'})),
      ...supplementalReviews.map(r => ({path: r.path, sha256: r.sha256, role: 'supplemental-full-read-review'}))
    ],
    summary: {original_findings: findings.length, original_pages: pages.length,
      pages_with_findings: pages.filter(p => p.original_finding_count > 0).length,
      zero_finding_pages: pages.filter(p => p.original_finding_count === 0).length,
      completion_reported_pages: pages.filter(p => p.completion.length).length,
      completion_full_read_reported_pages: pages.filter(p => p.completion_full_read === 'reported-full').length,
      completion_full_read_not_reported_paths: pages.filter(p => p.completion_full_read !== 'reported-full').map(p => p.path),
      supplemental_full_read_pages: pages.filter(p => p.supplemental_read_reviews?.length).length,
      renewed_full_read_pages: pages.filter(p => p.renewed_full_read).length,
      renewed_full_read_not_reported_paths: pages.filter(p => !p.renewed_full_read).map(p => p.path),
      dispositions, original_severity: severity, by_original_report, by_completion_report,
      later_findings_included: 0, exact_completion_matches: completionFindings.size},
    pages, findings
  };
}

function selfTest() {
  const issue = {severity: 'P2', line: 7, excerpt: 'точная цитата\nвторая строка', issue: 'Ошибка', action: 'Исправить', question: 1};
  const second = {...issue, line: 8, issue: 'Проверить публикацию'};
  const source = (name, pages) => ({path: name, sha256: 'a'.repeat(64), document: {status: 'pending-review', pages}});
  const originals = [source('original.json', [{path: 'a.md', reading: 'full', read_sha256: 'b'.repeat(64), findings: [issue, second]}, {path: 'zero.md', reading: 'full', read_sha256: 'e'.repeat(64), findings: []}])];
  const completions = [source('completion.json', [{path: 'a.md', full_read_before: true, before_sha256: 'c'.repeat(64), after_sha256: 'd'.repeat(64), findings: [{...issue, outcome: 'fixed', evidence: 'Исправлена общая ось.', acceptance: 'pending-independent-review'}, {...second, status: 'open', resolution: 'Awaiting root build'}]}])];
  const inventory = source('inventory.json', [{path: 'a.md'}, {path: 'zero.md'}]);
  const result = reconcile(originals, completions, inventory);
  assert.equal(result.summary?.original_findings, 2, 'must retain every original finding');
  assert.equal(result.summary.original_pages, 2);
  assert.equal(result.summary.completion_full_read_reported_pages, 1);
  assert.equal(result.findings[0].excerpt, 'точная цитата\nвторая строка');
  assert.equal(result.findings[0].original_extra.question, 1);
  assert.equal(result.findings[0].disposition, 'fixed');
  assert.equal(result.findings[0].acceptance, 'pending-independent-review');
  assert.equal(result.findings[1].disposition, 'pending');
  assert.equal(result.findings[1].reported_disposition, 'open');
  assert.equal(result.pages[1].completion_full_read, 'not-reported-in-completion');
  assert.equal(result.pages[0].completion[0].recorded_hashes.after_sha256, 'd'.repeat(64));
  const mark = String.fromCharCode(96);
  const reviewRow = '| ' + mark + 'zero.md' + mark + ' | 1–EOF | 10 | 200 | ' + mark + 'e'.repeat(64) + mark + ' |';
  const review = {path: 'zero-review.md', sha256: 'f'.repeat(64), text: '# Review\n\n' + reviewRow + '\n'};
  const updated = reconcile(originals, completions, inventory, [review]);
  assert.equal(updated.summary.renewed_full_read_pages, 2, 'supplemental full reading must fill the gap');
  assert.equal(updated.summary.completion_full_read_reported_pages, 1, 'do not rewrite old completion evidence');
  assert.equal(updated.summary.supplemental_full_read_pages, 1);
  assert.deepEqual(updated.summary.renewed_full_read_not_reported_paths, []);
  assert.deepEqual(updated.findings, result.findings, 'reading evidence must not close or mutate findings');
  assert.deepEqual(updated.pages[1].original_audit, result.pages[1].original_audit);
  assert.equal(updated.pages[1].supplemental_read_reviews[0].line, 3);
  assert.equal(updated.pages[1].supplemental_read_reviews[0].recorded_sha256, 'e'.repeat(64));
  assert.equal(updated.sources.at(-1).sha256, 'f'.repeat(64));
  assert.throws(() => reconcile(originals, completions, inventory, [{...review, text: review.text + reviewRow}]), /Duplicate supplemental/);
  assert.throws(() => reconcile(originals, completions, inventory, [{...review, text: review.text.replace('e'.repeat(64), 'd'.repeat(64))}]), /Supplemental audit SHA mismatch/);
  assert.throws(() => reconcile(originals, completions, inventory, [{...review, text: review.text.replace('1–EOF', 'partial')}]), /No full-read evidence rows/);
  const mutate = fn => { const copy = structuredClone(completions); fn(copy[0].document.pages[0]); return copy; };
  assert.throws(() => reconcile(originals, mutate(p => p.findings.pop()), inventory), /Missing completion/);
  assert.throws(() => reconcile(originals, mutate(p => p.findings.push(p.findings[0])), inventory), /Duplicate completion/);
  assert.throws(() => reconcile(originals, mutate(p => p.findings[0].excerpt += '!'), inventory), /Unexpected completion/);
  assert.throws(() => reconcile(originals, mutate(p => p.findings[0].outcome = 'probably-fixed'), inventory), /Unknown disposition/);
  assert.throws(() => reconcile(originals, mutate(p => p.findings[0].status = 'open'), inventory), /Conflicting disposition/);
  const repeated = structuredClone(originals);
  repeated[0].document.pages[0].findings.push(issue);
  assert.throws(() => reconcile(repeated, completions, inventory), /Duplicate original/);
  console.log('PASS: exact fields, all findings, pending render, per-source hashes, zero-finding read boundary, missing/duplicate/mutated/unknown/conflicting records');
}

if (require.main === module) {
  if (process.argv.includes('--self-test')) selfTest();
  else {
    const load = file => { const bytes = fs.readFileSync(path.join(ROOT, file));
      return {path: file, sha256: crypto.createHash('sha256').update(bytes).digest('hex'), document: JSON.parse(bytes)}; };
    const originals = ORIGINALS.map(n => load(ORIGINAL_DIR + '/' + n + '.json'));
    const completions = COMPLETIONS.map(n => load(COMPLETION_DIR + '/' + n + '.json'));
    const inventory = load(ORIGINAL_DIR + '/inventory.json');
    const reviewPath = COMPLETION_DIR + '/questions-zero-finding-review.md';
    const reviewBytes = fs.readFileSync(path.join(ROOT, reviewPath));
    const review = {path: reviewPath, sha256: crypto.createHash('sha256').update(reviewBytes).digest('hex'), text: reviewBytes.toString('utf8')};
    const ledger = reconcile(originals, completions, inventory, [review]);
    assert.equal(ledger.summary.original_findings, 442, 'Original issue count changed; investigate rather than silently accepting');
    assert.equal(ledger.summary.original_pages, 280, 'Original inventory count changed; investigate rather than silently accepting');
    assert.equal(ledger.summary.supplemental_full_read_pages, 6, 'Expected explicit evidence for exactly six supplemental pages');
    assert.equal(ledger.summary.renewed_full_read_pages, 280, 'Renewed full reading is not complete');
    const encoded = JSON.stringify(ledger, null, 2) + '\n';
    if (process.argv.includes('--emit')) process.stdout.write(encoded);
    else if (process.argv.includes('--patch')) {
      const target = path.join(COMPLETION_DIR, 'finding-ledger.json');
      const absolute = path.join(ROOT, target);
      const previous = fs.existsSync(absolute) ? fs.readFileSync(absolute, 'utf8') : null;
      const header = previous === null ? '*** Add File: ' + target + '\n' : '*** Update File: ' + target + '\n@@\n';
      const removed = previous === null ? '' : previous.trimEnd().split('\n').map(line => '-' + line).join('\n') + '\n';
      process.stdout.write('*** Begin Patch\n' + header + removed + encoded.trimEnd().split('\n').map(line => '+' + line).join('\n') + '\n*** End Patch\n');
    } else {
      assert(process.argv.includes('--check'), 'Use --self-test, --emit, --patch or --check');
      const stored = fs.readFileSync(path.join(ROOT, COMPLETION_DIR, 'finding-ledger.json'), 'utf8');
      assert.equal(stored, encoded, 'Ledger differs from current source reports; review inputs and regenerate deliberately');
      console.log(JSON.stringify(ledger.summary, null, 2));
      console.log('PASS: 442 exact original findings, 280 pages, source-report hashes and dispositions reconciled');
    }
  }
}
module.exports = {reconcile};
