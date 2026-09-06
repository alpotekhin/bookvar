import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { parse } from 'yaml';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../..');
const courseRoot = resolve(root, '05 Источники/Courses/Stanford CS336 Spring 2026');
const assignmentRoot = resolve(courseRoot, 'Assignments/assignment1-basics');
const contractPath = resolve(root, '06 Практика/Contracts/stanford-cs336-a1.yml');
const practicePath = resolve(root, '06 Практика/20 Собрать языковую модель с нуля.md');

type Contract = {
  schema_version: number;
  source: {
    repository: string;
    commit: string;
    handout: { path: string; sha256: string };
    adapters: { path: string; sha256: string };
    tests: {
      root: string;
      tree_hash_algorithm: string;
      tree_includes: string;
      tree_excludes: string[];
      tree_sha256: string;
    };
  };
  official_interfaces: {
    adapters: string[];
    tests: Array<{ file: string; names: string[] }>;
  };
  profiles: Record<string, Record<string, unknown>>;
  evidence: { required_fields: string[]; event_types: string[] };
  acceptance: Record<string, Record<string, unknown>>;
  boundaries: Record<string, unknown>;
};

function sha256(data: Buffer | string): string {
  return createHash('sha256').update(data).digest('hex');
}

function functionNames(path: string, pattern: RegExp): string[] {
  return [...readFileSync(path, 'utf8').matchAll(pattern)].map((match) => match[1]).sort();
}

function testFiles(): string[] {
  return readdirSync(resolve(assignmentRoot, 'tests'))
    .filter((name) => /^test_.*\.py$/.test(name))
    .sort();
}

const generatedTestTreeDirectories = new Set(['__pycache__', '.pytest_cache']);

function testTreeFiles(relativeDirectory = 'tests'): string[] {
  return readdirSync(resolve(assignmentRoot, relativeDirectory), { withFileTypes: true })
    .flatMap((entry) => {
      const relativePath = `${relativeDirectory}/${entry.name}`;
      if (entry.isDirectory()) {
        return generatedTestTreeDirectories.has(entry.name) ? [] : testTreeFiles(relativePath);
      }
      if (!entry.isFile() || entry.name === '.DS_Store' || entry.name.endsWith('.pyc')) return [];
      return [relativePath];
    })
    .sort();
}

function testTreeSha256(files: string[]): string {
  const digest = createHash('sha256');
  for (const file of files) {
    digest.update(file);
    digest.update('\0');
    digest.update(readFileSync(resolve(assignmentRoot, file)));
    digest.update('\0');
  }
  return digest.digest('hex');
}

describe('Stanford CS336 Assignment 1 capstone contract', () => {
  const contract = parse(readFileSync(contractPath, 'utf8')) as Contract;
  const lock = JSON.parse(readFileSync(resolve(courseRoot, 'snapshot-lock.json'), 'utf8')) as {
    repositories: Array<{ repository: string; revision: string }>;
  };

  it('pins the archived assignment and checksums its normative interfaces', () => {
    const locked = lock.repositories.find((row) => row.repository === 'assignment1-basics')?.revision;
    expect(contract.schema_version).toBe(1);
    expect(contract.source.repository).toBe('stanford-cs336/assignment1-basics');
    expect(contract.source.commit).toBe(locked);

    const handout = resolve(assignmentRoot, contract.source.handout.path);
    const adapters = resolve(assignmentRoot, contract.source.adapters.path);
    expect(sha256(readFileSync(handout))).toBe(contract.source.handout.sha256);
    expect(sha256(readFileSync(adapters))).toBe(contract.source.adapters.sha256);
    const treeFiles = testTreeFiles();
    expect(contract.source.tests.tree_hash_algorithm).toBe('sha256(sorted relative path + NUL + bytes + NUL)');
    expect(contract.source.tests.tree_includes).toBe('all regular files under tests/** recursively');
    expect(contract.source.tests.tree_excludes).toEqual([
      '**/__pycache__/**', '**/.pytest_cache/**', '**/*.pyc', '**/.DS_Store'
    ]);
    expect(treeFiles).toContain('tests/conftest.py');
    expect(treeFiles).toContain('tests/common.py');
    expect(treeFiles.some((file) => file.startsWith('tests/_snapshots/'))).toBe(true);
    expect(treeFiles.some((file) => file.startsWith('tests/fixtures/'))).toBe(true);
    expect(testTreeSha256(treeFiles)).toBe(contract.source.tests.tree_sha256);
  });

  it('enumerates every official adapter and test function exactly once', () => {
    const actualAdapters = functionNames(
      resolve(assignmentRoot, 'tests/adapters.py'),
      /^def ((?:run|get)_[A-Za-z0-9_]+)\(/gm
    );
    const contractAdapters = [...contract.official_interfaces.adapters].sort();
    expect(contractAdapters).toHaveLength(new Set(contractAdapters).size);
    expect(contractAdapters).toEqual(actualAdapters);

    const actualTests = testFiles().flatMap((file) =>
      functionNames(resolve(assignmentRoot, 'tests', file), /^def (test_[A-Za-z0-9_]+)\(/gm)
        .map((name) => `${file}::${name}`)
    ).sort();
    const contractTests = contract.official_interfaces.tests.flatMap((entry) =>
      entry.names.map((name) => `${entry.file}::${name}`)
    ).sort();
    expect(contractTests).toHaveLength(new Set(contractTests).size);
    expect(contractTests).toEqual(actualTests);
    expect(contractTests).toHaveLength(48);
  });

  it('defines executable local/full gates and evidence-backed pass/fail criteria', () => {
    expect(Object.keys(contract.profiles).sort()).toEqual(['cpu_local', 'gpu_full']);
    expect(contract.profiles.cpu_local.required_commands).toContain('uv run pytest');
    expect(contract.profiles.gpu_full.required_commands).toContain('uv run pytest');
    expect(contract.profiles.gpu_full.total_tokens).toBe(327_680_000);
    expect(contract.profiles.cpu_local.reduced_total_tokens).toBe(40_960_000);
    expect(contract.acceptance.cpu_local.validation_loss_max).toBe(2.0);
    expect(contract.acceptance.gpu_full.validation_loss_max).toBe(1.45);
    expect(contract.acceptance.gpu_full.test_policy).toBe('48 required tests pass; no skip or xfail');

    const requiredEvidence = new Set(contract.evidence.required_fields);
    for (const field of [
      'source_commit', 'student_commit', 'hardware', 'software_versions', 'command',
      'config', 'dataset_checksum', 'tokenizer_checksum', 'seed', 'dtype', 'shapes',
      'total_tokens', 'test_summary', 'loss_curve', 'tokens_per_second', 'wall_time_seconds',
      'peak_memory_bytes', 'checkpoint_checksum', 'generation_sample'
    ]) expect(requiredEvidence.has(field), `missing evidence field ${field}`).toBe(true);
    expect(contract.evidence.event_types.sort()).toEqual(
      ['checkpoint', 'environment', 'evaluation', 'generation', 'test_run', 'training_run']
    );
    expect(contract.boundaries.official_training_cli).toBe('not specified by Assignment 1');
    expect(contract.boundaries.solution_code).toBe('not included');
  });

  it('links the human capstone to the exact contract without embedding a solution', () => {
    const page = readFileSync(practicePath, 'utf8');
    expect(page).toContain('stanford-cs336-a1.yml');
    expect(page).toContain(contract.source.commit);
    expect(page).toContain('uv run pytest');
    expect(page).not.toMatch(/^def\s+(?:run_|train_|forward)/m);
  });
});
