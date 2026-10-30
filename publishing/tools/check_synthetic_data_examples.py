"""Reproduce the chapter's parser example and arithmetic, without an LLM.

uv run --offline --with text-unidecode==1.3 python publishing/tools/check_synthetic_data_examples.py
Only the 11 upstream CLI tests run; not Docker, all 81 tests, or model training.
The four parser variants execute in memory; saved originals are not modified.
"""
from __future__ import annotations

import contextlib
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/53 Синтетические данные и учебные программы.md"
ARCHIVE = ROOT / "05 Источники/Originals/synthetic-data-2026-10-25"
PACKAGE = ARCHIVE / "python-slugify"
HASHES = {
    "slugify/__init__.py": "43ef5b28242ff3db9fddb26bff21f130f84159737c602ce5c50c0afe476979f2",
    "slugify/__main__.py": "d64bfc035e5383fdfe96bc11495b87a18b80f3fca4beb6b73a1080d31628d5c6",
    "slugify/__version__.py": "0e1df2e0c9d602334919b6a8466ddd7e2cadc85fa20f977edce09f1b297ffd8f",
    "slugify/slugify.py": "3103ecc34bb68362d4fbc0414fb2b40ce946b40848c68287403e1f5a5b4a9656",
    "slugify/special.py": "f1b1c03c69d9fb591eb96e31d9665d15755bf4b1684bb8bd268e2dcbfd83f210",
    "test.py": "96a2fb00cb9a225c473cd0bcfab01286f16d47b51368a986a250e00a50ddcc92",
    "LICENSE": "30ba4dc69a9f4dce132dd703937c7a93b573e2524604d2d5f92c5066514c0d4f",
}


def apply_hunk(source, patch):
    """Apply the saved single-file, single-hunk patch with exact context checks."""
    lines = patch.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("@@ ")]
    assert len(starts) == 1
    index = starts[0]
    header = re.fullmatch(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@.*\n", lines[index])
    assert header is not None
    before = source.splitlines(keepends=True)
    cursor = int(header[1]) - 1
    result, removed, added = before[:cursor], 0, 0
    for line in lines[index+1:]:
        assert line[0] in " +-"
        if line[0] in " -":
            assert before[cursor] == line[1:]
            cursor += 1
            removed += 1
        if line[0] in " +":
            result.append(line[1:])
            added += 1
    assert (removed, added) == (int(header[2]), int(header[4]))
    return "".join([*result, *before[cursor:]])


def main():
    sys.dont_write_bytecode = True
    for name, digest in HASHES.items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest, name
    record = json.loads((ARCHIVE / "swesmith-task.json").read_text())
    assert record["instance_id"] == "un33k__python-slugify.872b3750.func_basic__nis2b6y4"
    assert hashlib.sha256(record["patch"].encode()).hexdigest() == (
        "cd80f21f8aa8c6eb77aabbe215bfdaab66924a44010c1489cbabd65e405b92d3"
    )
    assert len(record["FAIL_TO_PASS"]) == 2 and len(record["PASS_TO_PASS"]) == 79
    text = CHAPTER.read_text()
    fences = [body for _, body in re.findall(r"^(```|~~~)python\n(.*?)^\1\s*$", text, re.M | re.S)]
    assert len(fences) == 2
    for index, body in enumerate(fences, 1):
        exec(compile(body, f"{CHAPTER}:python-{index}", "exec"), {})
    sys.path.insert(0, str(PACKAGE))
    clean = (PACKAGE / "slugify/__main__.py").read_text()
    mutant = apply_hunk(clean, record["patch"])
    partial = mutant.replace("SEP = '<-'", "SEP = '->'")
    repaired = partial.replace("split(SEP, 2)", "split(SEP, 1)")
    assert repaired == clean
    results = {}
    for name, source in [("clean", clean), ("mutated", mutant),
                         ("delimiter_only", partial), ("fully_repaired", repaired)]:
        scope = {"__name__": "slugify.chapter_check", "__package__": "slugify"}
        exec(compile(source, str(PACKAGE / "slugify/__main__.py"), "exec"), scope)
        tests = {"__name__": "bookvar_upstream_cli_tests"}
        exec(compile((PACKAGE / "test.py").read_text(), str(PACKAGE / "test.py"), "exec"), tests)
        tests.update(parse_args=scope["parse_args"], slugify_params=scope["slugify_params"])
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests["TestCommandParams"])
        with contextlib.redirect_stderr(io.StringIO()):
            result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        failures = sorted(test._testMethodName for test, _ in result.failures + result.errors)
        expected = ["test_replacements_right", "test_replacements_wrong"] if name == "mutated" else []
        assert result.testsRun == 11 and failures == expected
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                edge = scope["parse_args"](["slugify", "--replacements", "A->B->C", "--", "A cat"])
            except SystemExit as error:
                assert name == "mutated" and error.code == 2
                edge_parts = None
            else:
                edge_parts = edge.replacements
                expected_parts = [["A", "B", "C"]] if name == "delimiter_only" else [["A", "B->C"]]
                assert edge_parts == expected_parts
                if name == "delimiter_only":
                    try:
                        scope["slugify"](**scope["slugify_params"](edge))
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("three replacement parts unexpectedly accepted")
        results[name] = {"passed": 11-len(failures), "run": 11,
                         "failed_names": failures, "edge_replacements": edge_parts}
    command = re.findall(r"^~~~bash\n(.*?)^~~~\s*$", text, re.M | re.S)
    assert len(command) == 1
    argv = shlex.split(command[0])
    assert argv[:3] == ["python", "-m", "slugify"]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    executed = subprocess.run([sys.executable, *argv[1:]], cwd=PACKAGE,
                              env=env, text=True, capture_output=True, check=True)
    assert executed.stdout == "b-cat\n"
    malformed = shlex.split('python -m slugify --replacements "A->B" "test string"')
    rejected = subprocess.run([sys.executable, *malformed[1:]], cwd=PACKAGE,
                             env=env, text=True, capture_output=True)
    assert rejected.returncode == 2 and "ORIGINAL->REPLACED" in rejected.stderr
    # Independent exact arithmetic for the prose examples, not a training result.
    accepted = (800 * Fraction(9, 10), 200 * Fraction(1, 5))
    assert accepted == (720, 40) and accepted[1] / sum(accepted) == Fraction(1, 19)
    assert Fraction(1, 2)*900 / (Fraction(1, 2)*100 + Fraction(1, 2)*900) == Fraction(9, 10)
    available, budget = (10*10**12, 10*10**9), 10**12
    epochs = tuple(Fraction(budget, 2*size) for size in available)
    assert epochs == (Fraction(1, 20), 50)
    ratio, small_budget = Fraction(1, 1000), 10**9
    small_sources = tuple(size*ratio for size in available)
    assert small_sources == (10**10, 10**7)
    assert tuple(Fraction(small_budget, 2)/size for size in small_sources) == epochs
    assert Fraction(3*available[1], budget) == Fraction(3, 100)
    assert budget - 3*available[1] == 970*10**9
    schedule = [(7, 2, 1), (5, 3, 2), (3, 4, 3)]
    totals = tuple(map(sum, zip(*schedule)))
    assert totals == (15, 9, 6)
    assert tuple(Fraction(value, sum(totals)) for value in totals) == (
        Fraction(1, 2), Fraction(3, 10), Fraction(1, 5))
    assert math.isclose(float(Fraction(999, 1000)**1000), .368, abs_tol=.0005)
    assert 10_000*2000 == 20_000_000 and 20_000_000/2500 == 8000
    # Phi-4 Table 5: unique pools and sampled-token weights have different units.
    sources = [(15, 1300, 1.2), (15, 290, 5.2), (40, 290, 13.8),
               (20, 820, 2.4), (10, 580, 1.7)]
    assert sum(weight for weight, _, _ in sources) == 100
    epochs_phi4 = [Fraction(10_000 * weight, 100 * pool) for weight, pool, _ in sources]
    assert all(abs(float(actual) - reported) < .06
               for actual, (_, _, reported) in zip(epochs_phi4, sources))
    assert epochs_phi4[2] == Fraction(400, 29)
    # Swapped-position consistency selects candidate identities, not positions.
    first_order, swapped_order = ("A", "B"), ("B", "A")
    assert first_order[0] != swapped_order[0]
    assert first_order[0] == swapped_order[1]
    assert Fraction('59.0') - Fraction('53.4') == Fraction('5.6')
    for row in ['| DCLM | 53.4 | 57.0 |',
                '| Full Nemotron-CC | 53.0 | 57.8 |',
                '| Nemotron-CC-HQ | 59.0 | 60.1 |']:
        assert row in text
    print(json.dumps({"python_fences": len(fences), "bash_stdout": executed.stdout.strip(),
                      "upstream_cli_tests": results,
                      "chapter_sha256": hashlib.sha256(text.encode()).hexdigest()},
                     ensure_ascii=False, indent=2))
    print("PASS actual chapter commands, pinned upstream CLI tests and arithmetic; no LLM training")


if __name__ == "__main__":
    main()
