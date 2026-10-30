"""Deterministic CPU checks for Bookvar 41a. No external dataset or PII."""
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from fractions import Fraction
from hashlib import sha256
from io import StringIO
from itertools import permutations
from math import isclose, sqrt
from pathlib import Path
import ast
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REL = Path("00 Учебник/11 Pre-training и Scaling/41a Дедупликация, PII и контроль качества корпуса.md")
text = (ROOT / REL).read_text()
blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
assert len(blocks) == 5  # Three image examples are executed by check_selection_gap_examples.py.
ns = {}
exec(compile(blocks[0], str(REL), "exec"), ns)
dedup = ns["exact_document_dedup"]
records = [("c", "Разный текст"), ("b", "текст  статьи"), ("a", "Текст статьи")]
for ordering in permutations(records):
    kept, duplicates = dedup(list(ordering), key_fn=lambda _: 0)
    assert kept == [("a", "Текст статьи"), ("c", "Разный текст")]
    assert duplicates == {"b": "a"}
    assert dedup(list(ordering)) == (kept, duplicates)
normal = ns["normalize_for_match"]
assert normal("cafe\u0301") == normal("café")
assert normal("a-b") != normal("ab")
assert normal("1.5") != normal("15")
assert dedup([]) == ([], {})
print("PASS actual chapter code: stable representatives, forced hash collision, Unicode, empty input")

lines = ["меню", "статья A", "меню", "статья B"]
assert list(dict.fromkeys(lines)) == ["меню", "статья A", "статья B"]
counts = Counter(lines)
assert [s for s in lines if counts[s] == 1] == ["статья A", "статья B"]
M = 10**9
assert isclose(M*(M-1)/(2*2**64), 0.027105054285, rel_tol=1e-9)
assert isclose(M*(M-1)/(2*2**32), 116415321.7105195)
print("PASS exact-line remove-all semantics and 32/64-bit collision estimates")

word = "banana"
positions = sorted(range(len(word)), key=lambda i: word[i:])
suffixes = [word[i:] for i in positions]
lcps = [0]
for previous, current in zip(suffixes, suffixes[1:]):
    length = 0
    while length < min(len(previous), len(current)) and previous[length] == current[length]:
        length += 1
    lcps.append(length)
printed_suffixes = re.findall(r"\|\s*(\d+)\s*\| `([a-z]+)` \|\s*(\d+)\s*\|", text)
assert printed_suffixes == [(str(p + 1), s, str(lcp))
                            for p, s, lcp in zip(positions, suffixes, lcps)]
assert [p + 1 for p in positions] == [6, 4, 2, 1, 5, 3]
assert lcps == [0, 1, 3, 0, 0, 2]
print("PASS every printed suffix-array row and LCP value, including overlapping ana occurrences")

def shingles(s, n):
    words = normal(s).split()
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}

def jaccard(a, b):
    if not a and not b:
        raise ValueError("Empty sets require an explicit separate policy")
    return len(a & b) / len(a | b)

assert jaccard(shingles("кот сидит на ковре", 2), shingles("кот спит на ковре", 2)) == 0.2
assert not shingles("один", 5)
try:
    jaccard(set(), set())
except ValueError:
    pass
else:
    raise AssertionError("Empty-set policy was ignored")
A, B = {1, 2, 3, 4}, {1, 2, 3, 5}
agree = 0
for perm in permutations(A | B):
    first_a = next(x for x in perm if x in A)
    first_b = next(x for x in perm if x in B)
    agree += first_a == first_b
assert agree == 72 and agree / 120 == jaccard(A, B) == .6
assert isclose(sqrt(.6*.4/100), .04898979485566356)
print("PASS shingling, empty-set guard, exact MinHash probability over all 120 permutations")

def candidate(sig_a, sig_b, r):
    assert len(sig_a) == len(sig_b) and len(sig_a) % r == 0
    return any(sig_a[i:i+r] == sig_b[i:i+r] for i in range(0, len(sig_a), r))

sig_a, sig_b = [1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 1, 2]
assert candidate(sig_a, sig_b, 2)
assert not candidate([1, 2, 3, 4], [3, 4, 1, 2], 2)  # same values, wrong positions
assert candidate(sig_a, sig_b, 2) and not (jaccard(A, B) >= .8)
expected_percent = [(0.3, .002952415132340125, 2.403599406790169),
                    (0.6, 2.986967621952663, 55.49184949228081),
                    (0.8, 43.3307820411204, 98.11305037025156),
                    (0.9, 88.27865978466163, 99.98673689785023)]
for s, p1, p2 in expected_percent:
    assert isclose(100 * (1-(1-s**10)**5), p1, rel_tol=1e-12)
    assert isclose(100 * (1-(1-s**5)**10), p2, rel_tol=1e-12)
assert isclose(1-(1-1/5)**5, .67232)
print("PASS LSH band positions, candidate verification, every probability table entry")

pipeline = {}
pipeline_stdout = StringIO()
with redirect_stdout(pipeline_stdout):
    exec(compile(blocks[1], str(REL) + ":pipeline", "exec"), pipeline)
expected_signatures = {
    "A": (1, 2, 1, 2, 2, 2), "B": (1, 2, 1, 1, 2, 2),
    "C": (2, 2, 1, 1, 2, 2), "D": (5, 1, 5, 5, 1, 1),
}
assert pipeline["signatures"] == expected_signatures
assert pipeline["candidates"] == {("A", "B"), ("A", "C"), ("B", "C")}
assert pipeline["verified"] == [("A", "B"), ("B", "C")]
assert pipeline["buckets"][(2, (2, 2))] == ["A", "B", "C"]
assert pipeline["buckets"][(1, (1, 1))] == ["B", "C"]
assert pipeline["buckets"][(2, (1, 1))] == ["D"]
printed_bands = re.findall(r"\| \$([ABCD])\$ \| `\((\d), (\d)\)` \| `\((\d), (\d)\)` \| `\((\d), (\d)\)` \|", text)
assert len(printed_bands) == 4
for name, *numbers in printed_bands:
    assert tuple(map(int, numbers)) == expected_signatures[name]
try:
    pipeline["signature"](set())
except ValueError:
    pass
else:
    raise AssertionError("Actual signature() must reject empty sets")
assert pipeline_stdout.getvalue().splitlines()[-2:] == [
    "[('A', 'B'), ('A', 'C'), ('B', 'C')]", "[('A', 'B'), ('B', 'C')]",
]
print("PASS actual end-to-end Markdown code, all signatures/bands, three candidates and two verified edges")

paper_probability = 1 - (1 - Fraction(4, 5) ** 20) ** 450
swapped_probability = 1 - (1 - Fraction(4, 5) ** 450) ** 20
assert isclose(float(paper_probability), .9945833962871292, rel_tol=1e-14)
assert isclose(float(swapped_probability), 4.915007015206306e-43, rel_tol=1e-14)
assert isclose(float(1 - (1 - Fraction(1, 2) ** 5) ** 25), .5478392215509674)
assert 25 * 5 == 125 < 128
assert Fraction(1) - Fraction(1, 10) == Fraction(9, 10)
assert Fraction(1) - Fraction(1, 5) == Fraction(4, 5)
print("PASS NearDup parameter mapping via exact fractions, soft threshold and strict edit-similarity boundary")

archive = ROOT / "05 Источники/Originals/corpus-dedup-2026-10-23"
source_ast = ast.parse((archive / "datasketch-lsh.py").read_text())
source_class = next(n for n in source_ast.body if isinstance(n, ast.ClassDef) and n.name == "MinHashLSH")
query = next(n for n in source_class.body if isinstance(n, ast.FunctionDef) and n.name == "query")
docstring = ast.get_docstring(query)
docstring = re.sub(r":\w+:`([^`]+)`", r"\1", docstring)
docstring = " ".join(docstring.split())
quote = next(line[2:] for line in text.splitlines() if line.startswith("> Results are based on minhash"))
assert " ".join(quote.replace("`", "").split()) in docstring
image_rel = Path("00 Учебник/Assets/Figures/curated/corpus-dedup-2026-10-23/lsh-characteristics.png")
assert (ROOT / image_rel).read_bytes() == (archive / "lsh-characteristics.png").read_bytes()
assert sha256((ROOT / image_rel).read_bytes()).hexdigest() == "3fee00876bfd50c407932722c423ef6badce2c60e6dbf030f8b435b848a405c6"
assert (ROOT / "site/public/licenses/datasketch-MIT.txt").read_bytes() == (archive / "datasketch-LICENSE.txt").read_bytes()
print("PASS original English excerpt, unchanged author figure and complete MIT license")

sets = {"A": {1, 2, 3}, "B": {1, 2, 3, 4}, "C": {2, 3, 4}}
assert jaccard(sets["A"], sets["B"]) == .75
assert jaccard(sets["B"], sets["C"]) == .75
assert jaccard(sets["A"], sets["C"]) == .5
edges = {(a, b) for a in sets for b in sets if a < b and jaccard(sets[a], sets[b]) >= .7}
assert edges == {("A", "B"), ("B", "C")}
representatives = []
for item in sorted(sets):
    if not any(jaccard(sets[item], sets[rep]) >= .7 for rep in representatives):
        representatives.append(item)
assert representatives == ["A", "C"]
print("PASS nontransitivity: one connected component but two direct-match representatives")

# This is only a finite, fictitious fixture for the transformation contract,
# not a PII recognizer quality test and not production email detection.
def fixture_mask(s):
    return re.subn(r"reader@example\.com", "|||EMAIL_ADDRESS|||", s)

masked, n = fixture_mask("Пишите на reader@example.com; |||EMAIL_ADDRESS|||")
assert n == 1 and fixture_mask(masked) == (masked, 0)
assert sum(Counter(["retained", "modified", "dropped", "quarantined"]).values()) == 4
print("PASS masking idempotence contract on a fictitious fixture; document balance")

group_sizes = [900, 50, 40, 10]
sample_errors = [1, 2, 3, 0]
weighted = sum(n * Fraction(e, 10) for n, e in zip(group_sizes, sample_errors)) / sum(group_sizes)
unweighted = Fraction(sum(sample_errors), 40)
assert weighted == Fraction(14, 125) and unweighted == Fraction(3, 20)
print("PASS stratified-audit weights: 11.2 percent vs incorrect unweighted 15 percent")

old = subprocess.check_output(["git", "show", "HEAD:" + str(REL)], cwd=ROOT, text=True)
old_ids = set(re.findall(r"<!-- source_unit_id: (.*?) -->", old))
new_ids = set(re.findall(r"<!-- source_unit_id: (.*?) -->", text))
assert old_ids == new_ids, (old_ids - new_ids, new_ids - old_ids)
old_anchors = set(re.findall(r'<a id="([^"]+)"', old))
new_anchors = set(re.findall(r'<a id="([^"]+)"', text))
assert old_anchors <= new_anchors
image_stems = re.findall(r"!\[\[02 Areas/ML & DL/(.*?)\]\]", text)
assert len(image_stems) == 4
for stem in image_stems:
    assert (ROOT / stem).is_file(), stem
assert "42 Next-token prediction|" in text
assert "minhash-agreement-trace.png" not in text
print(f"PASS {len(old_ids)} source-unit markers, original anchors, 4 figure paths, next chapter")
print("Chapter SHA-256:", sha256((ROOT / REL).read_bytes()).hexdigest())
print("All checks passed. CPU only; no corpus-scale quality or privacy claim.")
