"""Execute all seven new reader programs and independent source comparisons.

CPU arithmetic, not CLIP training, TorchRec execution or a performance benchmark.
"""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import re
import time
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
source_dir = ROOT / "05 Источники/Originals/image-id-selection-2026-10-29"
journal = json.loads((ROOT / "docs/audit/2026-10-29-selection-gaps-additions.json").read_text())
scopes = {}
for row in journal["files"]:
    text = (ROOT / row["source"]).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == row["after_sha256"]
    block = re.search(r"<!-- " + row["marker"] + r"-addition: start -->.*?<!-- "
                      + row["marker"] + r"-addition: end -->\n\n", text, re.S).group()
    assert hashlib.sha256(block.encode()).hexdigest() == row["block_sha256"]
    codes = re.findall(r"```python\n(.*?)\n```", block, re.S)
    assert len(codes) == row["new_programs"]
    scopes[row["marker"]] = []
    for i, code in enumerate(codes):
        namespace = {}
        exec(compile(code, row["source"] + f":addition{i+1}", "exec"), namespace)
        scopes[row["marker"]].append(namespace)

# Execute only the two original hashing functions, not a downloaded launcher.
original_hashing = ast.parse((source_dir / "imagehash.py").read_text())
names = {"phash", "dhash"}
functions = [n for n in original_hashing.body if isinstance(n, ast.FunctionDef) and n.name in names]
assert len(functions) == 2
source_hash_scope = {"numpy": np, "ANTIALIAS": Image.Resampling.LANCZOS,
                     "ImageHash": lambda bits: bits}
exec(compile(ast.Module(body=functions, type_ignores=[]), "ImageHash pinned functions", "exec"), source_hash_scope)
rng = np.random.default_rng(20261029)
for _ in range(30):
    image = Image.fromarray(rng.integers(0, 256, size=(31, 37, 3), dtype=np.uint8))
    np.testing.assert_array_equal(source_hash_scope["phash"](image),
        scopes["image-dedup"][1]["perceptual_bits"](image))
    np.testing.assert_array_equal(source_hash_scope["dhash"](image, hash_size=4),
        scopes["image-dedup"][0]["difference_bits"](image))

# The complete pinned similarity method runs on the assigned vectors, on CPU.
source_tree = ast.parse((source_dir / "semdedup.py").read_text())
cls = next(n for n in source_tree.body if isinstance(n, ast.ClassDef))
methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
           and n.name in {"_contains_duplicates", "semdedup"}]
source_scope = {"torch": torch, "np": np, "time": time}
exec(compile(ast.Module(body=methods, type_ignores=[]), "SemDeDup pinned methods", "exec"), source_scope)
adapter = type("PinnedSimilarity", (), {name: source_scope[name]
    for name in ["_contains_duplicates", "semdedup"]})()
vectors = scopes["image-dedup"][2]["unit_vectors"]
cluster = np.array([["A"], ["B"], ["C"]])
maximum = adapter.semdedup(cluster, torch.tensor(vectors), "cpu").numpy()
np.testing.assert_array_equal(maximum > .98, [False, True, True])
previous = np.zeros(3, dtype=bool)
for epsilon in [0., .001, .02, .07, .2, .9]:
    current = maximum > 1 - epsilon
    assert np.all(~previous | current)  # Larger epsilon cannot undo a removal.
    previous = current
assert not (np.array([.98]) > 1 - .02)[0]

# Retry deduplication preserves distinct events and rejects inconsistent retries.
event_scope = scopes["cold-ids"][0]
expected = event_scope["clean_events"]
for _ in range(30):
    permuted = [event_scope["events"][i] for i in rng.permutation(10)]
    assert event_scope["deduplicate_events"](permuted) == expected
try:
    event_scope["deduplicate_events"]([("e", "u", 10), ("e", "u", 20)])
except ValueError:
    pass
else:
    raise AssertionError("Conflicting retry accepted")

# Every permutation of retained IDs must move all three arrays consistently.
state = scopes["cold-ids"][1]
for retained in itertools.permutations([10, 20, 30]):
    selected = [0] + [state["old_rows"][raw] for raw in retained]
    for old in [state["old_weights"], state["old_m"], state["old_v"]]:
        new = old[selected].copy()
        for index, raw in enumerate(retained, 1):
            np.testing.assert_array_equal(new[index], old[state["old_rows"][raw]])
        assert not np.shares_memory(old, new)
# A weights-only remap cannot preserve the identity of optimizer moments.
assert not np.array_equal(state["new_m"][state["new_rows"][10]], state["old_m"][2])
assert 8 * 4 * 4 == 128
assert np.isclose((336 - 144) / 336, 4 / 7)
assert (18/30, 18/20) == (.6, .9)
assert 17/18 > 18/30 and 17/20 < 18/20
print(json.dumps({"actual_new_programs": 7, "source_hash_comparisons": 60,
    "pinned_semdedup_method_executed": True, "event_permutations": 30,
    "retained_id_permutations": 6, "clip_inference": False,
    "torchrec_execution": False, "benchmark_reproduced": False}, indent=2))
