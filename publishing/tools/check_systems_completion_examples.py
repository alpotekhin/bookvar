#!/usr/bin/env python3
"""Execute the displayed systolic, pruning, NAS and exit examples on CPU.

No accelerator timing, trained-model quality or NAS search is measured.
"""
import hashlib
import itertools
import json
from pathlib import Path
import re
import numpy as np
from check_hardware_source_completion import before_hardware_source_completion

ROOT = Path(__file__).resolve().parents[2]
JOURNAL = ROOT / "docs/audit/2026-10-29-systems-completion-additions.json"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    journal = json.loads(JOURNAL.read_text())
    examples = {}
    for row in journal["files"]:
        text = before_hardware_source_completion(
            (ROOT / row["source"]).read_text(), row["source"])
        assert digest(text) == row["after_sha256"]
        before = text
        for block in row["blocks"]:
            pattern = (r"<!-- systems-completion: " + block["id"] + r":start -->"
                       r"[\s\S]*?<!-- systems-completion: " + block["id"] + r":end -->\n\n")
            matches = re.findall(pattern, text)
            assert len(matches) == 1 and digest(matches[0]) == block["sha256"]
            before = before.replace(matches[0], "", 1)
            for ident, code in re.findall(
                r"<!-- checked-example: ([^>]+) -->\n```python\n(.*?)\n```",
                matches[0], re.S
            ):
                ns = {"np": np}
                exec(compile(code, f"{row['source']}:{ident}", "exec"), ns)
                examples[ident] = ns
        for change in row["frontmatter_changes"]:
            before = before.replace(change["new"], change["old"], 1)
        assert digest(before) == row["before_sha256"]

    assert len(examples) == 8
    fn = examples["hardware-systolic"]["weight_stationary"]
    rng = np.random.default_rng(20261029)
    shapes = 0
    for m, k, n in itertools.product(range(1, 7), repeat=3):
        a = rng.integers(-9, 10, size=(m, k))
        b = rng.integers(-9, 10, size=(k, n))
        actual, active = fn(a, b)
        np.testing.assert_array_equal(actual, a @ b)
        assert len(active) == m+k+n-2
        assert sum(active) == m*k*n
        # Independently enumerate each MAC's meeting time.
        ticks = [0]*(m+k+n-2)
        for i, kk, j in itertools.product(range(m), range(k), range(n)):
            ticks[i+kk+j] += 1
        assert active == ticks and max(active) <= k*n
        shapes += 1

    # Enumerate all 256 masks for the early-exit batch instead of only
    # evaluating the formula p**8 used in the chapter.
    p = .7
    mean = 0.
    for mask in itertools.product((False, True), repeat=8):
        probability = p**sum(mask) * (1-p)**(8-sum(mask))
        cost = 2.2 if all(mask) else 8.2
        mean += probability*cost
    np.testing.assert_allclose(mean, examples["compression-early-exit"]["batch_mean"])
    for n in range(1, 9):
        for _ in range(30):
            a, b = rng.choice([-1, 1], (2, n))
            pack = lambda v: sum(int(x == 1) << i for i, x in enumerate(v))
            agreement = (~(pack(a)^pack(b)) & ((1 << n)-1)).bit_count()
            assert 2*agreement-n == int(a @ b)
    print(json.dumps({
        "displayed_programs": list(examples), "systolic_shapes": shapes,
        "early_exit_masks": 256, "binary_vectors": 240,
        "bounded_restorations": len(journal["files"]),
        "scope": "CPU arithmetic and exact text preservation; no hardware performance or accuracy claim"
    }, indent=2))


if __name__ == "__main__":
    main()
