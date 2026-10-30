#!/usr/bin/env python3
"""Check the actual scheduling examples, not CUDA/NCCL concurrency."""

import ast
import hashlib
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md"
text = CHAPTER.read_text()
section = text.split("<!-- BEGIN REVIEW: DDP overlap 2026-10-26 -->")[1].split(
    "<!-- END REVIEW: DDP overlap 2026-10-26 -->"
)[0]
blocks = re.findall(r"```python\n(.*?)\n```", section, re.S)
assert len(blocks) == 7  # two CPU models, three source CUDA snippets, two API/setup snippets
for code in blocks:
    ast.parse(code)

scope = {}
cpu_blocks = [code for code in blocks if "def ordered_bucket_schedule(" in code
              or "eager_two_buckets, _ =" in code]
assert len(cpu_blocks) == 2
for code in cpu_blocks:
    exec(compile(code, str(CHAPTER) + ":CPU-overlap-model", "exec"), scope)
schedule = scope["ordered_bucket_schedule"]

assert scope["intervals"] == [(3, 5), (9, 11), (11, 13), (13, 15)]
assert scope["tail"] == 5
assert schedule([[2, 4, 6, 8]], [2] * 4)[0][-1][1] == 10
assert scope["eager_two_buckets"][-1][1] == 11
assert scope["whole_backward"][-1][1] == 12
assert scope["split_backward"][-1][1] == 9

rng = random.Random(26)
for _ in range(1000):
    ranks, count = rng.randint(1, 4), rng.randint(1, 6)
    ready = [[rng.randint(0, 20) for _ in range(count)] for _ in range(ranks)]
    durations = [rng.randint(0, 5) for _ in range(count)]
    intervals, tail = schedule(ready, durations)
    global_ready = [max(row[i] for row in ready) for i in range(count)]
    for i, (start, finish) in enumerate(intervals):
        # Independent max-plus expansion of the queue recurrence.
        expected = max(global_ready[k] + sum(durations[k:i + 1]) for k in range(i + 1))
        assert finish == expected
        assert start == finish - durations[i]
    assert tail == max(0, intervals[-1][1] - max(map(max, ready)))

for ready, durations in [([], [1]), ([[1]], []), ([[1], [1, 2]], [1]), ([[-1]], [1]), ([[1]], [-1])]:
    try:
        schedule(ready, durations)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid scheduling model input was accepted")

# Check every value in the printed three-column example table.
rows = re.findall(r"\| (\d) \| (\d+) / (\d+) \| (\d+)–(\d+) \|", section)
assert rows == [("0", "2", "3", "3", "5"), ("1", "4", "9", "9", "11"),
                ("2", "6", "7", "11", "13"), ("3", "8", "10", "13", "15")]
print("PASS: two actual chapter CPU fences and all four printed schedule rows")
print("PASS: 1000 queue cases vs independent max-plus expansion and five invalid inputs")
print("PASS: readiness counterexample and compilation model 11 -> 12 -> 9")
print("Syntax only: all seven new Python fences; CUDA/NCCL/API/setup fences not executed")
print("Chapter SHA-256:", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
print("No CUDA/NCCL, DDPOptimizer run, GPU timing or physical-overlap claim.")
