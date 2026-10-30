#!/usr/bin/env python3
"""Run the chapter's actual Dynamo frontend examples on CPU.

The recording backends return GraphModule.forward, not optimized kernels.
This check makes no claim about Inductor, AOTAutograd, CUDA or performance.
"""

import ast
import hashlib
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/10 ML Systems/08 GPU kernels и Triton — от программы к измерению.md"
text = CHAPTER.read_text()
section = text.split("<!-- BEGIN REVIEW: torch.compile 2026-10-26 -->")[1].split(
    "<!-- END REVIEW: torch.compile 2026-10-26 -->"
)[0]
blocks = re.findall(r"```python\n(.*?)\n```", section, re.S)
assert len(blocks) == 3

scopes = []
for index, code in enumerate(blocks, start=1):
    ast.parse(code)
    scope = {}
    exec(compile(code, str(CHAPTER) + f":Dynamo-example-{index}", "exec"), scope)
    scopes.append(scope)

assert len(scopes[0]["compile_records"]) == 3
assert len(scopes[1]["piece_records"]) == 2
assert len(scopes[2]["dynamic_records"]) == 1
observed = re.search(
    r"<!-- BEGIN OBSERVED: first Dynamo FX function -->\s*```text\n(.*?)\n```",
    section, re.S,
).group(1)
assert scopes[0]["compile_records"][0].strip() == observed.strip()

print("Three actual Markdown examples executed on CPU")
print("Static guards: 5 calls / 3 backend invocations, including old-cache reuse")
print("Printed FX function matches the actual captured GraphModule code")
print("Explicit graph break: 2 cached regions; fullgraph rejects the break")
print("Dynamic shapes plus tensor scalar: 3 calls / 1 backend invocation")
print("torch", torch.__version__, "; CUDA available:", torch.cuda.is_available())
print("Chapter SHA-256:", hashlib.sha256(CHAPTER.read_bytes()).hexdigest())
print("No Inductor/AOTAutograd code generation, CUDA execution or speedup claim.")
