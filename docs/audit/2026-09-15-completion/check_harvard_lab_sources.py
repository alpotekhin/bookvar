"""Verify the archived lab inventory against one pinned upstream commit.

Read-only: downloads source into memory, checks provenance/body/syntax, never
executes the downloaded code. This is not a Marimo runtime or visual test.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "45ecc8d82fcae70c149cdce550d3b3d3411df913"
BASE = f"https://raw.githubusercontent.com/harvard-edge/cs249r_book/{COMMIT}/"
manifest = json.loads((ROOT / "05 Источники/Courses/Harvard ML Systems/labs-slides-manifest.json").read_text())
labs = [item for item in manifest if item["kind"] == "lab"]
assert len(labs) == 34
assert len({item["source"] for item in labs}) == 34
tree_url = f"https://api.github.com/repos/harvard-edge/cs249r_book/git/trees/{COMMIT}?recursive=1"
with urllib.request.urlopen(tree_url, timeout=30) as response:
    tree = json.load(response)
assert not tree.get("truncated"), "Upstream tree is incomplete"
upstream_paths = {entry["path"] for entry in tree["tree"]
                  if entry["type"] == "blob"
                  and re.fullmatch(r"labs/vol[12]/lab_[^/]+\.py", entry["path"])}
assert {item["source"] for item in labs} == upstream_paths


def check(item):
    # Honour the caller's proxy environment; source remains unexecuted data.
    with urllib.request.urlopen(BASE + item["source"], timeout=30) as response:
        raw = response.read()
    expected = raw.decode("utf-8")
    markdown = (ROOT / item["output"]).read_text()
    section = markdown.split("## Complete original Marimo source\n", 1)[1]
    match = re.fullmatch(r"\s*```python\n(.*)\n```\s*", section, re.S)
    assert match, item["output"]
    body = match.group(1)
    assert body == expected.rstrip(), item["source"]
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == item["sha256"], item["source"]
    assert f"source_commit: {COMMIT}" in markdown
    compile(body, item["source"], "exec")
    return {"source": item["source"], "sha256": digest, "upstream_hash": "match",
            "archived_body": "match-except-final-whitespace", "python_syntax": "pass"}


with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(check, labs))
print(json.dumps({"commit": COMMIT, "count": len(results), "upstream_inventory": "exact-match", "results": results,
                  "runtime": "not-run", "visual": "not-reviewed"}, indent=2))
