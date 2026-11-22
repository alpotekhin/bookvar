"""Check the chapter's isolated teaching examples, arithmetic and source hashes."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/72 Безопасность агентных систем.md"
ARCHIVE = ROOT / "05 Источники/Originals/agent-security-2026-10-21"


def main():
    text = CHAPTER.read_text()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit = ROOT / "docs/audit"
    revised = json.loads((audit / "2026-11-22-agent-security-ru-revision.json").read_text())["files"][0]
    archived = json.loads((audit / "2026-11-09-agent-security-bound-approval-additions.json").read_text())["files"]
    original = next(row for row in archived if row["source"] == CHAPTER.relative_to(ROOT).as_posix())
    assert revised["source"] == original["source"]
    assert revised["before_sha256"] == original["after_sha256"]
    assert sha(CHAPTER) == revised["after_sha256"], "chapter hash is stale"
    blocks = re.findall(r"^~~~python\n(.*?)^~~~", text, re.M | re.S)
    assert len(blocks) == 4, len(blocks)
    # Execute only the chapter-authored snippets, never downloaded source files.
    with tempfile.TemporaryDirectory(prefix="bookvar-security-") as tmp:
        for index, code in enumerate(blocks):
            path = Path(tmp) / f"example{index}.py"
            path.write_text(code)
            subprocess.run([sys.executable, "-I", str(path)], check=True, timeout=10)

    # Synthetic detector stream: base rate, recall, false alarms and precision.
    attacks, clean = 100, 9900
    true_positive = F(9, 10) * attacks
    false_positive = F(1, 100) * clean
    assert (true_positive, false_positive) == (90, 99)
    assert attacks - true_positive == 10
    assert true_positive / (true_positive + false_positive) == F(10, 21)

    refs = json.loads((ARCHIVE / "references.json").read_text())
    assert original["before_sha256"] == refs["chapter_sha256"], "archived chapter hash is stale"
    for row in refs["sources"] + [refs["course_pdf"]]:
        assert sha(ROOT / row["file"]) == row["sha256"], row["file"]
    assets = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/agent-security-2026-10-21/assets.json").read_text())
    for row in assets:
        assert sha(ROOT / row["asset"]) == row["sha256"]
        if row["derivation"] in ("source-image-copy", "source-svg-copy"):
            assert (ROOT / row["asset"]).read_bytes() == (ROOT / row["source_asset"]).read_bytes()
    print("PASS: 4 defensive examples; detector arithmetic; chapter, source and figure hashes. No remote targets, agent or detector benchmarks executed.")


if __name__ == "__main__":
    main()
