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
    blocks = re.findall(r"^~~~python\n(.*?)^~~~", text, re.M | re.S)
    assert len(blocks) == 3, len(blocks)
    # Execute only the chapter-authored snippets, never downloaded source files.
    with tempfile.TemporaryDirectory(prefix="bookvar-security-") as tmp:
        for index, code in enumerate(blocks):
            path = Path(tmp) / f"example{index}.py"
            path.write_text(code)
            subprocess.run([sys.executable, "-I", str(path)], check=True, timeout=10)

    # Exact arithmetic for the synthetic retrieval example.
    query = (F(1), F(0))
    keys = {"A": (F(4, 5), F(3, 5)), "B": (F(3, 5), F(4, 5))}
    assert all(sum(v*v for v in key) == 1 for key in keys.values())
    score = lambda item: sum(a*b for a, b in zip(query, item[1]))
    assert [name for name, _ in sorted(keys.items(), key=score, reverse=True)[:2]] == ["A", "B"]
    keys["P"] = (F(1), F(0))
    top = [name for name, _ in sorted(keys.items(), key=score, reverse=True)[:2]]
    assert top == ["P", "A"]
    assert not all(name == "P" for name in top)  # Not AgentPoison's ASR-r success.

    # Synthetic detector stream: base rate, recall, false alarms and precision.
    attacks, clean = 100, 9900
    true_positive = F(9, 10) * attacks
    false_positive = F(1, 100) * clean
    assert (true_positive, false_positive) == (90, 99)
    assert attacks - true_positive == 10
    assert true_positive / (true_positive + false_positive) == F(10, 21)

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    refs = json.loads((ARCHIVE / "references.json").read_text())
    assert sha(CHAPTER) == refs["chapter_sha256"], "chapter hash is stale"
    for row in refs["sources"] + [refs["course_pdf"]]:
        assert sha(ROOT / row["file"]) == row["sha256"], row["file"]
    assets = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/agent-security-2026-10-21/assets.json").read_text())
    for row in assets:
        assert sha(ROOT / row["asset"]) == row["sha256"]
        if row["derivation"] in ("source-image-copy", "source-svg-copy"):
            assert (ROOT / row["asset"]).read_bytes() == (ROOT / row["source_asset"]).read_bytes()
    print("PASS: 3 extracted examples; synthetic retrieval and detector arithmetic; source/figure hashes. No upstream agents, attack search, detector training or benchmarks executed.")


if __name__ == "__main__":
    main()
