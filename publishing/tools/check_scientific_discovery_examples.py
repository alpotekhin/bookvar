#!/usr/bin/env python3
"""Check chapter 71's small examples, not the LaSR/ESCHER benchmarks."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/17 Tools и Agents/71 Агенты научного поиска и discovery.md"
ARCHIVE = ROOT / "05 Источники/Originals/scientific-discovery-2026-10-21"


def main():
    text = CHAPTER.read_text()
    code = re.findall(r"^```python\n(.*?)^```\s*$", text, re.M | re.S)
    assert len(code) == 3, len(code)
    with tempfile.TemporaryDirectory(prefix="bookvar-discovery-") as tmp:
        for index, block in enumerate(code):
            path = Path(tmp) / f"example{index}.py"
            path.write_text(block)
            subprocess.run([sys.executable, "-I", str(path)], check=True, timeout=10)

    # Independent rational checks for the worked explanations.
    assert F(2, 5) + F(1, 20) == F(9, 20)  # A's training criterion.
    assert F(0) + F(1, 20) * 2 == F(1, 10)  # B's criterion.
    assert (F(7, 10) + F(4, 10)) / 2 == F(55, 100)
    assert (F(7, 10) + F(6, 10)) / 2 == F(65, 100)
    assert (F(7, 10) + F(4, 10) + F(95, 100)) / 3 == F(41, 60)
    differences = [abs(F(18, 5) / r - 4 / F(r)**2) for r in [1, 2, 4, 8]]
    assert differences == [F(2, 5), F(4, 5), F(13, 20), F(31, 80)]
    assert [F(8317-6326, 100), F(9490-9477, 100),
            F(7362-7423, 100), F(6333-6200, 100)] == [F(1991, 100), F(13, 100), F(-61, 100), F(133, 100)]
    assert 43049 + 10763 == 53812
    delta = F(19181636, 10**12)
    assert abs(1 / (1 + delta) - 1) < F(2, 10**5)
    assert F(34, 100) < 1 / (1 + delta / F(1, 10**5)) < F(35, 100)

    refs = json.loads((ARCHIVE / "references.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(CHAPTER) == refs["chapter_sha256"]
    for row in refs["sources"]:
        assert sha(ROOT / row["file"]) == row["sha256"], row["file"]
    figures = json.loads((ROOT / "00 Учебник/Assets/Figures/curated/scientific-discovery-2026-10-21/assets.json").read_text())
    for row in figures:
        assert sha(ROOT / row["asset"]) == row["sha256"]
        assert (ROOT / row["asset"]).read_bytes() == (ROOT / row["source_asset"]).read_bytes()
    print("PASS: 3 extracted Python examples; rational toy calculations; data counts; source and figure hashes. No LaSR/ESCHER training, search benchmark or real CLIP measurement.")


if __name__ == "__main__":
    main()
