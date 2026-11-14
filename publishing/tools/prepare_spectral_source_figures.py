"""Extract complete original Wiskott figures, or verify their exact source pixels."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/spectral-2026-10-25/wiskott-laplacian.pdf"
SOURCE_SHA = "cb2ff89dff43e1bdcd9d3f77dea73b2b8a926a805665b28ef07fa6f6f84933fc"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/spectral-2026-10-31"
CASES = [
    (5, "wiskott-connected-heat.png", (280, 980, 2280, 1860)),
    (15, "wiskott-weighted-constraint.png", (280, 300, 2280, 1390)),
]

if __name__ == "__main__":
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    check = "--check" in sys.argv
    if not check:
        TARGET.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bookvar-spectral-") as tmp:
        for page, name, box in CASES:
            prefix = Path(tmp) / f"page-{page}"
            subprocess.run([
                "pdftoppm", "-f", str(page), "-l", str(page), "-r", "300",
                "-singlefile", "-png", str(SOURCE), str(prefix),
            ], check=True, capture_output=True)
            original = Image.open(prefix.with_suffix(".png")).convert("RGBA")
            assert original.size == (2550, 3300)
            expected = original.crop(box)
            path = TARGET / name
            if not check:
                expected.save(path)
            actual = Image.open(path).convert("RGBA")
            assert actual.size == expected.size
            assert ImageChops.difference(expected, actual).getbbox(alpha_only=False) is None
            print(name, actual.size, hashlib.sha256(path.read_bytes()).hexdigest())
