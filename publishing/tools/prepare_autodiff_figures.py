"""Render complete source objects, including captions, from the pinned Baydin PDF.

No redrawing, translated labels or compositing. --out is required so verification
can render outside the publication tree and compare pixels before accepting assets.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/frameworks-2026-10-27/baydin-jmlr-17-468.pdf"
SOURCE_SHA = "93eb2f7025ea098c18a65b428ac991dfacec2de56618c4e25c6a22b9cf443c59"
DPI = 400
# Full-page pixels at 400 dpi: 3400 x 4400. Each crop includes all source
# labels, arrows and the complete original caption, with white margins.
FIGURES = (
    ("baydin-figure4-graph.png", 9, (464, 513, 2940, 1528)),
    ("baydin-table2-forward.png", 10, (464, 547, 2940, 1784)),
    ("baydin-table3-reverse.png", 13, (464, 547, 2940, 2090)),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("Source PDF changed; inspect before rendering")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bookvar-autodiff-", dir="/var/tmp") as temp:
        for filename, page, box in FIGURES:
            prefix = Path(temp) / f"page-{page}"
            subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page),
                            "-r", str(DPI), "-singlefile", "-png",
                            str(SOURCE), str(prefix)], check=True)
            with Image.open(prefix.with_suffix(".png")) as full:
                assert full.size == (3400, 4400), full.size
                crop = full.crop(box)
                target = args.out / filename
                crop.save(target, dpi=(DPI, DPI))
                print(filename, "page", page, "crop", box, "size", crop.size,
                      "sha256", hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
