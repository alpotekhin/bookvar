#!/usr/bin/env python3
"""Extract original GPT-1 figures and complete panels without redrawing."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "05 Источники/Originals/full-transformer-2026-10-21/gpt1.pdf"
OUT = ROOT / "00 Учебник/Assets/Figures/curated/source-audit-25-31"
FIGURES = [
    ("gpt1-figure1.png", 4, (500, 300, 2550, 1150)),
    ("gpt1-figure1-block.png", 4, (550, 340, 980, 1140)),
    ("gpt1-figure2-transfer.png", 7, (535, 2210, 1510, 3055)),
    ("gpt1-figure2-zero-shot.png", 7, (1525, 2210, 2485, 3055)),
]
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == "eb81a65e0856bd38e855e1f1de6ee12e7c3eb92ae05c8270d345a475beb631d0"
with tempfile.TemporaryDirectory() as tmp:
    for page_number in sorted({p for _, p, _ in FIGURES}):
        prefix = Path(tmp) / f"page-{page_number}"
        subprocess.run(["pdftoppm", "-f", str(page_number), "-l", str(page_number),
                        "-r", "360", "-singlefile", "-png", str(PDF), str(prefix)], check=True)
        with Image.open(str(prefix) + ".png") as page:
            assert page.size == (3060, 3960)
            for name, number, box in FIGURES:
                if number != page_number:
                    continue
                crop = page.crop(box)
                target = OUT / name
                if "--check" in sys.argv or target.exists():
                    with Image.open(target) as actual:
                        assert actual.size == crop.size
                        assert ImageChops.difference(actual.convert("RGB"), crop.convert("RGB")).getbbox() is None
                else:
                    crop.save(target)
                print(name, page_number, box, hashlib.sha256(target.read_bytes()).hexdigest())
print("PASS original PDF pixels, 360 dpi; complete source figure and labelled panels")
