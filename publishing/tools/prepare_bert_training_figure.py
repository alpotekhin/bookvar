#!/usr/bin/env python3
"""Extract complete BERT Figure 1 and its English caption from the ACL PDF."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/architecture-patterns-2026-10-21/bert-naacl.pdf"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/bert-family-2026-10-22/bert-figure1.png"
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == "987545ffb087f1ece898142c403a516baeabeb70ce19089397fac6f7db12c3d4"
box = (330, 290, 2660, 1600)
with tempfile.TemporaryDirectory() as tmp:
    prefix = Path(tmp) / "page"
    subprocess.run(["pdftoppm", "-f", "3", "-l", "3", "-r", "360",
                    "-singlefile", "-png", str(SOURCE), str(prefix)], check=True)
    with Image.open(str(prefix) + ".png") as page:
        assert page.size == (2977, 4210)
        page.crop(box).save(TARGET)
with Image.open(TARGET) as image:
    image.load()
    print(json.dumps({"width": image.width, "height": image.height,
                      "sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest()}))
