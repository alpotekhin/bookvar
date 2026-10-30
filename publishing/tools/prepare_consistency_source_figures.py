"""Render complete existing ACL figures; copy the author's original VAT GIF.

No redrawing, label replacement or bitmap upscaling. PDF coordinates are
points from the official complete PDFs, not from a screenshot preview.
"""
from pathlib import Path
import hashlib
import json
import shutil
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/consistency-training-2026-10-29"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/consistency-training-2026-10-29"
TARGET.mkdir(parents=True, exist_ok=True)

figures = [
    ("huang-2022-lightweight.pdf", 4, (68, 65, 529, 253), "uda-fixmatch.png"),
    ("smart-acl-2020.pdf", 3, (70, 64, 293, 260), "smart-boundaries.png"),
]
for name, page_index, bounds, output in figures:
    with pymupdf.open(SOURCE / name) as doc:
        page = doc[page_index]
        clip = pymupdf.Rect(bounds)
        assert page.rect.contains(clip)
        page.get_pixmap(matrix=pymupdf.Matrix(400 / 72, 400 / 72),
                        clip=clip, alpha=False).save(TARGET / output)

shutil.copyfile(SOURCE / "vat.gif", TARGET / "vat.gif")
shutil.copyfile(SOURCE / "vat_tf-LICENSE.md", ROOT / "site/public/licenses/vat-tf-c5125d2-MIT.txt")
shutil.copyfile(SOURCE / "uda-LICENSE", ROOT / "site/public/licenses/uda-960684e-Apache-2.0.txt")
rows = []
for p in TARGET.iterdir():
    if p.suffix not in (".png", ".gif"):
        continue
    with Image.open(p) as im:
        rows.append({"file": p.name, "dimensions_px": list(im.size),
                     "frames": getattr(im, "n_frames", 1),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                     "bytes": p.stat().st_size})
print(json.dumps(rows, indent=2))
