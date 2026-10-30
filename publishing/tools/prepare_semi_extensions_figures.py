"""Render complete licensed MixText figures and their original captions."""
from pathlib import Path
import hashlib
import json
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/semi-supervised-extensions-2026-10-29/mixtext-acl-2020.pdf"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/semi-supervised-extensions-2026-10-29"
TARGET.mkdir(parents=True, exist_ok=True)
FIGURES = [
    (0, (303, 220, 532, 475), "tmix-original.png"),
    (4, (68, 68, 532, 278), "mixtext-original.png"),
]

with pymupdf.open(SOURCE) as doc:
    assert len(doc) == 11
    for page_index, bounds, name in FIGURES:
        page = doc[page_index]
        rect = pymupdf.Rect(bounds)
        assert page.rect.contains(rect)
        # Native diagrams are 954×880 and 1704×628 raster objects in the PDF.
        # 240 dpi preserves readable vector captions without enlarging those bitmaps.
        pix = page.get_pixmap(matrix=pymupdf.Matrix(240/72, 240/72), clip=rect, alpha=False)
        output = TARGET / name
        pix.save(output)
        print(json.dumps({"file": name, "dimensions_px": [pix.width, pix.height],
                          "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                          "bytes": output.stat().st_size,
                          "source_images": page.get_images(full=True)}, default=str))
