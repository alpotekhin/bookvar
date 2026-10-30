"""Render the complete licensed DropConnect figure without redrawing it."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/structured-dropout-2026-10-29"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/structured-dropout-2026-10-29"
CHAPTER = "00 Учебник/01 Основы нейронных сетей/06 Dropout и регуляризация.md"
CROP = (340, 890, 2110, 1685)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def relative(p):
    return p.relative_to(ROOT).as_posix()

def main():
    pdf = SOURCE / "senen-cerda-sanders-2025.pdf"
    assert digest(pdf) == "1296d2ee001235816ebe59c3f17420c39e65538fea0af69ddc2c3f400705f9af"
    TARGET.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bookvar-structured-dropout-") as directory:
        prefix = str(Path(directory) / "page2")
        subprocess.run(["pdftoppm", "-f", "2", "-l", "2", "-r", "288",
                        "-singlefile", "-png", str(pdf), prefix], check=True)
        with Image.open(prefix + ".png") as image:
            assert image.size == (2448, 3168)
            image.crop(CROP).save(TARGET / "dropconnect-original.png")
    asset = TARGET / "dropconnect-original.png"
    row = {
        "id": "curated-structured-dropout-" + digest(asset)[:12],
        "asset": relative(asset),
        "author": "Albert Senen-Cerda and Jaron Sanders",
        "title": "Original Figure 1: full network, DropConnect sample and arborescence",
        "source_url": "https://jmlr.org/papers/v26/20-1396.html",
        "source_image_url": "https://jmlr.org/papers/volume26/20-1396/20-1396.pdf#page=2",
        "source_asset": relative(pdf), "source_asset_sha256": digest(pdf),
        "source_page": 2, "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_evidence": relative(pdf), "license_evidence_sha256": digest(pdf),
        "license_evidence_location": "PDF page 1, copyright and license footer",
        "copyright": "Copyright 2025 Albert Senen-Cerda and Jaron Sanders",
        "derivation": "pdf-page-render-crop", "render_dpi": 288,
        "render_page_size_px": [2448, 3168], "crop_box_px": list(CROP),
        "modifications": "All three panels and complete original caption cropped from PDF page 2. No redraw, relabeling or translation. The caption's remove/keep probability inconsistency is explained adjacent to the figure.",
        "last_verified": "2026-10-29", "metadata_status": "verified",
        "provenance_confirmation": "source-license-verified", "used_in": [CHAPTER],
        "dimensions_px": [1770, 795], "width": 1770, "height": 795,
        "sha256": digest(asset), "bytes": asset.stat().st_size,
    }
    (TARGET / "assets.json").write_text(json.dumps([row], ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(SOURCE / "timm-LICENSE",
                    ROOT / "site/public/licenses/timm-beabdc3a-Apache-2.0.txt")
    print(json.dumps(row, ensure_ascii=False))

if __name__ == "__main__":
    main()
