"""Render the complete native Figure 1 from Dolz et al., no redraw."""
from pathlib import Path
import hashlib
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "05 Источники/Originals/winograd-2026-10-27/dolz-2023.pdf"
assert hashlib.sha256(PDF.read_bytes()).hexdigest() == \
    "c71022d1c7bdfef9e423974f82c6ea407e45a304c83e5d45f6b7992a68841536"
OUT = ROOT / "00 Учебник/Assets/Figures/curated/winograd-2026-10-27/dolz-workflow"
OUT.parent.mkdir(parents=True, exist_ok=True)
# PDF physical page 4. Crop contains the whole native figure, not page body.
# Original caption is reproduced beside the image in the chapter.
subprocess.run(["pdftoppm", "-f", "4", "-l", "4", "-r", "400",
                "-x", "572", "-y", "305", "-W", "1295", "-H", "1278",
                "-png", "-singlefile", str(PDF), str(OUT)], check=True)
print(OUT.with_suffix(".png"))
print("SHA256", hashlib.sha256(OUT.with_suffix(".png").read_bytes()).hexdigest())
