"""Render the author/course figures without modifying their scientific content."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import shutil
import subprocess
import sys
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    ("05 Источники/Originals/activation-functions-2026-10-21/gelus-author.pdf",
     "cebe02ef59174a386761849c09d66814332b9befb7dde28fdbdfdb72942427ec",
     2, "00 Учебник/Assets/Figures/curated/source-audit-1-18/gelu-figure-1.png",
     (912, 1040, 1696, 1632)),  # Figure 1 and complete original caption.
    ("05 Источники/Courses/Stanford CS336 Spring 2026/Assignments/assignment1-basics/cs336_assignment1_basics.pdf",
     "cf402d1c6d3c66da035bdf33b3e2c1bbc4bb92a78b006ee36577e925a265e328",
     21, "00 Учебник/Assets/Figures/curated/source-first-32-34/cs336-silu-vs-relu.png",
     (633, 293, 1420, 1084)),  # Complete plot, axes and legend, no course heading.
]
renderer = shutil.which("pdftoppm") or "/opt/homebrew/bin/pdftoppm"
for source_path, digest, page, target_path, box in CASES:
    source = ROOT / source_path
    target = ROOT / target_path
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with TemporaryDirectory(prefix="bookvar-activation-panel-") as folder:
        prefix = Path(folder) / "page"
        subprocess.run([renderer, "-f", str(page), "-l", str(page), "-r", "240",
                        "-singlefile", "-png", str(source), str(prefix)], check=True)
        original = Image.open(prefix.with_suffix(".png")).convert("RGBA")
        assert original.size == (2040, 2640)
        expected = original.crop(box)
        if "--check" not in sys.argv:
            expected.save(target)
        actual = Image.open(target).convert("RGBA")
        assert actual.size == (box[2]-box[0], box[3]-box[1])
        assert ImageChops.difference(expected, actual).getbbox(alpha_only=False) is None
        print("PASS: exact PDF pixel crop", target.name, actual.size,
              hashlib.sha256(target.read_bytes()).hexdigest())

