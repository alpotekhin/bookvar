"""Keep course/paper pixels; separate original subplot panels for narrow pages."""
from pathlib import Path
import hashlib
import sys
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "05 Источники/Originals/evaluation-2026-10-21"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/classical-ml-evaluation"
CASES = [
    (SOURCES / "pr-roc.png", "pr-curve.png", (0, 68, 643, 694)),
    (SOURCES / "pr-roc.png", "roc-curve.png", (645, 68, 1280, 694)),
    (SOURCES / "calibration.png", "calibration-curve.png", (0, 0, 1000, 500)),
    (SOURCES / "calibration.png", "calibration-legend.png", (785, 325, 985, 447)),
    (SOURCES / "regression-residuals.png", "regression-actual.png", (0, 100, 540, 585)),
    (SOURCES / "regression-residuals.png", "regression-residual.png", (550, 100, 1069, 585)),
    (ROOT / "Assets/Sources/Scikit-learn MOOC/figures/nested_cross_validation_diagram.png", "nested-first-outer.png", (0, 95, 915, 370)),
    (ROOT / "Assets/Sources/Scikit-learn MOOC/figures/nested_cross_validation_diagram.png", "nested-legend.png", (960, 95, 1310, 355)),
]

if __name__ == "__main__":
    for source, name, box in CASES:
        original = Image.open(source).convert("RGBA")
        assert 0 <= box[0] < box[2] <= original.width
        assert 0 <= box[1] < box[3] <= original.height
        expected = original.crop(box)
        if "--check" not in sys.argv:
            expected.save(TARGET / name)
        actual = Image.open(TARGET / name).convert("RGBA")
        assert ImageChops.difference(expected, actual).getbbox(alpha_only=False) is None
        print(name, actual.size, hashlib.sha256((TARGET / name).read_bytes()).hexdigest())
