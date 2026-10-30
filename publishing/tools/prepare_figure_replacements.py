"""Export an unchanged, full composited source-GIF frame for web reading.

The 26 MB original remains in Originals; the published PNG preserves every
pixel of frame 171. No crop, relabeling, redraw or resampling.
"""
from pathlib import Path
import hashlib
import json
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/figure-replacements-2026-10-30/09-gpt3-generating-react-code-example.gif"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/figure-replacements-2026-10-30/gpt3-demonstrations-frame-171.png"
EXPECTED_SOURCE_SHA256 = "82263d0d9a499c1b423590bd033ad7f7422298f441f229e01a5705d2cd15385d"


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SOURCE_SHA256
    with Image.open(SOURCE) as source:
        assert source.size == (1438, 813) and source.n_frames == 172
        source.seek(171)
        original = source.convert("RGB")
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        original.save(TARGET, format="PNG", optimize=True)
    print(json.dumps({"asset": str(TARGET.relative_to(ROOT)), "width": 1438,
        "height": 813, "source_frame": 171, "source_frames": 172,
        "sha256": hashlib.sha256(TARGET.read_bytes()).hexdigest(),
        "bytes": TARGET.stat().st_size}))


if __name__ == "__main__":
    main()
