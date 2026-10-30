"""Check the untouched D2L sources and reproduce the complete existing PNGs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import cairosvg
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/neural-language-model-2026-10-21"
CURATED = ROOT / "00 Учебник/Assets/Figures/curated"
expected = {
    "d2l-language-model-data": "b22c0c38c2d987ed2b4f7c8b394f8d8442e592e42c321f63775fa1d388840373",
    "d2l-gpt-decoder": "c586784e0c1404f7fe0ffcda1354f02cfe64f708f80eb9c6e5fc1d32248ea964",
    "d2l-softmaxreg": "f426511da025523e7eac27ce91ccb4ef3b402fe0aad75bf0ce9207340bbfe547",
}
for stem, digest in expected.items():
    source = SOURCE / f"{stem}.svg"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    if stem == "d2l-softmaxreg":
        target = CURATED / "neural-language-model-2026-10-21" / f"{stem}.svg"
        assert source.read_bytes() == target.read_bytes()
        print("PASS byte-identical source SVG", stem)
        continue
    original = CURATED / "ntp-review-2026-10" / f"{stem}-original.svg"
    assert original.read_bytes() == source.read_bytes()
    png = CURATED / "ntp-review-2026-10" / f"{stem}.png"
    with TemporaryDirectory(prefix="bookvar-neural-lm-figure-") as folder:
        render = Path(folder) / "source.png"
        cairosvg.svg2png(url=str(source), write_to=str(render),
                        output_width=2000, background_color="white")
        expected_image = Image.open(render).convert("RGBA")
        actual_image = Image.open(png).convert("RGBA")
        assert expected_image.size == actual_image.size
        assert ImageChops.difference(expected_image, actual_image).getbbox(alpha_only=False) is None
        print("PASS complete source rasterization", stem, actual_image.size)
