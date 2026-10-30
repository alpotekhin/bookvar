"""Extract the original plot and legend as separate panels; never redraw them."""
from pathlib import Path
import hashlib
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "05 Источники/Originals/ensembles-2026-10-21"
TARGET = ROOT / "00 Учебник/Assets/Figures/curated/classical-ml-ensembles"
CASES = [
    ("gradient-initial-residuals.png", "gradient-initial-plot.png", (0, 0, 610, 520)),
    ("gradient-initial-residuals.png", "gradient-initial-legend.png", (615, 145, 934, 240)),
    ("gradient-fit-residuals.png", "gradient-residual-plot.png", (0, 0, 635, 487)),
    ("gradient-fit-residuals.png", "gradient-residual-legend.png", (640, 112, 957, 240)),
]


def assert_same_pixels(expected, actual):
    assert actual.size == expected.size, "Panel dimensions differ from the source crop"
    assert actual.mode == expected.mode == "RGBA", "Compare all four RGBA channels"
    assert actual.tobytes() == expected.tobytes(), "Panel pixels differ from the source crop"


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        reference = Image.new("RGBA", (1, 1), (0, 0, 0, 255))
        assert_same_pixels(reference, reference.copy())
        # An RGBA difference with an unchanged alpha channel can have an empty
        # getbbox(); byte comparison must reject changes to each color channel.
        mutations = [
            Image.new("RGBA", (1, 1), color)
            for color in [(1, 0, 0, 255), (0, 1, 0, 255), (0, 0, 1, 255), (0, 0, 0, 254)]
        ] + [Image.new("RGBA", (2, 1), (0, 0, 0, 255))]
        for mutation in mutations:
            try:
                assert_same_pixels(reference, mutation)
            except AssertionError:
                continue
            raise AssertionError("Pixel comparison accepted a changed panel")
        print("PASS: unchanged pixels accepted; R/G/B/alpha and size mutations rejected")
        raise SystemExit(0)
    for source, target, box in CASES:
        im = Image.open(SOURCE / source).convert("RGBA")
        assert 0 <= box[0] < box[2] <= im.width and 0 <= box[1] < box[3] <= im.height
        expected = im.crop(box)
        if "--check" not in sys.argv:
            expected.save(TARGET / target)
        actual = Image.open(TARGET / target).convert("RGBA")
        assert_same_pixels(expected, actual)
        print(target, actual.size, hashlib.sha256((TARGET / target).read_bytes()).hexdigest())
