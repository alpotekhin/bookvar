#!/usr/bin/env python3
"""Export one complete original Alammar animation frame, without a redraw/crop."""
from pathlib import Path
import argparse
import hashlib
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '05 Источники/Originals/gpt3-2026-10-22/08-gpt3-tokens-transformer-blocks.gif'
TARGET = ROOT / '00 Учебник/Assets/Figures/curated/gpt3-2026-10-22/tokens-through-stack-frame-385.png'
SOURCE_SHA = '650dbd12fa7e3636d24919bd7814a4711bfac6173a459213b8a0164cc849a06d'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    with Image.open(SOURCE) as original:
        assert original.size == (1438, 813) and original.n_frames == 386
        original.seek(385)
        frame = original.convert('RGB')
        if args.check:
            with Image.open(TARGET) as output:
                assert output.size == frame.size
                assert ImageChops.difference(frame, output.convert('RGB')).getbbox() is None
        else:
            TARGET.parent.mkdir(parents=True, exist_ok=True)
            frame.save(TARGET)
    print('PASS complete composited source frame 385, no crop or changed labels')
    print(hashlib.sha256(TARGET.read_bytes()).hexdigest(), TARGET.relative_to(ROOT))


if __name__ == '__main__':
    main()
