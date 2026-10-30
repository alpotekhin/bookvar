"""Export the three original inline CS231n SVGs; do not redraw their contents."""
from pathlib import Path
import argparse
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '05 Источники/Originals/backpropagation-2026-10-21/cs231n-optimization-2.md'
TARGET = ROOT / '00 Учебник/Assets/Figures/curated/backpropagation-2026-10-21'
NAMES = ('cs231n-chain.svg', 'cs231n-sigmoid.svg', 'cs231n-gates.svg')

def figures():
    items = re.findall(r'<svg\b.*?</svg>', SOURCE.read_text(), re.S)
    assert len(items) == 3
    for name, svg in zip(NAMES, items):
        # Inline HTML normalizes viewbox; standalone XML requires viewBox and xmlns.
        # Preserve every original drawing element, coordinate, label and color.
        svg = svg.replace('viewbox=', 'viewBox=', 1)
        svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        yield name, svg + '\n'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, svg in figures():
        if args.check:
            assert (TARGET / name).read_text() == svg, name
        else:
            (TARGET / name).write_text(svg)
        print(name, 'matches original inline SVG' if args.check else 'exported')
