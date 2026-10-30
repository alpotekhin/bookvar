#!/usr/bin/env python3
"""Render Harvard's unchanged TCO plotting cell, not a Bookvar replacement.

uv run --no-project --python <bundled-python> --no-build \
  --with matplotlib==3.9.4 --with numpy==2.1.3 \
  python publishing/tools/render_finops_source_figure.py
"""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / '05 Источники/Originals/mlops-2026-10-20/harvard-ops-scale.qmd'
OUTPUT = ROOT / '00 Учебник/Assets/Figures/curated/finops-2026-10-26/harvard-tco-iceberg.png'


def source_cell():
    data = SOURCE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == '21fd839e4d7825e2a61f379447a94db10162f2dff954005a1b413c1da573f5cd'
    section = data.decode().split('::: {#fig-tco-iceberg ', 1)[1].split('\n:::', 1)[0]
    return re.search(r'^```\{python\}\n(.*?)^```', section, re.M | re.S)[1]


def main():
    cell = source_cell()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Keep every original plotting instruction. Intercept show only to export
    # the entire original canvas AND its out-of-canvas labels at 300 dpi.
    # The source does not call tight_layout, so a bare canvas export clips
    # the long y-axis labels. Expand bounds without moving any artists.
    def save_show(*args, **kwargs):
        fig = plt.gcf()
        fig.canvas.draw()
        bounds = Bbox.union([Bbox.from_bounds(0, 0, *fig.get_size_inches()),
                             fig.get_tightbbox(fig.canvas.get_renderer())]).padded(.05)
        fig.savefig(OUTPUT, dpi=300, facecolor='white', bbox_inches=bounds,
                    metadata={'Software': 'Harvard source cell; Matplotlib 3.9.4'})
    plt.show = save_show
    namespace = {}
    exec(compile(cell, str(SOURCE) + ':fig-tco-iceberg', 'exec'), namespace)
    assert namespace['values'] == [40, 10, 20, 10, 5, 5, 5, 5]
    assert sum(namespace['values']) == 100
    print(json.dumps({'asset': str(OUTPUT.relative_to(ROOT)),
                      'sha256': hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      'source_cell_sha256': hashlib.sha256(cell.encode()).hexdigest(),
                      'dimensions_px': list(Image.open(OUTPUT).size),
                      'matplotlib': matplotlib.__version__}, indent=2))


if __name__ == '__main__':
    main()
