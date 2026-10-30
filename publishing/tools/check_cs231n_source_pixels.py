#!/usr/bin/env python3
"""Reproduce the two existing full-slide figures from the verified 2025 PDF."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / '05 Источники/Originals/cs231n-provenance-2026-10-29/lecture_2.pdf'
FIG = ROOT / '00 Учебник/Assets/Figures/curated/foundations-source-first'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    assert sha(PDF) == '191488d0a96f85574e8e2ebb8246865d3aac6d84da88fe0962aa1651f814565e'
    records = []
    with tempfile.TemporaryDirectory(prefix='bookvar-cs231n-pixels-') as temp:
        for name, physical, printed, asset_id in [
            ('cs231n-softmax-cross-entropy.png', 75, 77, 'curated-cs231n-softmax-ce-0000000000a6'),
            ('cs231n-linear-decision-boundaries.png', 61, 63, 'curated-cs231n-boundaries-0000000000a7'),
        ]:
            output = Path(temp) / name.removesuffix('.png')
            subprocess.run(['pdftoppm', '-f', str(physical), '-l', str(physical),
                            '-r', '160', '-png', '-singlefile', str(PDF), str(output)], check=True)
            original = Image.open(FIG / name).convert('RGB')
            fresh = Image.open(output.with_suffix('.png')).convert('RGB')
            assert original.size == fresh.size == (1600, 900)
            difference = ImageChops.difference(original, fresh)
            assert difference.getbbox() is None, name
            records.append({
                'id': asset_id, 'asset': str((FIG / name).relative_to(ROOT)),
                'sha256': sha(FIG / name), 'source_page': physical,
                'source_printed_page': printed, 'dimensions_px': list(original.size),
                'render_dpi': 160, 'identical_rgb_pixels': True,
                'mean_absolute_error_rgb': ImageStat.Stat(difference).mean,
                'modifications': 'Existing complete-slide PNG retained unchanged; exact RGB pixel identity to a fresh full-page pdftoppm render of the verified source PDF.',
            })
    result = {
        'reviewed_at': '2026-10-29',
        'source_url': 'https://cs231n.stanford.edu/slides/2025/lecture_2.pdf',
        'source_asset': str(PDF.relative_to(ROOT)),
        'source_asset_sha256': sha(PDF), 'source_bytes': PDF.stat().st_size,
        'scope': 'Source/physical-page/RGB-pixel verification only. Does not create a new image license; existing named-slide rights limitation remains.',
        'render_tool': subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True).stderr.splitlines()[0],
        'records': records,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
