#!/usr/bin/env python3
"""Validate pinned V4 config arithmetic and unchanged original CS336 figures."""
import hashlib
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / '00 Учебник/Assets/Figures/curated/cs336-residuals-2026-10-29'
SRC = ROOT / '05 Источники/Architecture/DeepSeek V4 2026-10-29'
COURSE = ROOT / '05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Save the same check receipt printed to stdout')
    args = parser.parse_args()
    config = json.loads((SRC / 'config.json').read_text())
    schedule = config['compress_ratios']
    assert config['num_hidden_layers'] == 61 and config['num_nextn_predict_layers'] == 1
    assert len(schedule) == config['num_hidden_layers'] + config['num_nextn_predict_layers'] == 62
    decoder = schedule[:config['num_hidden_layers']]
    counts = {str(rate): decoder.count(rate) for rate in (0, 4, 128)}
    assert counts == {'0': 0, '4': 30, '128': 31}
    assert schedule[config['num_hidden_layers']:] == [0]
    assert config['max_position_embeddings'] == 1048576
    assert config['index_topk'] == 1024 and config['sliding_window'] == 128
    assert config['head_dim'] == 512 and config['index_head_dim'] == 128
    assert config['num_key_value_heads'] == 1 and config['qk_rope_head_dim'] == 64
    n = config['max_position_embeddings']
    pools = {'CSA': n // 4, 'HCA': n // 128}
    assert pools == {'CSA': 262144, 'HCA': 8192}
    assert sha(SRC / 'LICENSE') == sha(ROOT / 'site/public/licenses/deepseek-v4-89d501ae-MIT.txt')
    pairs = {
        'deepseek-v4-csa.png': 'images/deepseek-v4-attention.png',
        'arc-agi-3.png': 'images/arc-agi-3.png',
        'gcg-historical-examples.png': 'images/gcg-examples.png',
        'arc-task-grids.jpg': 'var/files/image-d1a33e9159cfdb77197551bbbecc6a76-https_arcprize_org_media_images_arc-task-grids_jpg',
        'arc-agi-2-example.png': 'var/files/image-a0338c9fb72d1163cfc8ac66fea4e4ed-https_arcprize_org_media_images_blog_arc-agi-2-unsolved-1_png',
    }
    for target, original in pairs.items():
        assert (FIG / target).read_bytes() == (COURSE / original).read_bytes(), target
    with tempfile.TemporaryDirectory(prefix='bookvar-v4-figure-') as temp:
        output = Path(temp) / 'hca'
        subprocess.run(['pdftoppm', '-f', '11', '-l', '11', '-r', '300',
                        '-x', '650', '-y', '330', '-W', '1200', '-H', '730',
                        '-png', '-singlefile', str(SRC / 'DeepSeek_V4.pdf'), str(output)], check=True)
        assert output.with_suffix('.png').read_bytes() == (FIG / 'deepseek-v4-hca.png').read_bytes()
    trace = json.loads((COURSE / 'var/traces/lecture_02.json').read_text())
    data = [trace['steps'][step - 1]['renderings'][0]['data'] for step in (125, 369, 427, 926)]
    assert data == ['images/cpu-gpu.png'] + ['images/deep-network.png'] * 3
    chapter = (ROOT / '02 Атлас моделей/Семейства/DeepSeek.md').read_text()
    assert 'source-excerpt: deepseek-v4-local-window' in chapter
    assert len(re.findall(r'!\[\[.*cs336-residuals.*\]\]', chapter)) == 2
    result = {
        'scope': 'config counts, source bytes, reproducible PDF extraction; no V4 model inference',
        'config_sha256': sha(SRC / 'config.json'), 'decoder_layer_counts': counts,
        'mtp_compression_ratios': schedule[config['num_hidden_layers']:],
        'logical_pool_counts_at_1048576': pools, 'original_byte_copies': len(pairs),
        'hca_pdf_reproduction': 'passed', 'lecture02_renderings': data,
        'assets': {p.name: sha(p) for p in sorted(FIG.iterdir()) if p.suffix in {'.png', '.jpg'}},
        'chapters': {name: sha(ROOT / name) for name in [
            '02 Атлас моделей/Семейства/DeepSeek.md',
            '00 Учебник/18 Evaluation и методология/59 Оценивание моделей и контаминация.md',
            '00 Учебник/18 Evaluation и методология/59a Responsible systems.md']},
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
