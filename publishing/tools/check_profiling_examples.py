import ast
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from fractions import Fraction
from hashlib import sha256

import torch

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '00 Учебник/10 ML Systems/07 Profiling ML-нагрузки.md'
text = chapter.read_text()
addition_match = re.search(r'<!-- BEGIN REVIEW: training step 2026-10-27 -->\n(.*?)<!-- END REVIEW: training step 2026-10-27 -->\n\n', text, re.S)
assert addition_match is not None
baseline = text[:addition_match.start()] + text[addition_match.end():]
baseline = baseline.replace('last_updated: 2026-10-27', 'last_updated: 2026-10-23')
baseline = baseline.replace('last_verified: 2026-10-27', 'last_verified: 2026-10-23')
assert sha256(baseline.encode()).hexdigest() == 'e930009037625adff0485b39cac3d91b30340abfe665a16049f439df785deac8'
blocks = re.findall(r'```python\n(.*?)\n```', text, re.S)
for i, block in enumerate(blocks):
    ast.parse(block, filename=f'S7-python-block-{i+1}')
print(f'AST: {len(blocks)} Python blocks passed')

broken = []
for match in re.finditer(r'\[\[([^\]|]+)(?:\|[^\]]*)?\]\]', text):
    target = match.group(1).split('#')[0].removeprefix('02 Areas/ML & DL/')
    path = ROOT / target
    if not path.exists() and not Path(str(path) + '.md').exists():
        broken.append(target)
assert not broken, broken
print('Wiki links and image paths: all resolved')

previous_dir = Path.cwd()
with tempfile.TemporaryDirectory(prefix='s7-profiling-cpu-') as scratch:
    workdir = Path(scratch)
    try:
        os.chdir(workdir)
        namespace = {}
        exec(compile(blocks[0], 'S7-profiler-example', 'exec'), namespace)
        trace_paths = list((workdir / 'artifacts/profiler').glob('*.json'))
        assert len(trace_paths) == 1 and trace_paths[0].name == 'step-5.json', trace_paths
        trace = json.loads(trace_paths[0].read_text())
        events = trace['traceEvents']
        counts = {name: sum(e.get('name') == name for e in events)
                  for name in ['forward_and_loss', 'backward', 'optimizer_step']}
        assert set(counts.values()) == {3}, counts
        assert sorted(e['name'] for e in events if e.get('name', '').startswith('ProfilerStep#')) == [
            'ProfilerStep#2', 'ProfilerStep#3', 'ProfilerStep#4']
        assert all(torch.isfinite(p).all() for p in namespace['model'].parameters())
        print('Profiler CPU export:', torch.__version__, 'CUDA:', torch.cuda.is_available())
        print('Trace events:', len(events), 'stage counts:', counts, '; temporary JSON removed after inspection')
    finally:
        os.chdir(previous_dir)

gelu_namespace = {'torch': torch}
gelu_block = next(b for b in blocks if 'def naive_gelu' in b)
exec(gelu_block, gelu_namespace)
for shape in [(17,), (7, 13), (0,)]:
    for dtype in [torch.float32, torch.float64]:
        base = torch.randn(*shape, dtype=dtype)
        x1 = base.clone().requires_grad_()
        x2 = base.clone().requires_grad_()
        y1 = gelu_namespace['naive_gelu'](x1)
        y2 = gelu_namespace['builtin_gelu'](x2)
        torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)
        y1.sum().backward()
        y2.sum().backward()
        torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-5, atol=1e-6)
print('GeLU CPU outputs/gradients: 3 shapes x 2 dtypes passed; torch.compile wrapper created, optimized backend not run')

history_block = next(b for b in blocks if '_record_memory_history' in b)
for fail in [False, True]:
    calls = []
    def run_steps():
        calls.append('run_steps')
        if fail:
            raise ValueError('injected failure')
    memory = SimpleNamespace(
        _record_memory_history=lambda **kw: calls.append(('record', kw)),
        _dump_snapshot=lambda filename: calls.append(('dump', filename)),
    )
    fake = SimpleNamespace(cuda=SimpleNamespace(memory=memory, synchronize=lambda: calls.append('sync')))
    try:
        exec(history_block, {'torch': fake, 'run_steps': run_steps})
    except ValueError:
        assert fail
    assert calls[-2:] == [('dump', 'snapshot.pickle'), ('record', {'enabled': None})], calls
print('Memory history cleanup control flow: success and injected failure passed (mock, not CUDA)')

assert 110 + 18 + 275 + 17 == 420
assert max(110, 18 + 275 + 17) == 310
assert 310 - 17 == 293

# Parse the chapter's illustrative tables, not a separate copy of the numbers.
def markdown_rows(header):
    body = text.split(header + '\n', 1)[1].split('\n\n', 1)[0]
    return [[cell.strip() for cell in row.strip('|').split('|')]
            for row in body.splitlines()[1:]]

host = markdown_rows('| Событие | Self CPU, мс | CPU total, мс |')
assert [(int(r[1]), int(r[2])) for r in host] == [(2, 12), (8, 8), (2, 2)]
assert sum(int(r[1]) for r in host) == int(host[0][2]) == 12
assert sum(int(r[2]) for r in host) == 22
first, second = (3, 8), (6, 10)
overlap = min(first[1], second[1])-max(first[0], second[0])
total = sum(b-a for a,b in (first, second))
assert (total, overlap, total-overlap, max(first[1], second[1])) == (9,2,7,10)

cost, eager, compiled = Fraction(8), Fraction(12,1000), Fraction(10,1000)
assert cost/(eager-compiled) == 4000
assert (1000*eager, cost+1000*compiled) == (12,18)
assert 4000*eager == cost+4000*compiled == 48
assert 4001*eager > cost+4001*compiled
memory = markdown_rows('| Состояние | Allocated | Reserved |')
states = [(int(r[1]),int(r[2])) for r in memory]
assert states == [(100,128),(160,192),(200,256),(160,256),(100,256)]
assert max(a for a,_ in states) == 200 and states[-1][1]-states[-1][0] == 156

archive = ROOT / '05 Источники/Courses/Stanford CS336 Spring 2026/Lectures/repository/var/traces/lecture_06.json'
source_steps = json.loads(archive.read_text())['steps']
for index, expected in ((245,9),(248,1),(251,1)):
    rows = [r['data'] for r in source_steps[index]['renderings'] if isinstance(r.get('data'),str)]
    launch = next(row for row in rows if re.search(r'\b(?:cudaLaunchKernel|cuLaunchKernel)\b',row))
    assert int(launch.split()[-1]) == expected
print('Worked examples: nesting/overlap, compile break-even, allocator lifetimes, 420/310/293 passed')
print('Pinned CS336 source outputs: 9/1/1 launches confirmed; not new GPU measurements')

# Preserve the source fragment, then run the separate complete editorial program.
addition = addition_match.group(1)
new_blocks = re.findall(r'```python\n(.*?)\n```', addition, re.S)
assert len(new_blocks) == 6
harvard = (ROOT / '05 Источники/Originals/frameworks-2026-10-27/chapter-2bd97c5.qmd').read_text()
source_fragment = harvard.split('{#lst-training-step-anatomy', 1)[1].split('```{.python}\n', 1)[1].split('```', 1)[0].strip()
assert new_blocks[0].strip() == source_fragment
try:
    torch.randint(0, 10, (32), device='cpu')
except TypeError:
    pass
else:
    raise AssertionError('Expected the source integer-size call to be rejected')
assert torch.randint(0, 10, (32,), device='cpu').shape == (32,)
step_namespace = {'__name__': '__training_step_example__'}
for i, block in enumerate(new_blocks[1:], 1):
    exec(compile(block, f'S7-actual-training-step-{i}', 'exec'), step_namespace)
    print(f'Actual training-step block {i}: PASS')

from collections import Counter
expected_products = Counter([
    ('mlp.forward', ((32, 784), (784, 256))),
    ('mlp.forward', ((32, 256), (256, 10))),
    ('mlp.backward', ((256, 32), (32, 10))),
    ('mlp.backward', ((32, 10), (10, 256))),
    ('mlp.backward', ((784, 32), (32, 256))),
])
assert Counter(step_namespace['fixed_input_products']) == expected_products
assert Counter(step_namespace['trainable_input_products']) == expected_products + Counter([
    ('mlp.backward', ((32, 256), (256, 784))),
])
assert step_namespace['loss_after'] < step_namespace['loss_before']

# Independently vary the dimensions and whether an input derivative is requested.
for batch, dim, hidden, classes in [(1, 3, 2, 2), (2, 4, 3, 2),
                                   (7, 3, 5, 4), (3, 1, 2, 3)]:
    for input_grad in (False, True):
        torch.manual_seed(batch + hidden)
        xx = torch.randn(batch, dim, dtype=torch.float64, requires_grad=input_grad)
        w1 = torch.randn(dim, hidden, dtype=xx.dtype, requires_grad=True)
        bb1 = torch.randn(hidden, dtype=xx.dtype, requires_grad=True)
        w2 = torch.randn(hidden, classes, dtype=xx.dtype, requires_grad=True)
        bb2 = torch.randn(classes, dtype=xx.dtype, requires_grad=True)
        target = torch.randint(classes, (batch,))
        aa = xx @ w1 + bb1
        hh = aa.relu()
        scores = hh @ w2 + bb2
        torch.nn.functional.cross_entropy(scores, target).backward()
        with torch.no_grad():
            gg = scores.softmax(-1)
            gg[torch.arange(batch), target] -= 1
            gg /= batch
            grad_a = (gg @ w2.T) * (aa > 0)
            manual = (xx.T @ grad_a, grad_a.sum(0), hh.T @ gg, gg.sum(0))
            for tensor, expected in zip((w1, bb1, w2, bb2), manual):
                torch.testing.assert_close(tensor.grad, expected)
            if input_grad:
                torch.testing.assert_close(xx.grad, grad_a @ w1.T)
            else:
                assert xx.grad is None
print('PASS: exact Harvard fragment, rejected integer size, five actual runnable blocks; '
      'all parameter gradients and SGD updates; exact 5/6 CPU mm events and shapes; '
      'matrix-work totals; 8 independent gradient cases. No CUDA timing or kernel-count claim.')
print('chapter_sha256='+sha256(chapter.read_bytes()).hexdigest())
