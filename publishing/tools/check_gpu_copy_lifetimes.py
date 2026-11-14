"""Execute the actual GPU chapter programs and check a finite dependency model.
CPU checks are not CUDA execution, profiler reproduction or performance evidence.
"""
import ast
import hashlib
import itertools
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/10 ML Systems/02 GPU, CUDA и иерархия памяти.md'
body = CHAPTER.read_text()
blocks = re.findall(r'^(~~~|```)python\n([\s\S]*?)^\1\s*$', body, re.M)
assert len(blocks) == 8
# Retain the independent tile/sector/bank/occupancy and double-buffer checks.
runpy.run_path(str(ROOT / 'publishing/tools/check_gpu_foundations_examples.py'))
namespace = {}
for number, (_, block) in enumerate(blocks, 1):
    exec(compile(block, f'{CHAPTER}:program-{number}', 'exec'), namespace)
print('PASS all 8 actual chapter programs; CUDA stream executed=' +
      str(namespace['cuda_stream_example_executed']))

# Verify every source statement in the official corrected example, not just
# selected API names. Source body is pinned by its complete-file SHA below.
source = ROOT / '05 Источники/Originals/harvard-performance-2026-10-26/pytorch-v2.14.1-cuda.md'
if source.exists():
    raw = source.read_bytes()
else:
    from urllib.request import urlopen
    url = ('https://raw.githubusercontent.com/pytorch/pytorch/'
           '5c4886908584029761b579af026dcfb627c84070/docs/source/notes/cuda.md')
    with urlopen(url, timeout=30) as response:
        raw = response.read()
assert hashlib.sha256(raw).hexdigest() == '133ae6298b2696c121a866eeb1e0433c2344fb2d938188b43ebdddea0b09bf8a'
section = raw.decode().split('The fixed version of this example is:')[1]
original = re.search(r'```python\n(.*?)\n```', section, re.S).group(1)
guarded = next(block for _, block in blocks if 'cuda_stream_example_executed' in block)
guard = next(node for node in ast.parse(guarded).body if isinstance(node, ast.If))
original_ast = [ast.dump(node) for node in ast.parse(original).body]
actual_ast = [ast.dump(node) for node in guard.body[:len(original_ast)]]
assert actual_ast == original_ast
assert len(guard.body) == len(original_ast) + 2
print('PASS original corrected PyTorch example statements, pinned source bytes')

model = namespace['read_interleaving']
events = ['copy_left', 'copy_right', 'read_left', 'read_right',
          'reuse_host', 'reuse_device']
# Copy/read halves retain program order. The next device write belongs
# to the producer stream, after its copy. Three extra edges ensure safety.
base = [('copy_left', 'copy_right'), ('read_left', 'read_right'),
        ('copy_right', 'reuse_device')]
safety = [('copy_right', 'read_left'), ('copy_right', 'reuse_host'),
          ('read_right', 'reuse_device')]
orders = list(itertools.permutations(events))
assert len(orders) == 720
def follows(order, edges):
    index = {event: i for i, event in enumerate(order)}
    return all(index[a] < index[b] for a, b in edges)

safe_orders = [o for o in orders if follows(o, base + safety)]
assert safe_orders and all(model(o) == [1, 2, 3, 4] for o in safe_orders)
for edge in safety:
    weakened = base + [other for other in safety if other != edge]
    failures = [o for o in orders if follows(o, weakened) and model(o) != [1, 2, 3, 4]]
    assert failures, edge
    print('counterexample without', edge, failures[0], '->', model(failures[0]))
try:
    model(['unknown'])
except ValueError:
    pass
else:
    raise AssertionError('unknown event accepted')
assert 3 < 10  # Host source last-read and device input last-read differ.
print(f'PASS 720 finite event orders; {len(safe_orders)} satisfy all safety edges; '
      'each removed dependency admits a counterexample. Logical CPU model only.')
print('chapter_sha256=' + hashlib.sha256(CHAPTER.read_bytes()).hexdigest())

