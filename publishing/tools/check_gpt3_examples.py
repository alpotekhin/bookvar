#!/usr/bin/env python3
"""Execute both chapter examples and independently check the numerical claims."""
from pathlib import Path
import math
import re
import hashlib
import subprocess
import contextlib
import io
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
chapter = ROOT / '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/06 GPT-3 — in-context learning.md'
text = chapter.read_text()
blocks = re.findall(r'~~~python\n(.*?)\n~~~', text, re.S)
assert len(blocks) == 2
envs = []
for i, block in enumerate(blocks):
    env = {}
    exec(compile(block, str(chapter) + ':' + str(i + 1), 'exec'), env)
    envs.append(env)

a, b = envs
np.testing.assert_allclose(a['weights'].numpy(), [.75, .25], atol=1e-15)
assert math.isclose((a['weights'] @ a['values_a']).item(), .5, abs_tol=1e-15)
assert math.isclose((a['weights'] @ a['values_b']).item(), -.5, abs_tol=1e-15)

score = b['completion_logprob']
np.testing.assert_allclose(b['total'].exp().numpy(), [.18, .32], atol=1e-15)
np.testing.assert_allclose(b['total'].softmax(dim=0).numpy(), [.36, .64], atol=1e-15)
assert int(b['probs'][:, 0, :].argmax(-1)[0]) == 1
assert int(b['total'].argmax()) == 1

# Different prompt length and a manual scalar loop are independent of the gather.
torch.manual_seed(3)
logits = torch.randn(2, 6, 7, dtype=torch.float64)
ids = torch.tensor([[0, 3, 4, 2, 6, 1], [2, 1, 6, 0, 5, 3]])
for start in (1, 2, 4, 5):
    total, mean = score(logits, ids, start)
    for row in range(2):
        expected = 0.
        for pos in range(start, ids.shape[1]):
            values = logits[row, pos - 1].numpy()
            logz = np.logaddexp.reduce(values)
            expected += float(values[int(ids[row, pos])] - logz)
        assert math.isclose(float(total[row]), expected, abs_tol=1e-12)
        assert math.isclose(float(mean[row]), expected / (6 - start), abs_tol=1e-12)
    changed = logits.clone()
    if start > 1:
        changed[:, :start - 1] = torch.randn_like(changed[:, :start - 1]) * 50
        torch.testing.assert_close(score(changed, ids, start)[0], total)
print('PASS answer-only scoring, shifted positions and four prompt boundaries')

# End markers have conditional probabilities too; they can change a ranking.
assert .18 < .32 and .18 * .9 > .32 * .1
assert .4 > .3 and .4 < math.sqrt(.3)
assert .4 / .5 < .3 / .05
assert (2048 - 80 - 50 - 128) // 36 == 49
assert 80 + 49 * 36 + 50 + 128 == 2022
assert 80 + 50 * 36 + 50 + 128 == 2058
assert 96 * 128 == 12288 and 4 * 12288 == 49152
assert 96 * 2048**2 * 2 == 768 * 2**20
assert 2 * 96 * 2048 * 12288 * 2 == 9 * 2**30

full = torch.ones(8, 8, dtype=torch.bool).tril()
local = full & (torch.arange(8)[None, :] >= torch.arange(8)[:, None] - 2)
assert full[-1].nonzero().flatten().tolist() == list(range(8))
assert local[-1].nonzero().flatten().tolist() == [5, 6, 7]
assert int(full.sum()) == 36 and int(local.sum()) == 21
assert 2 * int(full.sum()) == 72
assert int(full.sum() + local.sum()) == 57
# Rows are queries, columns are previous-layer keys. Boolean composition
# describes possible paths, not guaranteed nonzero learned influence.
local_after_full = (local.to(torch.int64) @ full.to(torch.int64)) > 0
two_local = (local.to(torch.int64) @ local.to(torch.int64)) > 0
assert local_after_full[-1].nonzero().flatten().tolist() == list(range(8))
assert two_local[-1].nonzero().flatten().tolist() == list(range(3, 8))
# Ordinary causal packing has no automatic boundary at EOT (position 2).
assert full[3, :4].tolist() == [True] * 4

samples = np.random.default_rng(3).pareto(9, 500_000)
for s in (0., .9, 1.):
    theoretical = (2 - s)**-9
    observed = float((samples > 1 - s).mean())
    assert abs(theoretical - observed) < .003
assert math.isclose((2 - 0)**-9, 1 / 512)
assert math.isclose((2 - .9)**-9, .4240976183724848)
assert sum([60, 22, 8, 8, 3]) == 101
assert 2209 + 2944 == 5153
assert math.isclose(17 / 2000 * 100, .85)
assert 83 + 59 == 142 and 3 + 9 == 12 and 8 + 5 + 1 == 14
assert (8 + 5) * 10 + 2 == 132
print('PASS length/frequency/terminator scoring, context/memory/masks and Pareto filtering')

# Reuse the actual preceding chapter's small causal network. This is a
# mathematical cache example, not an unavailable GPT-3 implementation.
gpt2 = ROOT / '00 Учебник/06 Encoder, Decoder и Encoder-Decoder/05 GPT-2 — zero-shot через язык.md'
gpt2_blocks = re.findall(r'```python\n(.*?)\n```', gpt2.read_text(), re.S)
fixture = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(gpt2_blocks[1], str(gpt2), 'exec'), fixture)
network = fixture['TinyGPT2'](length=128).double().eval()
parameters_before = [p.detach().clone() for p in network.parameters()]
prefix_ids = torch.arange(80)[None, :] % 17
questions = [(torch.arange(20)[None, :] + offset) % 17 for offset in (4, 9)]
with torch.no_grad():
    _, shared = network(prefix_ids)
    saved = [(k.clone(), v.clone()) for k, v in shared]
    branch_caches = []
    for question in questions:
        whole, whole_cache = network(torch.cat((prefix_ids, question), dim=1))
        branch, branch_cache = network(question, past=shared)
        torch.testing.assert_close(branch, whole[:, 80:], rtol=1e-10, atol=1e-10)
        for (old_k, old_v), (full_k, full_v), (k, v), (saved_k, saved_v) in zip(
                shared, whole_cache, branch_cache, saved):
            torch.testing.assert_close(old_k, full_k[:, :, :80], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(old_v, full_v[:, :, :80], rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(k, full_k, rtol=1e-10, atol=1e-10)
            torch.testing.assert_close(v, full_v, rtol=1e-10, atol=1e-10)
            assert torch.equal(old_k, saved_k) and torch.equal(old_v, saved_v)
        branch_caches.append(branch_cache)
    assert not torch.allclose(branch_caches[0][-1][0][:, :, 80:],
                              branch_caches[1][-1][0][:, :, 80:], atol=1e-8)
    wrong, _ = network(questions[1], past=branch_caches[0])
    assert (wrong - branch).abs().max().item() > 1e-6
for before, after in zip(parameters_before, network.parameters()):
    assert torch.equal(before, after)
assert 100 * (80 + 20) == 10_000 and 80 + 100 * 20 == 2080
print('PASS shared prefix invariant, two full/cached branches, immutable parameters and wrong-query negative control')
print('PASS full/local reachability and 72 versus 57 allowed attention pairs')

# Bind the labels and values to the actual source row. Checking only the sum
# of 2209 and 2944 would miss swapping dirty and clean columns.
pdf = ROOT / '05 Источники/Originals/gpt3-2026-10-22/gpt3-v4.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest() == '97fd272f1fdfc18677462d0292f5fbf26ca86b4d1b485c2dba03269b643a0e83'
page = subprocess.run(['pdftotext', '-f', '45', '-l', '45', '-layout', str(pdf), '-'],
                      check=True, capture_output=True, text=True).stdout
row = re.search(r'^LAMBADA\s+test\s+acc\s+13\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)%', page, re.M)
assert row is not None
total_acc, total_n, dirty_acc, dirty_n, clean_acc, clean_n, clean_percent = row.groups()
assert (total_acc, total_n, dirty_acc, dirty_n, clean_acc, clean_n, clean_percent) == (
    '86.4', '5153', '86.9', '2209', '86.0', '2944', '57')
assert re.search(r'фильтр оставил 2944 из 5153.*?На них accuracy составила 86\.0%', text, re.S)
assert re.search(r'Остальные 2209 примеров с пересечениями имеют accuracy\s+86\.9%', text)
assert 'фильтр оставил 2209' not in text
note = (ROOT / 'Papers/GPT 3.0.md').read_text()
for fact in ['31 автор', '2020-05-28', '22 июля 2020', '2944 чистых примера с accuracy 86.0%',
             '2209 пересекающихся с accuracy 86.9%', 'разница 5.7 пункта',
             '10.2 процентного пункта', '82.8%', 'звёздочкой', '5140', '80.4%']:
    assert fact in note, fact
assert '45 авторов' not in note and 'впервые задачи задаются' not in note
print('PASS source Table C.1 dirty/clean columns, corrected legacy protocols and dates')
print('PASS chapter SHA256', hashlib.sha256(chapter.read_bytes()).hexdigest())
print('Toy calculations only; no GPT-3 weights or benchmark reproduction.')
