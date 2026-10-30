"""Execute S6's CPU snippets and check their data/attention invariants.

Requires PyTorch. Does not benchmark CUDA, pinned memory, or multiprocessing.
"""
from pathlib import Path
import ast
from fractions import Fraction
from hashlib import sha256
import math
import re

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/10 ML Systems/06 Data pipeline, padding и packing.md"


def main():
    text = CHAPTER.read_text()
    # The independent tf.data addition is executed by its TensorFlow checker.
    # Preserve every prior PyTorch/packing check below, without importing TF.
    text = re.sub(
        r"<!-- BEGIN REVIEW: tf.data cache 2026-10-28 -->\n.*?"
        r"<!-- END REVIEW: tf.data cache 2026-10-28 -->\n\n", "", text, flags=re.S)
    blocks = re.findall(r"```python\n(.*?)\n```", text, re.S)
    assert len(blocks) == 4, "Review changed chapter examples before updating this checker"
    for block in blocks:
        ast.parse(block)

    namespace = {}
    exec(compile(blocks[0], str(CHAPTER), "exec"), namespace)
    exec(compile(blocks[1], str(CHAPTER), "exec"), namespace)
    exec(compile(blocks[2], str(CHAPTER), "exec"), namespace)
    matrix = re.search(r"\\begin\{array\}\{ccc\|cccc\}(.*?)\\end\{array\}", text, re.S).group(1)
    rows = matrix.replace(r"\hline", "").strip().split(r"\\")
    shown_mask = [[int(x.strip()) for x in row.split("&")] for row in rows if row.strip()]
    assert shown_mask == namespace["allow"].to(torch.int).tolist()
    collate = namespace["collate_lm"]

    # PAD == EOS must not suppress a real end-of-document target.
    batch = namespace["batch"]
    assert (batch["labels"][:, 1:] != -100).sum().item() == 3
    assert batch["labels"][0, 2].item() == 2
    assert batch["labels"][1, 1].item() == 2
    for rows in ([], [{"input_ids": []}], [{"input_ids": [1]}], [{"input_ids": [[1, 2]]}]):
        try:
            collate(rows)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid shape/length accepted")
    variable = collate([{"input_ids": [1, 2]}, {"input_ids": [3, 4, 5, 2]}], pad_id=2)
    assert variable["attention_mask"].sum().item() == 6
    assert (variable["labels"][:, 1:] != -100).sum().item() == 4
    print("PASS: chapter collator; real EOS preserved; shifted target counts")

    lengths = [128, 140, 160, 1024]
    assert sum(lengths) == 1452
    assert len(lengths) * max(lengths) - sum(lengths) == 2644
    assert sum(x*x for x in lengths) == 1110160
    assert len(lengths)*max(lengths)**2 == 4194304
    assert sum(x*(x+1)//2 for x in lengths) == 555806
    assert len(lengths)*max(lengths)*(max(lengths)+1)//2 == 2099200
    lengths = [8, 64, 9, 60, 10, 56, 12, 52]
    slots = lambda xs: sum(2 * max(xs[i:i + 2]) for i in range(0, len(xs), 2))
    assert (sum(lengths), slots(lengths), slots(sorted(lengths))) == (271, 464, 282)
    assert [sum([7, 10, 8, 5][:i]) for i in range(5)] == [0, 7, 17, 25, 30]
    print("PASS: padding, bucketing, cumulative lengths")

    torch.testing.assert_close(namespace["packed"].flatten(),
        torch.tensor([1, 1.5, 2, 10, 15, 20, 25], dtype=torch.float64))
    altered = namespace["attend"](namespace["q"], namespace["k"], namespace["changed_v"])
    assert altered[0,0,3,0].item() == 79
    # Nonuniform scores and gradients complement the chapter's uniform example.
    for lengths in ([1,1], [2,3], [3,4], [1,2,4], [4,1,2]):
        torch.manual_seed(sum(lengths))
        total = sum(lengths)
        segments = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        mask = segments[:,None].eq(segments[None,:]) & torch.ones(total,total,dtype=torch.bool).tril()
        tensors = [torch.randn(1,2,total,3,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        packed = F.scaled_dot_product_attention(*tensors,attn_mask=mask,dropout_p=0)
        parts, offset = [], 0
        for length in lengths:
            parts.append(F.scaled_dot_product_attention(*[x[:,:,offset:offset+length] for x in tensors],
                is_causal=True,dropout_p=0))
            offset += length
        separate = torch.cat(parts,dim=2)
        torch.testing.assert_close(packed,separate,rtol=1e-12,atol=1e-12)
        gp = torch.autograd.grad(packed.square().sum(),tensors,retain_graph=True)
        gs = torch.autograd.grad(separate.square().sum(),tensors)
        for a,b in zip(gp,gs):
            torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    print("PASS: actual packed-attention block; 4/79 negative controls; five forward/backward partitions")

    # Test the mask taken directly from the chapter against real CPU SDPA.
    torch.manual_seed(17)
    q, k, v = [torch.randn(1, 1, 7, 4, dtype=torch.float64) for _ in range(3)]
    allow = namespace["allow"]
    packed = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=0)
    parts = [F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b],
             is_causal=True, dropout_p=0) for a, b in [(0, 3), (3, 7)]]
    torch.testing.assert_close(packed, torch.cat(parts, dim=2), rtol=1e-12, atol=1e-12)
    changed_v = v.clone()
    changed_v[:, :, :3] += 10
    isolated = F.scaled_dot_product_attention(q, k, changed_v, attn_mask=allow, dropout_p=0)
    torch.testing.assert_close(packed[:, :, 3:], isolated[:, :, 3:], rtol=0, atol=0)
    plain = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
    leaking = F.scaled_dot_product_attention(q, k, changed_v, is_causal=True, dropout_p=0)
    assert not torch.allclose(plain[:, :, 3:], leaking[:, :, 3:])
    print("PASS: packed vs separate attention; isolation; causal-only negative control")

    labels = torch.tensor([-100, 12, 2, -100, 22, 23, 2])
    assert labels[1:].ne(-100).sum().item() == 5
    assert labels[3].item() == -100  # No prediction of B's first token from A's EOS.
    assert math.isclose((100 * 0.5 + 10 * 2) / 110, 7 / 11)
    parameter = torch.tensor(1.0, requires_grad=True)
    for size, loss in [(100, 0.5), (10, 2.0)]:
        (parameter * size * loss / 110).backward()
    torch.testing.assert_close(parameter.grad, torch.tensor(7 / 11))
    print("PASS: packed label boundary and token-weighted accumulation")

    # Ordinary DDP averaging, simulated in CPU algebra rather than a cluster run.
    for sizes in ([100,10], [1,7], [0,3], [2,3,5], [1,2,3,8]):
        r, n = len(sizes), sum(sizes)
        weights = [Fraction(i+1,2) for i in range(r)]
        expected = sum(size*weight for size,weight in zip(sizes,weights))/n
        p = torch.tensor(1.,dtype=torch.float64,requires_grad=True)
        local = [p*r*size*float(weight)/n for size,weight in zip(sizes,weights)]
        torch.stack(local).mean().backward()
        assert math.isclose(p.grad.item(),float(expected),rel_tol=1e-14)
    sampler = torch.utils.data.DistributedSampler
    assert [list(sampler(range(5),2,r,shuffle=False,drop_last=False)) for r in range(2)] == [[0,2,4],[1,3,0]]
    assert [list(sampler(range(5),2,r,shuffle=False,drop_last=True)) for r in range(2)] == [[0,2],[1,3]]
    for r in range(2):
        indices = sampler(range(5),2,r,shuffle=False,drop_last=False)
        loader = torch.utils.data.DataLoader(list(indices),batch_size=2,drop_last=True)
        assert [b.tolist() for b in loader] == [[r,r+2]]
    readers = [list(range(reader,10,4)) for reader in range(4)]
    assert readers == [[0,4,8],[1,5,9],[2,6],[3,7]]
    assert sorted(x for reader in readers for x in reader) == list(range(10))
    assert (28+(3-1)*max(28,20)+20,3*(28+20)) == (104,144)
    for ptime in (7,20,28,31):
        for gtime in (9,20,28,41):
            for count in range(1,9):
                cpu_end, gpu_end = 0, 0
                for _ in range(count):
                    cpu_end += ptime
                    gpu_end = max(cpu_end,gpu_end)+gtime
                assert gpu_end == ptime+(count-1)*max(ptime,gtime)+gtime
    source = iter('abcdef')
    buffer = [next(source),next(source)]
    consumed = []
    for _ in range(2):
        consumed.append(buffer[0])
        buffer[0] = next(source)
    assert consumed == ['a','c'] and buffer == ['d','b'] and next(source) == 'e'
    print("PASS: DDP weighting algebra; real sampler/drop_last; four-reader sharding; two-stage timing")
    large_rgb = Fraction(32*640*640*3, 2**20)
    small_rgb = Fraction(32*224*224*3, 2**20)
    assert large_rgb == Fraction(75, 2) and small_rgb == Fraction(147, 32)
    assert f"{float(small_rgb):.2f}" == '4.59'
    assert f"{float(4*small_rgb):.2f}" == '18.38'
    for value in ('37.5 МиБ', '4.59 МиБ', '18.38 МиБ'):
        assert value in text
    print("PASS: RGB batch sizes; byte/FLOAT32 transfer example, not a decoder benchmark")
    print("chapter_sha256="+sha256(CHAPTER.read_bytes()).hexdigest())
    print(f"PyTorch {torch.__version__}; CPU checks only; CUDA loop compiled as Python, not executed")


if __name__ == "__main__":
    main()
