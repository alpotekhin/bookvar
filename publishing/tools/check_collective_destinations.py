"""Run actual 44a destination/split examples on four local CPU/Gloo ranks.

No CUDA, NCCL, physical overlap or communication benchmark is tested.
"""
import ast
from datetime import timedelta
import hashlib
from pathlib import Path
import re
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / '00 Учебник/11 Pre-training и Scaling/44a Processes, collectives и DDP.md'


def blocks():
    text = CHAPTER.read_text()
    section = text.split('<!-- BEGIN REVIEW: collective destinations 2026-10-27 -->')[1].split(
        '<!-- END REVIEW: collective destinations 2026-10-27 -->')[0]
    result = re.findall(r'^```python\n(.*?)^```', section, re.M | re.S)
    assert len(result) == 2
    for code in result:
        ast.parse(code)
    return result


def worker(rank, uri, code):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=uri, rank=rank, world_size=4,
                            timeout=timedelta(seconds=30))
    try:
        scope = {}
        for i, fence in enumerate(code, 1):
            exec(compile(fence, str(CHAPTER) + f':collective-destinations-{i}', 'exec'), scope)
        # Independent slice construction, rather than only literal expected arrays.
        count_table = scope['send_counts']
        assembled = []
        for source in range(4):
            offset = sum(count_table[source][:rank])
            count = count_table[source][rank]
            assembled.extend(range(10*source + offset, 10*source + offset + count))
        assert scope['output'].tolist() == assembled
        assert sum(sum(row) for row in count_table) == 27
        assert [sum(row) for row in count_table] == [6, 9, 5, 7]
        assert [sum(row[c] for row in count_table) for c in range(4)] == [9, 7, 6, 5]

        # Root need not be rank zero. Identical values are still four tensors,
        # not a single reduced value; deliberate arrival skew must not reorder them.
        item = torch.tensor([7, rank])
        gathered = [torch.empty_like(item) for _ in range(4)] if rank == 2 else None
        time.sleep(.01 * (3 - rank))
        returned = dist.gather(item, gather_list=gathered, dst=2)
        assert returned is None
        if rank == 2:
            assert torch.stack(gathered).tolist() == [[7, 0], [7, 1], [7, 2], [7, 3]]
        root_parts = [torch.tensor([2, 4, 6, 8])[i:i+1] for i in range(4)] if rank == 3 else None
        received = torch.empty(1, dtype=torch.int64)
        dist.scatter(received, scatter_list=root_parts, src=3)
        assert received.item() == 2*(rank + 1)

        # Zero-sized pieces still participate; one rank receives no payload.
        counts = [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]]
        receive_counts = [row[rank] for row in counts]
        payload = torch.full((sum(counts[rank]),), rank, dtype=torch.int64)
        output = torch.empty(sum(receive_counts), dtype=torch.int64)
        dist.all_to_all_single(output, payload, receive_counts, counts[rank])
        assert output.tolist() == [[], [0, 3], [], [1]][rank]
        if rank == 0:
            print('PASS two actual Markdown fences on four CPU/Gloo processes: scatter, gather, all-gather, broadcast, reduce, even/uneven all-to-all', flush=True)
            print('PASS independent slice/count reconstruction, nonzero roots, arrival skew, return contract and zero-sized splits', flush=True)
    finally:
        dist.destroy_process_group()


def main():
    if not dist.is_gloo_available():
        raise RuntimeError('CPU/Gloo required; no unexecuted test is marked passed')
    with tempfile.TemporaryDirectory(prefix='bookvar-collective-destinations-') as directory:
        context = mp.spawn(worker, args=((Path(directory) / 'store').as_uri(), blocks()),
                           nprocs=4, join=False)
        deadline = time.monotonic() + 90
        try:
            while not context.join(timeout=1):
                if time.monotonic() >= deadline:
                    raise TimeoutError('four-process validation exceeded 90 seconds')
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
    print('torch=' + torch.__version__ + '; no GPU/NCCL execution or timing claim')
    print('chapter_sha256=' + hashlib.sha256(CHAPTER.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
