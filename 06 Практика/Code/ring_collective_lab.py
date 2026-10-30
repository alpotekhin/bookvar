"""Real multiprocess ring reduce-scatter/all-gather, checked against Gloo.

Bookvar solution for EDLS Week 3. CPU execution checks communication and
ownership, not GPU/network performance. No remote rendezvous is supported.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def ring_sum(tensor, rank, world):
    # Equal wire chunks; zero padding is removed after gathering.
    chunk = math.ceil(tensor.numel() / world)
    padded = torch.zeros(chunk * world, dtype=tensor.dtype)
    padded[:tensor.numel()] = tensor
    blocks = list(padded.chunk(world))
    trace = []
    for phase in ('reduce_scatter', 'all_gather'):
        for step in range(world - 1):
            offset = 0 if phase == 'reduce_scatter' else 1
            send_idx = (rank - step + offset) % world
            recv_idx = (rank - step - 1 + offset) % world
            received = torch.empty_like(blocks[recv_idx])
            # Issue both nonblocking operations before waiting: no ring deadlock.
            pending = [dist.isend(blocks[send_idx].clone(), (rank + 1) % world),
                       dist.irecv(received, (rank - 1) % world)]
            for work in pending:
                work.wait()
            if phase == 'reduce_scatter':
                blocks[recv_idx].add_(received)
            else:
                blocks[recv_idx].copy_(received)
            trace.append(dict(phase=phase, step=step, sent_chunk=send_idx,
                              received_chunk=recv_idx, received=received.tolist(),
                              resulting_chunk=blocks[recv_idx].tolist()))
    return torch.cat(blocks)[:tensor.numel()], trace


def worker(rank, world, rendezvous, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=world, timeout=timedelta(seconds=30))
    cases = []
    try:
        for dtype in (torch.int64, torch.float32):
            for n in (1, world, world + 2, 17):
                local = torch.arange(n, dtype=dtype) + rank * 10
                reference = local.clone()
                dist.all_reduce(reference)
                actual, trace = ring_sum(local, rank, world)
                expected = world * torch.arange(n, dtype=dtype) + 10 * world * (world - 1) // 2
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                cases.append(dict(n=n, dtype=str(dtype), input=local.tolist(),
                                  output=actual.tolist(), trace=trace,
                                  elements_sent=2 * (world - 1) * math.ceil(n / world)))
        (result_dir / f'rank-{rank}.json').write_text(json.dumps(cases, indent=2))
    finally:
        dist.destroy_process_group()


def run(output):
    if not dist.is_gloo_available():
        raise SystemExit('Gloo unavailable; no collective was executed.')
    results = []
    for world in (2, 3, 4):
        with tempfile.TemporaryDirectory(prefix='bookvar-ring-') as temp:
            directory = Path(temp)
            mp.spawn(worker, args=(world, (directory / 'rendezvous').as_uri(), directory),
                     nprocs=world, join=True)
            results.append(dict(world_size=world, ranks=[
                json.loads((directory / f'rank-{r}.json').read_text()) for r in range(world)]))
    report = dict(scope='local CPU Gloo correctness; no GPU or network benchmark',
                  torch=torch.__version__, configurations=24, rank_checks=72, results=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: 24 configurations / 72 rank checks, real Gloo processes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
