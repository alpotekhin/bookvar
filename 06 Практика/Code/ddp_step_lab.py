"""One DDP update compared with a global-batch reference; optional CUDA timing.

Launch with torchrun --standalone --nproc-per-node=2. CPU/Gloo validates the
gradient contract; NCCL requires two visible NVIDIA GPUs and fails otherwise.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main(backend, output):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if world < 2:
        raise SystemExit('Use torchrun --standalone --nproc-per-node=2; one rank is not this test.')
    if backend == 'nccl':
        if not torch.cuda.is_available() or torch.cuda.device_count() < world:
            raise SystemExit('At least WORLD_SIZE visible NVIDIA GPUs required. No NCCL result produced.')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    torch.set_num_threads(1)
    dist.init_process_group(backend, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(31)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).to(device)
        reference = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).to(device)
        reference.load_state_dict(model.state_dict())
        wrapped = DDP(model, device_ids=[local_rank] if backend == 'nccl' else None)
        data = torch.arange(world * 4 * 8, device=device, dtype=torch.float32).reshape(world*4, 8) / 100
        target = torch.arange(world * 4 * 2, device=device, dtype=torch.float32).reshape(world*4, 2) / 100
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=.03)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        torch.nn.functional.mse_loss(wrapped(data.chunk(world)[rank]), target.chunk(world)[rank]).backward()
        torch.nn.functional.mse_loss(reference(data), target).backward()
        errors = []
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
            errors.append(float((actual.grad-expected.grad).abs().max()))
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
        numbers = torch.tensor([rank + 1.], device=device)
        dist.all_reduce(numbers)
        assert numbers.item() == world * (world + 1) / 2
        report = dict(backend=backend, ranks=world, rank=rank,
            gradient_max_abs_error=max(errors), global_batch=world*4,
            step_matches_global_batch=True, all_reduce_sum=numbers.item(),
            scope='DDP/global-batch correctness; small fixture, not a training throughput benchmark')
        if backend == 'nccl':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(numbers)
            end.record()
            end.synchronize()
            report.update(cuda_event_allreduce_ms=start.elapsed_time(end),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          gpu=torch.cuda.get_device_name())
        output.mkdir(parents=True, exist_ok=True)
        (output / f'ddp-{backend}-rank-{rank}.json').write_text(json.dumps(report, indent=2)+'\n')
        print(f'PASS {backend} rank={rank} gradient error={max(errors):.3g}', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.backend, args.output)
