"""Execute both actual chapter 44f Python fences on two CPU/Gloo ranks.

uv run --offline --with torch==2.14.1 python publishing/tools/check_expert_parallel_distributed.py
No CUDA/NCCL, Megatron runtime, hardware timing or memory measurements.
"""
from datetime import timedelta
from pathlib import Path
import hashlib
import json
import os
import re
import socket
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44f Expert и hybrid parallelism.md"


def close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def dense(x, weights, choices, gates):
    every_assignment = torch.einsum("ti,tkij->tkj", x, weights[choices])
    return (every_assignment * gates.unsqueeze(-1)).sum(dim=1)


def check_actual_example(namespace, rank):
    full_x = namespace["inputs"].clone().requires_grad_()
    full_w = namespace["weights"].clone().requires_grad_()
    full_g = namespace["coefficients"].clone().requires_grad_()
    ref = dense(full_x, full_w, namespace["choices"], full_g)
    close(ref, torch.tensor([[1., .25], [0., 1.], [1.5, 1.5], [2., .6]],
                            dtype=torch.float64))
    ref.sum().backward()
    part = slice(2 * rank, 2 * rank + 2)
    close(namespace["y"], ref[part])
    close(namespace["x"].grad, full_x.grad[part])
    close(namespace["gates"].grad, full_g.grad[part])
    close(namespace["local_weights"].grad, full_w.grad[part])


def check_case(namespace, rank, case):
    torch.manual_seed(44123)
    all_x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.]],
                         dtype=torch.float64)
    all_w = torch.randn(4, 2, 2, dtype=torch.float64)
    choices = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 2]])
    fixed_gates = torch.tensor([[.75, .25], [.6, .4], [.5, .5], [.2, .8]],
                               dtype=torch.float64)
    if case == "empty-destination":
        choices = torch.tensor([[0, 1]] * 4)
    tokens = (slice(0, 0) if rank == 0 else slice(0, 4)) if case == "empty-sender" else slice(2 * rank, 2 * rank + 2)
    expert_part = slice(2 * rank, 2 * rank + 2)
    local_x = all_x[tokens].clone().requires_grad_()
    local_w = all_w[expert_part].clone().requires_grad_()
    ref_x = all_x.clone().requires_grad_()
    ref_w = all_w.clone().requires_grad_()

    local_router = ref_router = None
    if case == "learned-router":
        initial_router = torch.tensor([[1., -.5, .2, -1.], [.1, 1.5, -.3, .7]],
                                      dtype=torch.float64)
        local_router = initial_router.clone().requires_grad_()
        ref_router = initial_router.clone().requires_grad_()
        local_scores, local_choices = torch.topk(local_x @ local_router, 2, dim=1)
        local_g = local_scores.softmax(dim=1)
        ref_scores, ref_choices = torch.topk(ref_x @ ref_router, 2, dim=1)
        ref_g = ref_scores.softmax(dim=1)
        # All selected/non-selected boundaries are separated in this fixture.
        sorted_logits = (ref_x @ ref_router).sort(dim=1, descending=True).values
        assert torch.all(sorted_logits[:, 1] - sorted_logits[:, 2] > .01)
    else:
        local_choices, ref_choices = choices[tokens], choices
        local_g = fixed_gates[tokens].clone().requires_grad_()
        ref_g = fixed_gates.clone().requires_grad_()

    result = namespace["expert_parallel_linear"](local_x, local_choices, local_g, local_w)
    reference = dense(ref_x, ref_w, ref_choices, ref_g)
    close(result, reference[tokens])
    # Every rank contributes its local term; jointly this is one global SUM.
    (result.square().sum() / 2).backward()
    (reference.square().sum() / 2).backward()
    close(local_x.grad, ref_x.grad[tokens])
    close(local_w.grad, ref_w.grad[expert_part])
    if local_router is not None:
        dist.all_reduce(local_router.grad)
        close(local_router.grad, ref_router.grad)
    else:
        close(local_g.grad, ref_g.grad[tokens])
    if case == "empty-destination" and rank == 1:
        assert torch.count_nonzero(local_w.grad) == 0


def worker(rank, port, fences, init_file):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    namespace = {"__name__": "__main__"}
    exec(compile(fences[0], str(CHAPTER) + ":fence1", "exec"), namespace)
    # The actual second fence performs init, the worked forward/backward,
    # and destroy_process_group; do not replace it with a test-only driver.
    exec(compile(fences[1], str(CHAPTER) + ":fence2", "exec"), namespace)
    check_actual_example(namespace, rank)
    dist.init_process_group("gloo", init_method="file://" + init_file,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        for case in ["uneven", "empty-destination", "empty-sender", "learned-router"]:
            check_case(namespace, rank, case)
            if rank == 0:
                print("PASS actual two-rank Gloo forward/backward:", case, flush=True)
    finally:
        dist.destroy_process_group()


def main():
    source = CHAPTER.read_text()
    fences = re.findall(r"~~~python\n(.*?)~~~", source, flags=re.S)
    assert len(fences) == 2
    for fence in fences:
        compile(fence, str(CHAPTER), "exec")
    assert len(re.findall(r"!\[\[", source)) == 4
    for marker in ["lecture-08-width-parallelism", "lecture-08-model-strategy-cases",
                   'id="dispatch-compute-combine"', 'id="tp-или-ep"',
                   "31.5 MiB", "21 GB", "57 ms", "ETP=1, EP=64"]:
        assert marker in source, marker
    logits = torch.tensor([3., 1.], dtype=torch.float64).log().requires_grad_()
    outputs = torch.tensor([[1., 0.], [1., 1.]], dtype=torch.float64)
    gates = logits.softmax(dim=0)
    loss = (gates @ outputs) @ torch.tensor([1., 2.], dtype=torch.float64)
    loss.backward()
    close(logits.grad, torch.tensor([-.375, .375], dtype=torch.float64))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="bookvar-ep-gloo-") as directory:
        mp.spawn(worker, args=(port, fences, str(Path(directory) / "rendezvous")), nprocs=2)
    print("PASS both actual Python fences, original outputs, input/gate/expert/router gradients")
    print(json.dumps({"torch": torch.__version__, "processes": 2,
                      "actual_python_fences": len(fences), "figure_count": 4,
                      "chapter_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest()}, indent=2))
    print("CUDA/NCCL, Megatron, timings and GPU peak memory NOT tested.")


if __name__ == "__main__":
    main()
