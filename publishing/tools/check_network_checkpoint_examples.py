#!/usr/bin/env python3
"""Executable examples for chapter 44g; no GPU or storage performance claims.

Run with CPU torch and NumPy installed. Everything written by these tests lives in
TemporaryDirectory. Includes real 2-rank save -> 3-rank load with Gloo/DTensor
and FSDP2/Adam. Requires NumPy for distributed metadata object collectives;
local atomic-generation tests are a separate application protocol, not DCP.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
CHAPTER = ROOT / "00 Учебник/11 Pre-training и Scaling/44g Network, storage и distributed checkpoints.md"


def example(text, name):
    pattern = (r"<!-- example: " + re.escape(name)
               + r" -->\s*(?P<fence>\x60{3}|~{3})python\n(?P<code>.*?)\n(?P=fence)")
    matches = list(re.finditer(pattern, text, flags=re.S))
    assert len(matches) == 1, (name, len(matches))
    return compile(matches[0]["code"], f"{CHAPTER}:{name}", "exec")


def namespace_from_chapter():
    text = CHAPTER.read_text(encoding="utf-8")
    names = (
        "network-arithmetic", "state-dict-alias", "dcp-training-state",
        "dcp-async-save", "checkpoint-queue",
        "dcp-vector-reshard", "dcp-resume-roundtrip",
        "dcp-fsdp-prepare",
    )
    ns = {"__name__": "bookvar_examples"}
    for name in names:
        exec(example(text, name), ns)
    return ns, names


def arithmetic(ns):
    checks = {
        "bandwidth": 50_000_000_000,
        "tensor_bytes": 350_000_000_000,
        "sent_per_rank": 699_316_406_250,
        "bandwidth_time": 13.986328125,
        "ring_time": 13.990420125,
        "small_us": 2.08192,
        "large_us": 85.88608,
        "injected_tb": 716.1,
        "fast_seconds": 20,
        "shared_seconds": 80,
        "full_buffer_bytes": 4_000_000_000_000,
        "per_node_two_buffers": 62_500_000_000,
    }
    for key, expected in checks.items():
        assert math.isclose(ns[key], expected, rel_tol=1e-12, abs_tol=1e-12), (
            key, ns[key], expected
        )
    assert ns["queue_after_arrival"] == [
        2_000_000_000_000, 2_500_000_000_000, 3_000_000_000_000,
        3_500_000_000_000, 4_000_000_000_000,
    ]
    # Equality is bounded in the deterministic recurrence, but has no headroom.
    queued = 0
    for _ in range(100):
        queued = max(0, queued - 2_000_000_000_000) + 2_000_000_000_000
        assert queued == 2_000_000_000_000
    # Check the prose's bottleneck/resharding examples independently.
    assert 4 * 400 / (2 * 400) == 2
    assert 2 * 400 / 4 == 200
    old = [(0, 6), (6, 12)]
    new = [(0, 4), (4, 8), (8, 12)]
    intersections = [
        [max(0, min(b, d) - max(a, c)) for a, b in old] for c, d in new
    ]
    assert intersections == [[4, 0], [2, 2], [0, 4]]
    vectors = [[1, 2], [3, 4], [5, 6], [7, 8]]
    local = [[sum(vectors[2 * node + rank][i] for rank in range(2))
              for i in range(2)] for node in range(2)]
    assert local == [[4, 6], [12, 14]]
    global_fragments = [sum(node[i] for node in local) for i in range(2)]
    assert global_fragments == [16, 20]
    return {**checks, "queue_TB": [q / 1e12 for q in ns["queue_after_arrival"]],
            "reshard_intersections": intersections,
            "hierarchical_local_sums": local, "hierarchical_result": global_fragments}


def tree_equal(actual, expected, path="state"):
    import torch
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            tree_equal(actual[key], expected[key], path + "." + str(key))
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            tree_equal(a, b, path + f"[{i}]")
    else:
        assert actual == expected, (path, actual, expected)


def dcp_resume(ns, root):
    import torch
    import torch.distributed.checkpoint as dcp
    make, step = ns["make_training"], ns["train_step"]
    visible = ns["verify_resume"](root / "visible-resume-roundtrip")
    assert visible["step"] == visible["cursor"] == 3
    # Deterministic data do not consume the training RNG.
    base = torch.arange(18, dtype=torch.double).reshape(6, 3) / 13
    batches = [
        (base + i / 10, torch.arange(6, dtype=torch.double).reshape(6, 1) / 7 - i / 9)
        for i in range(5)
    ]
    state = make(7)
    step(state, batches)
    step(state, batches)
    before = copy.deepcopy(state.state_dict())
    path = root / "dcp-step2"
    dcp.save({"training": state}, checkpoint_id=path, no_dist=True)
    expected_loss = step(state, batches)
    expected_after = copy.deepcopy(state.state_dict())

    restored = make(999)
    dcp.load({"training": restored}, checkpoint_id=path, no_dist=True)
    tree_equal(restored.state_dict(), before)
    restored_loss = step(restored, batches)
    torch.testing.assert_close(restored_loss, expected_loss, rtol=0, atol=0)
    tree_equal(restored.state_dict(), expected_after)

    negative = {}
    for omission in ("rng", "optimizer", "cursor", "scheduler"):
        candidate = make(321)
        dcp.load({"training": candidate}, checkpoint_id=path, no_dist=True)
        if omission == "rng":
            torch.manual_seed(222)
        elif omission == "optimizer":
            candidate.optimizer = torch.optim.Adam(candidate.model.parameters(), lr=0.0081)
            candidate.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                candidate.optimizer, gamma=0.9
            )
        elif omission == "cursor":
            candidate.cursor = 0
        else:
            candidate.scheduler.last_epoch = 0
        got_loss = step(candidate, batches)
        got_after = candidate.state_dict()
        try:
            tree_equal(got_after, expected_after)
        except AssertionError:
            negative[omission] = "detected"
        else:
            raise AssertionError(f"negative control {omission} was not detected")
        # RNG/cursor must change this next loss; fresh Adam changes the update.
        if omission in ("rng", "cursor"):
            assert not torch.equal(got_loss, expected_loss), omission

    # Hold the real DCP writer until after the live state has changed:
    # this proves the async path writes the staged snapshot, not an alias.
    import threading
    from unittest.mock import patch
    import torch.distributed.checkpoint.filesystem as filesystem
    original_write = filesystem.FileSystemWriter.write_data
    writer_entered = threading.Event()
    allow_write = threading.Event()

    def delayed_write(writer, *args, **kwargs):
        writer_entered.set()
        if not allow_write.wait(15):
            raise TimeoutError("test writer was not released")
        return original_write(writer, *args, **kwargs)

    async_state = make(13)
    step(async_state, batches)
    staged_expected = copy.deepcopy(async_state.state_dict())
    async_path = root / "dcp-async"
    with patch.object(filesystem.FileSystemWriter, "write_data", delayed_write):
        future = ns["save_in_background"](async_state, async_path)
        try:
            assert writer_entered.wait(15)
            assert not future.done()
            step(async_state, batches)
        finally:
            allow_write.set()
        future.result(timeout=20)
    loaded_async = make(777)
    dcp.load({"training": loaded_async}, checkpoint_id=async_path, no_dist=True)
    tree_equal(loaded_async.state_dict(), staged_expected)

    return {
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "resume_loss": expected_loss.item(),
        "loss_abs_error": abs((restored_loss - expected_loss).item()),
        "model_optimizer_scheduler_rng_step_cursor": "bitwise equal",
        "negative_controls": negative,
        "async_snapshot_after_live_mutation": "exactly equals staged step",
        "async_writer_held_until_after_mutation": True,
        "scope": "CPU, one process; real DCP filesystem writer",
        "visible_roundtrip": visible,
    }


def fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_synced(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def prepare_generation(root, name, payloads, step):
    """Local single-writer test protocol, not part of DCP."""
    folder = root / name
    folder.mkdir()
    manifest = {"schema": 1, "step": step, "files": {}}
    for filename, payload in payloads.items():
        write_synced(folder / filename, payload)
        manifest["files"][filename] = {
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()
        }
    write_synced(folder / "manifest.json", json.dumps(manifest, sort_keys=True).encode())
    fsync_directory(folder)
    fsync_directory(root)
    return folder


def validate_generation(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["schema"] == 1
    for name, metadata in manifest["files"].items():
        payload = (folder / name).read_bytes()
        if len(payload) != metadata["bytes"]:
            raise ValueError("size mismatch: " + name)
        if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
            raise ValueError("checksum mismatch: " + name)
    return manifest


def publish(root, name):
    validate_generation(root / name)
    pointer = root / "latest.tmp"
    write_synced(pointer, (name + "\n").encode())
    os.replace(pointer, root / "latest")
    fsync_directory(root)


def read_latest(root):
    # Capture pointer once; all following reads use this immutable generation.
    name = (root / "latest").read_text().strip()
    return name, validate_generation(root / name)


def local_generations(root):
    root.mkdir()
    payloads = {"rank0.bin": b"abcdef", "rank1.bin": b"123456"}
    prepare_generation(root, "step-1", payloads, 1)
    publish(root, "step-1")
    assert read_latest(root)[1]["step"] == 1

    # Interrupted creation cannot change latest.
    incomplete = root / "step-2-interrupted"
    incomplete.mkdir()
    write_synced(incomplete / "rank0.bin", b"partial")
    assert read_latest(root)[0] == "step-1"
    try:
        publish(root, incomplete.name)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing manifest accepted")

    # A complete new generation also remains invisible until pointer publication.
    prepare_generation(root, "step-2", payloads, 2)
    assert read_latest(root)[0] == "step-1"
    publish(root, "step-2")
    assert read_latest(root)[1]["step"] == 2

    errors = {}
    for case in ("checksum", "missing", "size"):
        folder = prepare_generation(root, "bad-" + case, payloads, 3)
        target = folder / "rank1.bin"
        if case == "missing":
            target.unlink()
        else:
            # deliberate corruption in this test-only temporary directory
            target.write_bytes(b"654321" if case == "checksum" else b"short")
        try:
            publish(root, folder.name)
        except (ValueError, FileNotFoundError) as exc:
            errors[case] = type(exc).__name__
        else:
            raise AssertionError("invalid generation was published: " + case)
        assert read_latest(root)[0] == "step-2"

    return {
        "interruption_before_manifest": "old generation retained",
        "interruption_before_pointer": "old generation retained",
        "negative_controls": errors,
        "scope": "local single-writer protocol; no power-failure test, no object-store test",
    }


def distributed_worker(rank, world, phase, store, checkpoint):
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    import torch.distributed.checkpoint as dcp

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(store).as_uri(), rank=rank, world_size=world,
        timeout=timedelta(seconds=25),
    )
    try:
        ns = {"__name__": "bookvar_reshard"}
        exec(example(CHAPTER.read_text(), "dcp-vector-reshard"), ns)
        tensor = ns["reshard_vector"](phase, checkpoint)
        assert tensor.size() == (12,)
    finally:
        dist.destroy_process_group()


def real_resharding(root):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    if not dist.is_gloo_available():
        raise RuntimeError("Gloo unavailable; distributed resharding has NOT been validated")
    checkpoint = root / "reshard-two-to-three"
    for phase, world in (("save", 2), ("load", 3)):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--distributed-phase", phase, "--world", str(world),
            "--store", str(root / ("rendezvous-" + phase)),
            "--checkpoint", str(checkpoint),
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=65)
        if result.returncode:
            raise AssertionError(
                f"distributed {phase} failed\n{result.stdout}\n{result.stderr}"
            )
    # Reading the same distributed representation into an ordinary CPU tensor.
    target = {"vector": torch.full((12,), -1.0, dtype=torch.float64)}
    dcp.load(target, checkpoint_id=checkpoint, no_dist=True)
    torch.testing.assert_close(target["vector"], torch.arange(12, dtype=torch.float64),
                               rtol=0, atol=0)
    return {
        "saved_world_size": 2, "loaded_world_size": 3, "elements": 12,
        "new_local_ranges": [[0, 4], [4, 8], [8, 12]],
        "also_loaded_full_tensor_single_process": True,
        "scope": "real CPU Gloo processes and DTensor; local filesystem; not FSDP/TP/GPU",
    }


def fsdp_restore_worker(rank, world, phase, store, checkpoint):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(store).as_uri(), rank=rank, world_size=world,
        timeout=timedelta(seconds=40),
    )
    try:
        ns, _ = namespace_from_chapter()
        State, step = ns["TrainingState"], ns["train_step"]

        def model(seed):
            torch.manual_seed(seed)
            # No stochastic layer: changing world size need not preserve dropout masks.
            return torch.nn.Sequential(
                torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 1)
            ).double()

        net = model(7 if phase == "save" else 999)
        mesh = init_device_mesh("cpu", (world,))
        state = ns["prepare_fsdp_training"](net, mesh)

        ref_net = model(7)
        ref_optimizer = torch.optim.Adam(ref_net.parameters(), lr=.01)
        reference = State(ref_net, ref_optimizer,
                          torch.optim.lr_scheduler.ExponentialLR(ref_optimizer, gamma=.9))
        base = torch.arange(18, dtype=torch.double).reshape(6, 3) / 13
        full_batches = [
            (base + i / 10, torch.arange(6, dtype=torch.double).reshape(6, 1) / 7 - i / 9)
            for i in range(5)
        ]
        width = 6 // world
        rows = slice(rank * width, (rank + 1) * width)
        local_batches = [(x[rows], y[rows]) for x, y in full_batches]

        if phase == "save":
            for _ in range(2):
                step(state, local_batches)
            expected_steps = 2
        else:
            dcp.load({"training": state}, checkpoint_id=checkpoint)
            assert state.step == state.cursor == 2
            step(state, local_batches)
            expected_steps = 3
        for _ in range(expected_steps):
            step(reference, full_batches)

        actual, expected = state.state_dict(), reference.state_dict()
        assert actual["step"] == actual["cursor"] == expected_steps
        assert actual["scheduler"] == expected["scheduler"]
        assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]

        def compare(value, target):
            if isinstance(value, DTensor):
                value = value.full_tensor()
            torch.testing.assert_close(value, target, rtol=1e-11, atol=1e-12)

        assert actual["model"].keys() == expected["model"].keys()
        for name, value in actual["model"].items():
            compare(value, expected["model"][name])
        assert actual["optimizer"]["state"].keys() == expected["optimizer"]["state"].keys()
        for name, moments in actual["optimizer"]["state"].items():
            for key, value in moments.items():
                compare(value, expected["optimizer"]["state"][name][key])
        if phase == "save":
            dcp.save({"training": state}, checkpoint_id=checkpoint)
        if rank == 0:
            print("PASS FSDP2 DCP", phase, world, "ranks; model/Adam/scheduler/step/cursor", flush=True)
    finally:
        dist.destroy_process_group()


def real_fsdp_resume(root):
    checkpoint = root / "fsdp-two-to-three"
    for phase, world in (("save", 2), ("load", 3)):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--distributed-phase", phase, "--world", str(world),
            "--store", str(root / ("fsdp-rendezvous-" + phase)),
            "--checkpoint", str(checkpoint), "--fsdp-checkpoint",
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=90)
        if result.returncode:
            raise AssertionError(
                f"FSDP2 DCP {phase} failed\n{result.stdout}\n{result.stderr}"
            )
        assert "PASS FSDP2 DCP" in result.stdout
    return {
        "saved_world_size": 2, "loaded_world_size": 3, "saved_step": 2, "resumed_step": 3,
        "checked": ["model", "Adam moments/steps/groups", "scheduler", "step", "cursor"],
        "reference": "unsharded FP64 training on the same global batches",
        "scope": "real CPU FSDP2/Gloo/DCP; no TP/EP/HSDP/GPU or stochastic trajectory promise",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed-phase", choices=["save", "load"])
    parser.add_argument("--world", type=int)
    parser.add_argument("--store")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fsdp-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.distributed_phase:
        import torch.multiprocessing as mp
        worker = fsdp_restore_worker if args.fsdp_checkpoint else distributed_worker
        mp.spawn(worker, args=(
            args.world, args.distributed_phase, args.store, args.checkpoint
        ), nprocs=args.world, join=True)
        return
    ns, names = namespace_from_chapter()
    import torch
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="bookvar-network-checkpoint-") as tmp:
        root = Path(tmp)
        result = {
            "markdown_sha256": hashlib.sha256(CHAPTER.read_bytes()).hexdigest(),
            "executed_markdown_blocks": list(names),
            "arithmetic": arithmetic(ns),
            "dcp_resume": dcp_resume(ns, root),
            "local_generation_protocol": local_generations(root / "generations"),
            "distributed_resharding": real_resharding(root),
            "fsdp_checkpoint_resume": real_fsdp_resume(root),
            "not_tested": [
                "GPU/NCCL/RDMA throughput or overlap", "network congestion",
                "multi-host filesystems", "power loss or hardware fault durability",
                "object-store publication", "TP/EP/HSDP/GPU optimizer resharding",
                "identical stochastic trajectories across world-size changes",
            ],
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("PASS: 44g arithmetic, snapshots, local protocol, DCP resume and CPU resharding")


if __name__ == "__main__":
    main()
